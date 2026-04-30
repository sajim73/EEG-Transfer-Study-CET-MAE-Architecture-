#!/usr/bin/env python3
import argparse
import json
import math
import os
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

from models.load_cetmae_encoder import build_cetmae_encoder, describe_checkpoint


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_table(path: str) -> pd.DataFrame:
    for sep in [',', ';', '\t']:
        try:
            df = pd.read_csv(path, sep=sep, engine='python')
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    raise ValueError(f'Could not read table: {path}')


def normalize_text(x: str) -> str:
    if x is None:
        return ''
    x = str(x)
    x = x.replace('\u2019', "'").replace('\u2018', "'").replace('\u201c', '"').replace('\u201d', '"')
    x = x.replace('`', "'")
    return re.sub(r'\s+', ' ', x).strip().lower()


def load_records(path: str):
    ext = Path(path).suffix.lower()
    if ext in ['.pt', '.pth']:
        obj = torch.load(path, map_location='cpu')
    elif ext in ['.pkl', '.pickle']:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
    else:
        raise ValueError('Supported data formats: .pt/.pth/.pkl/.pickle')
    if isinstance(obj, dict):
        if 'records' in obj and isinstance(obj['records'], list):
            return obj['records']
        merged = []
        for _, v in obj.items():
            if isinstance(v, list):
                merged.extend(v)
        if merged:
            return merged
    if isinstance(obj, list):
        return obj
    raise ValueError('Unsupported record container format')


def pick_first(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_eeg_and_mask(rec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    eeg = pick_first(rec, ['input_embeddings', 'eeg', 'x', 'features', 'embeddings'])
    if eeg is None:
        raise KeyError('Missing EEG key')
    eeg = to_numpy(eeg).astype(np.float32)
    if eeg.ndim == 1:
        eeg = eeg[None, :]
    mask = pick_first(rec, ['input_attn_mask', 'attention_mask', 'mask', 'attn_mask'])
    if mask is None:
        mask = np.ones((eeg.shape[0],), dtype=np.int64)
    else:
        mask = to_numpy(mask).astype(np.int64).reshape(-1)
    if len(mask) != eeg.shape[0]:
        raise ValueError('Mask length does not match EEG length')
    return eeg, mask


def build_reading_label_maps(labels_csv: str):
    df = read_table(labels_csv)
    df.columns = [str(c).strip() for c in df.columns]
    required = {'paragraph_id', 'sentence_id', 'sentence', 'control'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    sent_map, pair_map = {}, {}
    for _, row in df.iterrows():
        label = 1 if str(row['control']).strip().upper() == 'CONTROL' else 0
        sent_map[normalize_text(row['sentence'])] = label
        pair_map[(str(row['paragraph_id']).strip(), str(row['sentence_id']).strip())] = label
    return sent_map, pair_map


def build_sentiment_label_maps(labels_csv: str, label_col: str, text_col: str = None, id_col: str = None):
    df = read_table(labels_csv)
    df.columns = [str(c).strip() for c in df.columns]
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}'")
    sent_map, id_map = {}, {}
    for _, row in df.iterrows():
        label = row[label_col]
        if pd.isna(label):
            continue
        if text_col is not None and text_col in df.columns:
            sent_map[normalize_text(row[text_col])] = label
        if id_col is not None and id_col in df.columns:
            id_map[str(row[id_col]).strip()] = label
    return sent_map, id_map


def attach_reading_labels(records, sent_map, pair_map):
    examples = []
    for idx, rec in enumerate(records):
        try:
            eeg, mask = get_eeg_and_mask(rec)
        except Exception:
            continue
        sentence = pick_first(rec, ['sentence', 'text', 'target_text', 'raw_text'], '')
        paragraph_id = pick_first(rec, ['paragraph_id', 'paragraph', 'para_id'], None)
        sentence_id = pick_first(rec, ['sentence_id', 'sent_id'], None)
        label = None
        if paragraph_id is not None and sentence_id is not None:
            label = pair_map.get((str(paragraph_id).strip(), str(sentence_id).strip()))
        if label is None:
            label = sent_map.get(normalize_text(sentence))
        examples.append({
            'sample_id': idx,
            'eeg': eeg,
            'mask': mask,
            'label': None if label is None else int(label),
            'sentence': sentence,
            'subject': pick_first(rec, ['subject', 'subject_id', 'subj'], 'unknown'),
            'paragraph_id': None if paragraph_id is None else str(paragraph_id).strip(),
            'sentence_id': None if sentence_id is None else str(sentence_id).strip(),
        })
    return examples


def attach_sentiment_labels(records, sent_map, id_map):
    examples = []
    raw_labels = []
    for idx, rec in enumerate(records):
        try:
            eeg, mask = get_eeg_and_mask(rec)
        except Exception:
            continue
        rid = pick_first(rec, ['ID', 'id', 'sentence_index', 'item_id'], None)
        sentence = pick_first(rec, ['sentence', 'text', 'target_text', 'raw_text'], '')
        label = None
        if rid is not None:
            label = id_map.get(str(rid).strip())
        if label is None:
            label = sent_map.get(normalize_text(sentence))
        examples.append({
            'sample_id': idx,
            'eeg': eeg,
            'mask': mask,
            'raw_label': label,
            'sentence': sentence,
            'subject': pick_first(rec, ['subject', 'subject_id', 'subj'], 'unknown'),
            'paragraph_id': pick_first(rec, ['paragraph_id', 'paragraph', 'para_id'], None),
            'sentence_id': pick_first(rec, ['sentence_id', 'sent_id'], None),
        })
        if label is not None:
            raw_labels.append(label)
    uniq = sorted(list(set([x for x in raw_labels if x is not None])))
    label2id = {lab: i for i, lab in enumerate(uniq)}
    for ex in examples:
        ex['label'] = None if ex['raw_label'] is None else int(label2id[ex['raw_label']])
    return examples, label2id


def grouped_split(examples, seed=42, train_frac=0.7, val_frac=0.15):
    groups = defaultdict(list)
    has_subject = any(str(ex.get('subject', 'unknown')) != 'unknown' for ex in examples)
    if has_subject:
        for i, ex in enumerate(examples):
            groups[str(ex.get('subject', 'unknown'))].append(i)
    else:
        for i in range(len(examples)):
            groups[str(i)].append(i)
    keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = max(1, int(round(train_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:n_train + n_val])
    test_keys = set(keys[n_train + n_val:])
    if not test_keys:
        moved = next(iter(val_keys))
        val_keys.remove(moved)
        test_keys.add(moved)

    def collect(which, split_name):
        out = []
        for k in which:
            for idx in groups[k]:
                ex = dict(examples[idx])
                ex['split'] = split_name
                out.append(ex)
        return out
    return collect(train_keys, 'train'), collect(val_keys, 'val'), collect(test_keys, 'test')


class EEGDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            'eeg': torch.tensor(ex['eeg'], dtype=torch.float32),
            'mask': torch.tensor(ex['mask'], dtype=torch.long),
            'label': -1 if ex.get('label') is None else int(ex['label']),
            'subject': ex.get('subject', 'unknown'),
            'sentence': ex.get('sentence', ''),
            'paragraph_id': ex.get('paragraph_id'),
            'sentence_id': ex.get('sentence_id'),
            'sample_id': ex.get('sample_id'),
            'split': ex.get('split', 'unknown'),
        }


def collate_fn(batch):
    max_len = max(item['eeg'].shape[0] for item in batch)
    feat_dim = batch[0]['eeg'].shape[1]
    eeg = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch),), -1, dtype=torch.long)
    meta = {k: [] for k in ['subject', 'sentence', 'paragraph_id', 'sentence_id', 'sample_id', 'split']}
    for i, item in enumerate(batch):
        L = item['eeg'].shape[0]
        eeg[i, :L] = item['eeg']
        mask[i, :L] = item['mask']
        labels[i] = int(item['label'])
        for k in meta:
            meta[k].append(item[k])
    return eeg, mask, labels, meta


def plot_pca(embeddings: np.ndarray, meta_df: pd.DataFrame, out_dir: str):
    if len(embeddings) < 2:
        return
    X2 = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    df = meta_df.copy()
    df['pca1'] = X2[:, 0]
    df['pca2'] = X2[:, 1]
    df.to_csv(Path(out_dir) / 'embedding_pca2.csv', index=False)
    for color_col in ['label_name', 'subject', 'split']:
        if color_col not in df.columns or df[color_col].isna().all():
            continue
        vals = df[color_col].fillna('NA').astype(str)
        uniq = list(pd.unique(vals))
        if len(uniq) > 20:
            vc = vals.value_counts()
            keep = set(vc.index[:19])
            vals = vals.map(lambda z: z if z in keep else 'OTHER')
            uniq = list(pd.unique(vals))
        fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
        cmap = plt.cm.get_cmap('tab20', max(2, len(uniq)))
        for i, u in enumerate(uniq):
            idx = vals == u
            ax.scatter(df.loc[idx, 'pca1'], df.loc[idx, 'pca2'], s=14, alpha=0.8, label=u, color=cmap(i))
        ax.set_title(f'PCA embeddings colored by {color_col}')
        ax.set_xlabel('pca1')
        ax.set_ylabel('pca2')
        ax.grid(alpha=0.2)
        if len(uniq) <= 20:
            ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f'embedding_pca2_{color_col}.png', bbox_inches='tight')
        plt.close(fig)


def extract_embeddings(model, loader, device, pooling='mean'):
    model.eval()
    all_embeddings, rows = [], []
    with torch.no_grad():
        for eeg, mask, labels, meta in loader:
            eeg = eeg.to(device)
            mask = mask.to(device)
            pooled = model(eeg, mask, pooling=pooling)
            emb = pooled.detach().cpu().numpy()
            all_embeddings.append(emb)
            for i in range(len(meta['subject'])):
                rows.append({
                    'sample_id': meta['sample_id'][i],
                    'subject': meta['subject'][i],
                    'sentence': meta['sentence'][i],
                    'paragraph_id': meta['paragraph_id'][i],
                    'sentence_id': meta['sentence_id'][i],
                    'split': meta['split'][i],
                    'label': int(labels[i].item()) if int(labels[i].item()) >= 0 else np.nan,
                })
    embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((0, model.output_dim), dtype=np.float32)
    meta_df = pd.DataFrame(rows)
    return embeddings.astype(np.float32), meta_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--pretrain-path', default='./models/huggingface/bart-large')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--task-type', choices=['unlabeled', 'reading', 'sentiment'], default='unlabeled')
    parser.add_argument('--labels-csv', default=None)
    parser.add_argument('--label-col', default='label')
    parser.add_argument('--text-col', default='sentence')
    parser.add_argument('--id-col', default=None)
    parser.add_argument('--pooling', choices=['mean', 'last'], default='mean')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-frac', type=float, default=0.7)
    parser.add_argument('--val-frac', type=float, default=0.15)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed(args.seed)
    save_json(Path(args.output_dir) / 'run_config.json', vars(args))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    records = load_records(args.data)
    label2id = None
    if args.task_type == 'reading':
        if args.labels_csv is None:
            raise ValueError('--labels-csv is required for reading task')
        sent_map, pair_map = build_reading_label_maps(args.labels_csv)
        examples = attach_reading_labels(records, sent_map, pair_map)
        label_name_map = {0: 'noncontrol', 1: 'control'}
    elif args.task_type == 'sentiment':
        if args.labels_csv is None:
            raise ValueError('--labels-csv is required for sentiment task')
        sent_map, id_map = build_sentiment_label_maps(args.labels_csv, args.label_col, args.text_col, args.id_col)
        examples, label2id = attach_sentiment_labels(records, sent_map, id_map)
        label_name_map = {v: str(k) for k, v in label2id.items()}
        save_json(Path(args.output_dir) / 'label_map.json', label2id)
    else:
        examples = []
        for idx, rec in enumerate(records):
            try:
                eeg, mask = get_eeg_and_mask(rec)
            except Exception:
                continue
            examples.append({
                'sample_id': idx,
                'eeg': eeg,
                'mask': mask,
                'label': None,
                'sentence': pick_first(rec, ['sentence', 'text', 'target_text', 'raw_text'], ''),
                'subject': pick_first(rec, ['subject', 'subject_id', 'subj'], 'unknown'),
                'paragraph_id': pick_first(rec, ['paragraph_id', 'paragraph', 'para_id'], None),
                'sentence_id': pick_first(rec, ['sentence_id', 'sent_id'], None),
            })
        label_name_map = {}

    train_ex, val_ex, test_ex = grouped_split(examples, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac)
    split_examples = {'train': train_ex, 'val': val_ex, 'test': test_ex}
    split_rows = []
    for split_name, arr in split_examples.items():
        for ex in arr:
            split_rows.append({
                'sample_id': ex['sample_id'], 'split': split_name, 'subject': ex['subject'],
                'paragraph_id': ex.get('paragraph_id'), 'sentence_id': ex.get('sentence_id'), 'sentence': ex.get('sentence'),
                'label': ex.get('label'),
            })
    pd.DataFrame(split_rows).to_csv(Path(args.output_dir) / 'split_manifest.csv', index=False)

    loader = DataLoader(EEGDataset(train_ex + val_ex + test_ex), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    model = build_cetmae_encoder(args.checkpoint, device=device, pretrain_path=args.pretrain_path, freeze=True)
    save_json(Path(args.output_dir) / 'encoder_summary.json', describe_checkpoint(model))
    embeddings, meta_df = extract_embeddings(model, loader, device=device, pooling=args.pooling)

    if 'label' in meta_df.columns:
        meta_df['label_name'] = meta_df['label'].map(lambda x: label_name_map.get(int(x), str(int(x))) if pd.notna(x) else 'NA')
    meta_df.to_csv(Path(args.output_dir) / 'embeddings_metadata.csv', index=False)
    np.savez_compressed(
        Path(args.output_dir) / 'embeddings.npz',
        embeddings=embeddings,
        y_true=np.asarray([-1 if pd.isna(x) else int(x) for x in meta_df.get('label', pd.Series([-1] * len(meta_df)))]),
        subject=np.asarray(meta_df.get('subject', pd.Series(['unknown'] * len(meta_df))).astype(str)),
        sentence=np.asarray(meta_df.get('sentence', pd.Series([''] * len(meta_df))).astype(str)),
        paragraph_id=np.asarray(meta_df.get('paragraph_id', pd.Series([''] * len(meta_df))).astype(str)),
        sentence_id=np.asarray(meta_df.get('sentence_id', pd.Series([''] * len(meta_df))).astype(str)),
        split=np.asarray(meta_df.get('split', pd.Series(['unknown'] * len(meta_df))).astype(str)),
    )

    stats = {
        'n_examples': int(len(meta_df)),
        'embedding_dim': int(embeddings.shape[1]) if len(embeddings) else int(model.output_dim),
        'task_type': args.task_type,
        'pooling': args.pooling,
    }
    if len(embeddings):
        norms = np.linalg.norm(embeddings, axis=1)
        stats.update({
            'embedding_norm_mean': float(norms.mean()),
            'embedding_norm_std': float(norms.std()),
        })
    pd.DataFrame([stats]).to_csv(Path(args.output_dir) / 'embedding_summary.csv', index=False)
    save_json(Path(args.output_dir) / 'summary.json', stats)
    if len(embeddings) >= 2:
        plot_pca(embeddings, meta_df, args.output_dir)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
