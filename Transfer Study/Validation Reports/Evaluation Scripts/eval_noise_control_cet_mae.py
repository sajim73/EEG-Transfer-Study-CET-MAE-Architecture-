#!/usr/bin/env python3
import json
import math
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

try:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
except Exception:
    roc_curve = auc = precision_recall_curve = average_precision_score = None


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
    x = ' '.join(x.strip().lower().split())
    return x


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


def strip_module_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        out[k[7:]] = v if k.startswith('module.') else v
        if not k.startswith('module.'):
            out[k] = v
    return out


def load_checkpoint_state(path: str):
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict):
        for key in ['state_dict', 'model_state_dict', 'model', 'net']:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise ValueError('Checkpoint does not contain a state_dict-like object')
    return strip_module_prefix(ckpt)


def pick_first(d, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_eeg_and_mask(rec):
    eeg = pick_first(rec, ['input_embeddings', 'eeg', 'x', 'features', 'embeddings'])
    if eeg is None:
        raise KeyError('Missing EEG key. Expected one of input_embeddings/eeg/x/features/embeddings')
    eeg = to_numpy(eeg).astype(np.float32)
    if eeg.ndim == 1:
        eeg = eeg[None, :]

    mask = pick_first(rec, ['input_attn_mask', 'attention_mask', 'mask', 'attn_mask'])
    if mask is None:
        mask = np.ones((eeg.shape[0],), dtype=np.int64)
    else:
        mask = to_numpy(mask).astype(np.int64).reshape(-1)
    if mask.shape[0] != eeg.shape[0]:
        raise ValueError(f'Mask length {mask.shape[0]} != EEG length {eeg.shape[0]}')
    return eeg, mask


def build_reading_label_maps(labels_csv: str):
    df = read_table(labels_csv)
    df.columns = [str(c).strip() for c in df.columns]
    required = {'paragraph_id', 'sentence_id', 'sentence', 'control'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing columns in labels csv: {missing}')
    sent_map = {}
    pair_map = {}
    for _, row in df.iterrows():
        sent = normalize_text(row['sentence'])
        label = 1 if str(row['control']).strip().upper() == 'CONTROL' else 0
        sent_map[sent] = label
        pair_map[(str(row['paragraph_id']).strip(), str(row['sentence_id']).strip())] = label
    return sent_map, pair_map


def load_control_subset_pairs(path):
    if path is None:
        return None
    df = read_table(path)
    df.columns = [str(c).strip() for c in df.columns]
    pairs = set()
    for _, row in df.iterrows():
        p = str(row.get('paragraph_id', '')).strip()
        s = str(row.get('sentence_id', '')).strip()
        if p and s and p != 'nan' and s != 'nan':
            pairs.add((p, s))
    return pairs


def attach_reading_labels(records, sent_map, pair_map):
    examples = []
    dropped = 0
    for idx, rec in enumerate(records):
        try:
            eeg, mask = get_eeg_and_mask(rec)
        except Exception:
            dropped += 1
            continue
        sentence = pick_first(rec, ['sentence', 'text', 'target_text', 'raw_text'], '')
        paragraph_id = pick_first(rec, ['paragraph_id', 'paragraph', 'para_id'], None)
        sentence_id = pick_first(rec, ['sentence_id', 'sent_id'], None)

        label = None
        if paragraph_id is not None and sentence_id is not None:
            label = pair_map.get((str(paragraph_id).strip(), str(sentence_id).strip()))
        if label is None:
            label = sent_map.get(normalize_text(sentence))
        if label is None:
            dropped += 1
            continue

        examples.append({
            'sample_id': idx,
            'eeg': eeg,
            'mask': mask,
            'label': int(label),
            'sentence': sentence,
            'subject': pick_first(rec, ['subject', 'subject_id', 'subj'], 'unknown'),
            'paragraph_id': None if paragraph_id is None else str(paragraph_id).strip(),
            'sentence_id': None if sentence_id is None else str(sentence_id).strip(),
        })
    return examples, dropped


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
        if label is None:
            continue
        raw_labels.append(label)
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
    uniq = sorted(list(set(raw_labels)))
    label2id = {lab: i for i, lab in enumerate(uniq)}
    for ex in examples:
        ex['label'] = int(label2id[ex['raw_label']])
    return examples, label2id


def grouped_split(examples, seed=42, train_frac=0.7, val_frac=0.15):
    groups = defaultdict(list)
    has_subject = any(ex.get('subject', 'unknown') != 'unknown' for ex in examples)
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


class EEGSentenceDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            'eeg': torch.tensor(ex['eeg'], dtype=torch.float32),
            'mask': torch.tensor(ex['mask'], dtype=torch.long),
            'label': torch.tensor(ex['label'], dtype=torch.long),
            'subject': ex.get('subject', 'unknown'),
            'sentence': ex.get('sentence', ''),
            'paragraph_id': ex.get('paragraph_id'),
            'sentence_id': ex.get('sentence_id'),
            'sample_id': ex.get('sample_id'),
            'split': ex.get('split'),
            'condition': ex.get('condition', 'real'),
        }


def collate_fn(batch):
    max_len = max(item['eeg'].shape[0] for item in batch)
    feat_dim = batch[0]['eeg'].shape[1]
    eeg = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.long)
    meta = {k: [] for k in ['subject', 'sentence', 'paragraph_id', 'sentence_id', 'sample_id', 'split', 'condition']}
    for i, item in enumerate(batch):
        L = item['eeg'].shape[0]
        eeg[i, :L] = item['eeg']
        mask[i, :L] = item['mask']
        labels[i] = item['label']
        for k in meta:
            meta[k].append(item[k])
    return eeg, mask, labels, meta


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else math.nan


def macro_f1_score(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(f1s)) if f1s else math.nan


def balanced_accuracy(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rec = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        rec.append(0.0 if (tp + fn) == 0 else tp / (tp + fn))
    return float(np.mean(rec)) if rec else math.nan


def per_class_metrics(y_true, y_pred, num_classes, label_names=None):
    rows = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for c in range(num_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        support = int(np.sum(y_true == c))
        prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        rec = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        rows.append({
            'label_id': c,
            'label_name': label_names.get(c, str(c)) if isinstance(label_names, dict) else str(c),
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'support': support,
        })
    return pd.DataFrame(rows)


def inverse_label_map(label2id):
    return {int(v): str(k) for k, v in label2id.items()}


def save_split_manifest(examples_by_split, out_path):
    rows = []
    for split_name, examples in examples_by_split.items():
        for ex in examples:
            rows.append({
                'sample_id': ex.get('sample_id'),
                'split': split_name,
                'label': ex.get('label'),
                'subject': ex.get('subject'),
                'paragraph_id': ex.get('paragraph_id'),
                'sentence_id': ex.get('sentence_id'),
                'sentence': ex.get('sentence'),
                'condition': ex.get('condition', 'real'),
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def plot_history(history_df, output_dir):
    if history_df.empty:
        return
    for metric in ['loss', 'accuracy', 'macro_f1', 'balanced_accuracy']:
        train_col = f'train_{metric}'
        val_col = f'val_{metric}'
        if train_col in history_df.columns and val_col in history_df.columns:
            fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
            ax.plot(history_df['epoch'], history_df[train_col], marker='o', label='train')
            ax.plot(history_df['epoch'], history_df[val_col], marker='o', label='val')
            ax.set_xlabel('epoch')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} by epoch')
            ax.grid(alpha=0.2)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(Path(output_dir) / f'history_{metric}.png', bbox_inches='tight')
            plt.close(fig)


def plot_gradient_history(grad_df, output_dir):
    if grad_df.empty:
        return
    grad_df.to_csv(Path(output_dir) / 'gradient_history.csv', index=False)
    for key in ['group', 'param_name']:
        if key not in grad_df.columns:
            continue
        top = grad_df.groupby(key)['mean_grad_norm'].mean().sort_values(ascending=False).head(8).index.tolist()
        sub = grad_df[grad_df[key].isin(top)].copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
        for name, g in sub.groupby(key):
            ax.plot(g['epoch'], g['mean_grad_norm'], marker='o', linewidth=1.2, label=str(name))
        ax.set_xlabel('epoch')
        ax.set_ylabel('mean grad norm')
        ax.set_title(f'Gradient norms by {key}')
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f'gradient_history_by_{key}.png', bbox_inches='tight')
        plt.close(fig)


def plot_class_distribution(examples_by_split, output_dir, label_names=None):
    rows = []
    for split_name, examples in examples_by_split.items():
        for ex in examples:
            rows.append({'split': split_name, 'label': ex['label']})
    if not rows:
        return
    df = pd.DataFrame(rows)
    dist = df.groupby(['split', 'label']).size().reset_index(name='count')
    if label_names is not None:
        dist['label_name'] = dist['label'].map(lambda x: label_names.get(int(x), str(x)))
    else:
        dist['label_name'] = dist['label'].astype(str)
    dist.to_csv(Path(output_dir) / 'class_distribution.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    splits = list(pd.unique(dist['split']))
    labels = list(pd.unique(dist['label_name']))
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(splits))
    for i, split in enumerate(splits):
        sub = dist[dist['split'] == split]
        counts = []
        for lab in labels:
            tmp = sub[sub['label_name'] == lab]
            counts.append(int(tmp.iloc[0]['count']) if len(tmp) else 0)
        ax.bar(x + i * width, counts, width=width, label=split)
    ax.set_xticks(x + width * (len(splits) - 1) / 2)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('count')
    ax.set_title('Class distribution by split')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / 'class_distribution.png', bbox_inches='tight')
    plt.close(fig)


def save_predictions_and_embeddings(eval_out, split_dir, label_names=None):
    ensure_dir(split_dir)
    meta = eval_out['meta'].copy()
    meta['y_true'] = eval_out['y_true']
    meta['y_pred'] = eval_out['y_pred']
    if label_names is not None:
        meta['y_true_name'] = meta['y_true'].map(lambda x: label_names.get(int(x), str(x)))
        meta['y_pred_name'] = meta['y_pred'].map(lambda x: label_names.get(int(x), str(x)))
    probs = eval_out.get('probabilities')
    if probs is not None:
        for i in range(probs.shape[1]):
            meta[f'prob_{i}'] = probs[:, i]
    meta.to_csv(Path(split_dir) / 'predictions.csv', index=False)

    emb = eval_out.get('embeddings')
    if emb is not None:
        np.savez_compressed(
            Path(split_dir) / 'embeddings.npz',
            embeddings=emb.astype(np.float32),
            y_true=np.asarray(eval_out['y_true']),
            y_pred=np.asarray(eval_out['y_pred']),
            subject=np.asarray(meta.get('subject', pd.Series(['unknown'] * len(meta))).astype(str)),
            sentence=np.asarray(meta.get('sentence', pd.Series([''] * len(meta))).astype(str)),
            paragraph_id=np.asarray(meta.get('paragraph_id', pd.Series([''] * len(meta))).astype(str)),
            sentence_id=np.asarray(meta.get('sentence_id', pd.Series([''] * len(meta))).astype(str)),
            split=np.asarray(meta.get('split', pd.Series([split_dir.name] * len(meta))).astype(str)),
            condition=np.asarray(meta.get('condition', pd.Series(['real'] * len(meta))).astype(str)),
        )
        norms = np.linalg.norm(emb, axis=1)
        stats_rows = []
        for y in sorted(pd.unique(meta['y_true'])):
            idx = meta['y_true'] == y
            stats_rows.append({
                'label': int(y),
                'label_name': label_names.get(int(y), str(y)) if label_names else str(y),
                'n': int(idx.sum()),
                'mean_embedding_norm': float(norms[idx].mean()) if idx.sum() else np.nan,
                'std_embedding_norm': float(norms[idx].std()) if idx.sum() else np.nan,
            })
        pd.DataFrame(stats_rows).to_csv(Path(split_dir) / 'embedding_stats.csv', index=False)
        pca2 = PCA(n_components=2, random_state=42).fit_transform(emb)
        pca_df = meta.copy()
        pca_df['pca1'] = pca2[:, 0]
        pca_df['pca2'] = pca2[:, 1]
        pca_df.to_csv(Path(split_dir) / 'embedding_pca2.csv', index=False)
        for color_col in ['y_true', 'y_pred', 'subject']:
            if color_col in pca_df.columns and pca_df[color_col].notna().any():
                uniq = list(pd.unique(pca_df[color_col].astype(str)))
                if len(uniq) > 20:
                    vc = pca_df[color_col].astype(str).value_counts()
                    keep = set(vc.index[:19])
                    plot_vals = pca_df[color_col].astype(str).map(lambda z: z if z in keep else 'OTHER')
                else:
                    plot_vals = pca_df[color_col].astype(str)
                fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
                cmap = plt.cm.get_cmap('tab20', max(2, len(pd.unique(plot_vals))))
                for i, u in enumerate(pd.unique(plot_vals)):
                    idx = plot_vals == u
                    ax.scatter(pca_df.loc[idx, 'pca1'], pca_df.loc[idx, 'pca2'], s=14, alpha=0.78, label=u, color=cmap(i))
                ax.set_title(f'PCA embeddings colored by {color_col}')
                ax.set_xlabel('pca1')
                ax.set_ylabel('pca2')
                ax.grid(alpha=0.18)
                if len(pd.unique(plot_vals)) <= 20:
                    ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
                fig.tight_layout()
                fig.savefig(Path(split_dir) / f'embedding_pca2_{color_col}.png', bbox_inches='tight')
                plt.close(fig)


def plot_confusion_artifacts(y_true, y_pred, split_dir, label_names=None):
    labels = sorted(list(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist())))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    lab_names = [label_names.get(int(x), str(x)) if label_names else str(x) for x in labels]
    cm_df = pd.DataFrame(cm, index=[f'true_{x}' for x in lab_names], columns=[f'pred_{x}' for x in lab_names])
    cm_df.to_csv(Path(split_dir) / 'confusion_matrix.csv')
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=180)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(lab_names, rotation=20, ha='right')
    ax.set_yticklabels(lab_names)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title('Confusion matrix')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(Path(split_dir) / 'confusion_matrix.png', bbox_inches='tight')
    plt.close(fig)


def plot_prob_histograms(y_true, probabilities, split_dir, label_names=None):
    if probabilities is None or probabilities.shape[1] != 2:
        return
    scores = probabilities[:, 1]
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=180)
    for lab in sorted(pd.unique(pd.Series(y_true))):
        idx = np.asarray(y_true) == lab
        ax.hist(scores[idx], bins=20, alpha=0.55, label=label_names.get(int(lab), str(lab)) if label_names else str(lab))
    ax.set_xlabel('predicted probability of class 1')
    ax.set_ylabel('count')
    ax.set_title('Probability histogram by true class')
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(Path(split_dir) / 'probability_histogram.png', bbox_inches='tight')
    plt.close(fig)


def plot_roc_pr_artifacts(y_true, probabilities, split_dir):
    if probabilities is None or probabilities.shape[1] != 2 or roc_curve is None:
        return
    y_true = np.asarray(y_true)
    scores = probabilities[:, 1]
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(Path(split_dir) / 'roc_curve.csv', index=False)
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=180)
    ax.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC curve')
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(Path(split_dir) / 'roc_curve.png', bbox_inches='tight')
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    pd.DataFrame({'precision': precision, 'recall': recall}).to_csv(Path(split_dir) / 'pr_curve.csv', index=False)
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=180)
    ax.plot(recall, precision, label=f'AP={ap:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-recall curve')
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(Path(split_dir) / 'pr_curve.png', bbox_inches='tight')
    plt.close(fig)

    bins = np.linspace(0, 1, 11)
    bin_ids = np.digitize(scores, bins) - 1
    rows = []
    for b in range(10):
        idx = bin_ids == b
        if idx.sum() == 0:
            continue
        rows.append({
            'bin_left': float(bins[b]),
            'bin_right': float(bins[b + 1]),
            'mean_pred': float(scores[idx].mean()),
            'empirical_pos_rate': float(y_true[idx].mean()),
            'count': int(idx.sum()),
        })
    rel = pd.DataFrame(rows)
    rel.to_csv(Path(split_dir) / 'reliability_diagram.csv', index=False)
    if not rel.empty:
        fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=180)
        ax.plot(rel['mean_pred'], rel['empirical_pos_rate'], marker='o')
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Observed positive rate')
        ax.set_title('Reliability diagram')
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(Path(split_dir) / 'reliability_diagram.png', bbox_inches='tight')
        plt.close(fig)


def save_subject_metrics(meta_df, split_dir):
    if 'subject' not in meta_df.columns or meta_df['subject'].isna().all():
        return
    if 'y_true' not in meta_df.columns or 'y_pred' not in meta_df.columns:
        return
    rows = []
    for subject, g in meta_df.groupby('subject'):
        rows.append({
            'subject': subject,
            'n': int(len(g)),
            'accuracy': float((g['y_true'] == g['y_pred']).mean()),
        })
    subj = pd.DataFrame(rows).sort_values('accuracy', ascending=False)
    subj.to_csv(Path(split_dir) / 'subject_metrics.csv', index=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(subj)), 4.2), dpi=180)
    ax.bar(subj['subject'].astype(str), subj['accuracy'])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('accuracy')
    ax.set_title('Subject-wise accuracy')
    ax.grid(axis='y', alpha=0.2)
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
        tick.set_ha('right')
    fig.tight_layout()
    fig.savefig(Path(split_dir) / 'subject_accuracy.png', bbox_inches='tight')
    plt.close(fig)


def save_eval_artifacts(eval_out, split_dir, num_classes, label_names=None):
    ensure_dir(split_dir)
    metrics = {
        'loss': eval_out['loss'],
        'accuracy': eval_out['accuracy'],
        'macro_f1': eval_out['macro_f1'],
        'balanced_accuracy': eval_out['balanced_accuracy'],
        'n': int(len(eval_out['y_true'])),
    }
    save_json(Path(split_dir) / 'metrics.json', metrics)
    per_class = per_class_metrics(eval_out['y_true'], eval_out['y_pred'], num_classes, label_names=label_names)
    per_class.to_csv(Path(split_dir) / 'per_class_metrics.csv', index=False)
    save_predictions_and_embeddings(eval_out, Path(split_dir), label_names=label_names)
    plot_confusion_artifacts(eval_out['y_true'], eval_out['y_pred'], Path(split_dir), label_names=label_names)
    plot_prob_histograms(eval_out['y_true'], eval_out.get('probabilities'), Path(split_dir), label_names=label_names)
    plot_roc_pr_artifacts(eval_out['y_true'], eval_out.get('probabilities'), Path(split_dir))
    meta = eval_out['meta'].copy()
    meta['y_true'] = eval_out['y_true']
    meta['y_pred'] = eval_out['y_pred']
    save_subject_metrics(meta, Path(split_dir))


def run_epoch(model, loader, optimizer, device, num_classes, criterion=None, train=True, record_gradients=False, label_names=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.train() if train else model.eval()

    losses, ys, ps = [], [], []
    probs_all, embeds_all = [], []
    meta_batches = []
    grad_rows = []

    for eeg, mask, labels, meta in loader:
        eeg = eeg.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            logits, embeddings = model(eeg, mask, return_embeddings=True)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                if record_gradients:
                    group_stats = defaultdict(list)
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        norm = float(p.grad.detach().data.norm(2).item())
                        group = 'head' if 'head' in name else ('encoder' if ('backbone' in name or 'e_branch' in name or 'fc_eeg' in name) else 'other')
                        group_stats[group].append(norm)
                        grad_rows.append({'param_name': name, 'group': group, 'mean_grad_norm': norm})
                    optimizer.step()
                else:
                    optimizer.step()

        pred = torch.argmax(logits, dim=1)
        prob = torch.softmax(logits, dim=1)
        losses.append(float(loss.item()))
        ys.extend(labels.detach().cpu().numpy().tolist())
        ps.extend(pred.detach().cpu().numpy().tolist())
        probs_all.append(prob.detach().cpu().numpy())
        embeds_all.append(embeddings.detach().cpu().numpy())
        meta_batches.append(pd.DataFrame(meta))

    out = {
        'loss': float(np.mean(losses)) if losses else math.nan,
        'accuracy': accuracy_score(ys, ps),
        'macro_f1': macro_f1_score(ys, ps, num_classes),
        'balanced_accuracy': balanced_accuracy(ys, ps, num_classes),
        'y_true': np.asarray(ys),
        'y_pred': np.asarray(ps),
        'probabilities': np.concatenate(probs_all, axis=0) if probs_all else None,
        'embeddings': np.concatenate(embeds_all, axis=0) if embeds_all else None,
        'meta': pd.concat(meta_batches, ignore_index=True) if meta_batches else pd.DataFrame(),
    }
    grad_df = pd.DataFrame(grad_rows)
    if not grad_df.empty:
        grad_df = grad_df.groupby(['group', 'param_name'], as_index=False)['mean_grad_norm'].mean()
    return out, grad_df


def aggregate_epoch_gradients(epoch, grad_df):
    if grad_df.empty:
        return pd.DataFrame()
    out = grad_df.copy()
    out['epoch'] = epoch
    return out


def save_run_comparison(metrics_by_split, output_dir, title='Split metrics'):
    rows = []
    for split, m in metrics_by_split.items():
        row = {'split': split}
        row.update(m)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / 'split_metrics.csv', index=False)
    for metric in ['accuracy', 'macro_f1', 'balanced_accuracy']:
        if metric in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4.2), dpi=180)
            ax.bar(df['split'], df[metric])
            ax.set_ylim(0, 1.0)
            ax.set_ylabel(metric)
            ax.set_title(f'{title}: {metric}')
            ax.grid(axis='y', alpha=0.2)
            fig.tight_layout()
            fig.savefig(Path(output_dir) / f'split_{metric}.png', bbox_inches='tight')
            plt.close(fig)


def compute_train_stats(examples):
    vals = []
    for ex in examples:
        mask = np.asarray(ex['mask']).astype(bool)
        vals.append(np.asarray(ex['eeg'])[mask])
    vals = np.concatenate(vals, axis=0)
    mean = vals.mean(axis=0).astype(np.float32)
    std = vals.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def make_gaussian_noise_copy(examples, mean, std, seed=42, condition='gaussian'):
    rng = np.random.default_rng(seed)
    out = []
    for ex in examples:
        new_ex = dict(ex)
        new_ex['eeg'] = rng.normal(loc=mean, scale=std, size=np.asarray(ex['eeg']).shape).astype(np.float32)
        new_ex['condition'] = condition
        out.append(new_ex)
    return out


def make_permuted_copy(examples, seed=42, condition='permuted'):
    rng = np.random.default_rng(seed)
    eegs = [np.asarray(ex['eeg']).copy() for ex in examples]
    rng.shuffle(eegs)
    out = []
    for ex, eeg in zip(examples, eegs):
        new_ex = dict(ex)
        new_ex['eeg'] = eeg.astype(np.float32)
        new_ex['condition'] = condition
        out.append(new_ex)
    return out


class FrozenCETMAEEncoder(nn.Module):
    def __init__(self, checkpoint_path: str, device: torch.device, pretrain_path: str = './models/huggingface/bart-large'):
        super().__init__()
        from model_mae_bart import CETMAE_project_late_bart
        self.backbone = CETMAE_project_late_bart(pretrain_path=pretrain_path, device=device)
        state = load_checkpoint_state(checkpoint_path)
        self.backbone.load_state_dict(state, strict=False)
        self.output_dim = int(self.backbone.fc_eeg.out_features)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def forward(self, eeg: torch.Tensor, mask: torch.Tensor):
        if eeg.size(-1) == int(self.backbone.fc_eeg.in_features):
            x = self.backbone.pos_embed_e(eeg)
            x = self.backbone.e_branch(x, src_key_padding_mask=(mask == 0))
            x = self.backbone.act(self.backbone.fc_eeg(x))
        elif eeg.size(-1) == self.output_dim:
            x = eeg
        else:
            raise ValueError(f'Unexpected EEG feature dim {eeg.size(-1)}')
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        return pooled


class LinearProbe(nn.Module):
    def __init__(self, encoder: FrozenCETMAEEncoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.output_dim, num_classes)

    def forward(self, eeg, mask, return_embeddings=False):
        with torch.no_grad():
            z = self.encoder(eeg, mask)
        logits = self.head(z)
        return (logits, z) if return_embeddings else logits


class CETMAEFinetuner(nn.Module):
    def __init__(self, checkpoint_path: str, num_classes: int, device: torch.device, pretrain_path: str = './models/huggingface/bart-large', dropout: float = 0.1):
        super().__init__()
        from model_mae_bart import CETMAE_project_late_bart
        self.backbone = CETMAE_project_late_bart(pretrain_path=pretrain_path, device=device)
        state = load_checkpoint_state(checkpoint_path)
        self.backbone.load_state_dict(state, strict=False)
        self.output_dim = int(self.backbone.fc_eeg.out_features)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.output_dim, num_classes)

    def encode(self, eeg, mask):
        if eeg.size(-1) == int(self.backbone.fc_eeg.in_features):
            x = self.backbone.pos_embed_e(eeg)
            x = self.backbone.e_branch(x, src_key_padding_mask=(mask == 0))
            x = self.backbone.act(self.backbone.fc_eeg(x))
        elif eeg.size(-1) == self.output_dim:
            x = eeg
        else:
            raise ValueError(f'Unexpected EEG feature dim {eeg.size(-1)}')
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, eeg, mask, return_embeddings=False):
        z = self.encode(eeg, mask)
        logits = self.head(self.dropout(z))
        return (logits, z) if return_embeddings else logits


import argparse


def train_condition(condition_name, split_examples, args, device, encoder, output_dir):
    ensure_dir(output_dir)
    train_ex = split_examples['train']
    val_ex = split_examples['val']
    test_ex = split_examples['test']

    plot_class_distribution(split_examples, output_dir, label_names={0: 'noncontrol', 1: 'control'})
    save_split_manifest(split_examples, Path(output_dir) / 'split_manifest.csv')

    train_loader = torch.utils.data.DataLoader(EEGSentenceDataset(train_ex), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(EEGSentenceDataset(val_ex), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(EEGSentenceDataset(test_ex), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = LinearProbe(encoder, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_rows = []
    grad_rows = []
    best_state = None
    best_val = -1.0
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        train_out, train_grad = run_epoch(model, train_loader, optimizer, device, 2, train=True, record_gradients=True)
        val_out, _ = run_epoch(model, val_loader, optimizer, device, 2, train=False)
        history_rows.append({
            'epoch': epoch,
            'train_loss': train_out['loss'],
            'train_accuracy': train_out['accuracy'],
            'train_macro_f1': train_out['macro_f1'],
            'train_balanced_accuracy': train_out['balanced_accuracy'],
            'val_loss': val_out['loss'],
            'val_accuracy': val_out['accuracy'],
            'val_macro_f1': val_out['macro_f1'],
            'val_balanced_accuracy': val_out['balanced_accuracy'],
        })
        if not train_grad.empty:
            grad_rows.append(aggregate_epoch_gradients(epoch, train_grad))
        if val_out['macro_f1'] > best_val:
            best_val = val_out['macro_f1']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, Path(output_dir) / f'best_{condition_name}.pt')
            patience_left = args.patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            break

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(Path(output_dir) / 'history.csv', index=False)
    plot_history(history_df, output_dir)
    if grad_rows:
        plot_gradient_history(pd.concat(grad_rows, ignore_index=True), output_dir)

    model.load_state_dict(best_state)
    metrics = {}
    for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        out, _ = run_epoch(model, loader, optimizer=None, device=device, num_classes=2, train=False)
        save_eval_artifacts(out, Path(output_dir) / split_name, 2, label_names={0: 'noncontrol', 1: 'control'})
        metrics[split_name] = {'loss': out['loss'], 'accuracy': out['accuracy'], 'macro_f1': out['macro_f1'], 'balanced_accuracy': out['balanced_accuracy']}
    save_json(Path(output_dir) / 'metrics.json', metrics)
    return metrics['test']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--labels-csv', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--pretrain-path', default='./models/huggingface/bart-large')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-frac', type=float, default=0.7)
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--patience', type=int, default=8)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    save_json(Path(args.output_dir) / 'run_config.json', vars(args))
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sent_map, pair_map = build_reading_label_maps(args.labels_csv)
    raw_records = load_records(args.data)
    examples, dropped = attach_reading_labels(raw_records, sent_map, pair_map)
    if len(examples) < 10:
        raise ValueError(f'Too few matched examples: {len(examples)}')
    train_ex, val_ex, test_ex = grouped_split(examples, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac)
    for ex in train_ex + val_ex + test_ex:
        ex['condition'] = 'real'

    mean, std = compute_train_stats(train_ex)
    condition_splits = {
        'real': {'train': train_ex, 'val': val_ex, 'test': test_ex},
        'gaussian': {
            'train': make_gaussian_noise_copy(train_ex, mean, std, seed=args.seed + 11, condition='gaussian'),
            'val': make_gaussian_noise_copy(val_ex, mean, std, seed=args.seed + 12, condition='gaussian'),
            'test': make_gaussian_noise_copy(test_ex, mean, std, seed=args.seed + 13, condition='gaussian'),
        },
        'permuted': {
            'train': make_permuted_copy(train_ex, seed=args.seed + 21, condition='permuted'),
            'val': make_permuted_copy(val_ex, seed=args.seed + 22, condition='permuted'),
            'test': make_permuted_copy(test_ex, seed=args.seed + 23, condition='permuted'),
        },
    }

    encoder = FrozenCETMAEEncoder(args.checkpoint, device=device, pretrain_path=args.pretrain_path).to(device)
    results = {}
    rows = []
    for name, splits in condition_splits.items():
        cond_dir = Path(args.output_dir) / name
        test_metrics = train_condition(name, splits, args, device, encoder, cond_dir)
        results[name] = test_metrics
        row = {'condition': name}
        row.update(test_metrics)
        rows.append(row)

    comp = pd.DataFrame(rows)
    comp.to_csv(Path(args.output_dir) / 'noise_control_results.csv', index=False)
    for metric in ['accuracy', 'macro_f1', 'balanced_accuracy']:
        fig, ax = plt.subplots(figsize=(6, 4.2), dpi=180)
        ax.bar(comp['condition'], comp[metric])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(metric)
        ax.set_title(f'Noise control comparison: {metric}')
        ax.grid(axis='y', alpha=0.2)
        fig.tight_layout()
        fig.savefig(Path(args.output_dir) / f'noise_control_{metric}.png', bbox_inches='tight')
        plt.close(fig)

    summary = {
        'n_total': len(examples),
        'n_dropped_unmatched': dropped,
        'n_train': len(train_ex),
        'n_val': len(val_ex),
        'n_test': len(test_ex),
        'results': results,
        'delta_real_minus_gaussian_macro_f1': results['real']['macro_f1'] - results['gaussian']['macro_f1'],
        'delta_real_minus_permuted_macro_f1': results['real']['macro_f1'] - results['permuted']['macro_f1'],
        'delta_real_minus_gaussian_bal_acc': results['real']['balanced_accuracy'] - results['gaussian']['balanced_accuracy'],
        'delta_real_minus_permuted_bal_acc': results['real']['balanced_accuracy'] - results['permuted']['balanced_accuracy'],
    }
    save_json(Path(args.output_dir) / 'noise_control_results.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
