#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_embeddings(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    arrays = {k: data[k] for k in data.files}
    emb_key = None
    for key in ['embeddings', 'x', 'features']:
        if key in arrays:
            emb_key = key
            break
    if emb_key is None:
        raise ValueError(f'No embedding array found in {npz_path}. Keys: {list(arrays.keys())}')
    label_key = 'labels' if 'labels' in arrays else ('y' if 'y' in arrays else None)
    if label_key is None:
        raise ValueError('No labels found in embeddings NPZ.')
    split_key = 'splits' if 'splits' in arrays else ('split' if 'split' in arrays else None)
    ids_key = 'sample_ids' if 'sample_ids' in arrays else None
    return arrays[emb_key], arrays[label_key], arrays.get(split_key), arrays.get(ids_key)


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, num_classes))

    def forward(self, x):
        return self.net(x)


def main():
    p = argparse.ArgumentParser(description='Evaluate saved CET-MAE sentiment probe checkpoint on precomputed embeddings.')
    p.add_argument('--embeddings-npz', required=True)
    p.add_argument('--checkpoint', required=True, help='Saved probe checkpoint with model_state_dict.')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--split-name', default='test')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size', type=int, default=256)
    args = p.parse_args()

    ensure_dir(args.output_dir)
    x, y, splits, sample_ids = load_embeddings(args.embeddings_npz)
    if splits is not None:
        mask = np.asarray(splits).astype(str) == args.split_name
        x, y = x[mask], y[mask]
        if sample_ids is not None:
            sample_ids = sample_ids[mask]
    elif sample_ids is None:
        sample_ids = np.arange(len(y)).astype(str)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    input_dim = int(ckpt.get('input_dim', x.shape[1]))
    num_classes = int(ckpt.get('num_classes', len(np.unique(y))))
    dropout = float(ckpt.get('dropout', 0.0))

    model = LinearProbe(input_dim, num_classes, dropout=dropout)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()

    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    all_y, all_pred, all_prob = [], [], []
    with torch.no_grad():
        offset = 0
        for bx, by in dl:
            logits = model(bx.to(args.device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred = probs.argmax(axis=1)
            all_y.extend(by.numpy().tolist())
            all_pred.extend(pred.tolist())
            all_prob.extend(probs.tolist())
            offset += len(by)

    all_y = np.array(all_y)
    all_pred = np.array(all_pred)
    all_prob = np.array(all_prob)

    metrics = {
        'split': args.split_name,
        'accuracy': float(accuracy_score(all_y, all_pred)),
        'macro_f1': float(f1_score(all_y, all_pred, average='macro')),
        'weighted_f1': float(f1_score(all_y, all_pred, average='weighted')),
        'n_samples': int(len(all_y)),
    }
    pd.DataFrame([metrics]).to_csv(Path(args.output_dir) / f'{args.split_name}_metrics.csv', index=False)
    pd.DataFrame(confusion_matrix(all_y, all_pred)).to_csv(Path(args.output_dir) / f'{args.split_name}_confusion_matrix.csv', index=False)

    preds = pd.DataFrame({'sample_id': sample_ids if sample_ids is not None else np.arange(len(all_y)), 'y_true': all_y, 'y_pred': all_pred})
    for i in range(all_prob.shape[1]):
        preds[f'prob_{i}'] = all_prob[:, i]
    preds.to_csv(Path(args.output_dir) / f'{args.split_name}_predictions.csv', index=False)

    with open(Path(args.output_dir) / f'{args.split_name}_classification_report.json', 'w') as f:
        json.dump(classification_report(all_y, all_pred, output_dict=True), f, indent=2)


if __name__ == '__main__':
    main()
