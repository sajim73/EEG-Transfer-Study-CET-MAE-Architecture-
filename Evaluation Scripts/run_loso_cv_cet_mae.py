#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def discover_subjects(npz_path: str, subject_key: str = 'subjects'):
    data = np.load(npz_path, allow_pickle=True)
    if subject_key not in data.files:
        alt = {'subjects': ['subject', 'subject_id']}.get(subject_key, [])
        for k in alt:
            if k in data.files:
                subject_key = k
                break
    if subject_key not in data.files:
        raise ValueError(f'No subject key found in {npz_path}. Keys: {list(data.files)}')
    subjects = np.asarray(data[subject_key]).astype(str)
    return sorted(pd.unique(subjects).tolist())


def build_command(args, heldout_subject: str, fold_dir: Path):
    cmd = [
        sys.executable,
        args.script,
        '--npz', args.npz,
        '--output-dir', str(fold_dir),
        '--holdout-subject', str(heldout_subject),
    ]
    if args.checkpoint:
        cmd.extend(['--checkpoint', args.checkpoint])
    if args.random_init:
        cmd.append('--random-init')
    if args.device:
        cmd.extend(['--device', args.device])
    if args.batch_size is not None:
        cmd.extend(['--batch-size', str(args.batch_size)])
    if args.epochs is not None:
        cmd.extend(['--epochs', str(args.epochs)])
    if args.encoder_lr is not None:
        cmd.extend(['--encoder-lr', str(args.encoder_lr)])
    if args.head_lr is not None:
        cmd.extend(['--head-lr', str(args.head_lr)])
    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))
    return cmd


def main():
    p = argparse.ArgumentParser(description='Run leave-one-subject-out CV for CET-MAE transfer scripts.')
    p.add_argument('--script', required=True, help='Training script to run for each held-out subject.')
    p.add_argument('--npz', required=True, help='Dataset NPZ with subjects key.')
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--random-init', action='store_true')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--subject-key', default='subjects')
    p.add_argument('--device', default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--encoder-lr', type=float, default=None)
    p.add_argument('--head-lr', type=float, default=None)
    p.add_argument('--extra-args', default='')
    args = p.parse_args()

    ensure_dir(args.output_dir)
    subjects = discover_subjects(args.npz, args.subject_key)
    pd.DataFrame({'heldout_subject': subjects}).to_csv(Path(args.output_dir) / 'subjects.csv', index=False)

    fold_rows = []
    for subj in subjects:
        fold_dir = Path(args.output_dir) / f'fold_{subj}'
        ensure_dir(fold_dir)
        cmd = build_command(args, subj, fold_dir)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        with open(fold_dir / 'stdout.log', 'w') as f:
            f.write(proc.stdout)
        with open(fold_dir / 'stderr.log', 'w') as f:
            f.write(proc.stderr)
        status = 'ok' if proc.returncode == 0 else 'failed'
        row = {'heldout_subject': subj, 'status': status, 'returncode': proc.returncode}
        metrics_path = fold_dir / 'test_metrics.csv'
        if metrics_path.exists():
            try:
                m = pd.read_csv(metrics_path).iloc[0].to_dict()
                row.update({f'test_{k}' if k not in ['split'] else k: v for k, v in m.items()})
            except Exception:
                pass
        fold_rows.append(row)

    summary = pd.DataFrame(fold_rows)
    summary.to_csv(Path(args.output_dir) / 'loso_summary.csv', index=False)

    ok = summary[summary['status'] == 'ok'].copy()
    agg = {}
    for col in ['test_accuracy', 'test_macro_f1', 'test_weighted_f1', 'test_loss']:
        if col in ok.columns:
            agg[f'{col}_mean'] = float(pd.to_numeric(ok[col], errors='coerce').mean())
            agg[f'{col}_std'] = float(pd.to_numeric(ok[col], errors='coerce').std())
    agg['n_folds'] = int(len(subjects))
    agg['n_successful_folds'] = int((summary['status'] == 'ok').sum())
    with open(Path(args.output_dir) / 'loso_aggregate.json', 'w') as f:
        json.dump(agg, f, indent=2)


if __name__ == '__main__':
    main()
