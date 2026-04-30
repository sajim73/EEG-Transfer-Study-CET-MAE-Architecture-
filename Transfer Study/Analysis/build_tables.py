#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_table(path: str) -> pd.DataFrame:
    for sep in [',', ';', '\t']:
        try:
            df = pd.read_csv(path, sep=sep, engine='python')
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    raise ValueError(f'Could not read table: {path}')


def safe_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        kk = f'{prefix}.{k}' if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, kk))
        else:
            out[kk] = v
    return out


def find_run_dirs(base_dir):
    base = Path(base_dir)
    runs = []
    for p in [base] + [x for x in base.rglob('*') if x.is_dir()]:
        if any((p / fname).exists() for fname in ['metrics.json', 'noise_control_results.json', 'history.csv', 'split_manifest.csv', 'run_config.json']):
            runs.append(p)
    dedup = []
    seen = set()
    for r in runs:
        rp = r.resolve()
        if rp not in seen:
            dedup.append(r)
            seen.add(rp)
    return dedup


def infer_run_type(run_dir: Path):
    if (run_dir / 'noise_control_results.json').exists() or (run_dir / 'noise_control_results.csv').exists():
        return 'noise_control'
    if (run_dir / 'best_probe.pt').exists():
        return 'probe'
    if (run_dir / 'best_finetune_sentiment.pt').exists():
        return 'finetune_sentiment'
    if run_dir.name in ['real', 'gaussian', 'permuted']:
        return 'noise_condition'
    return 'generic'


def list_split_dirs(run_dir: Path):
    return [p for p in [run_dir / 'train', run_dir / 'val', run_dir / 'test'] if p.exists() and p.is_dir()]


def load_run_row(run_dir: Path):
    row = {'run_dir': str(run_dir), 'run_name': run_dir.name, 'run_type': infer_run_type(run_dir)}
    for fname, prefix in [('metrics.json', 'metrics'), ('noise_control_results.json', 'noise')]:
        obj = safe_json(run_dir / fname)
        if obj:
            row.update(flatten(obj, prefix))
    hist = run_dir / 'history.csv'
    if hist.exists():
        h = read_table(hist)
        row['epochs_recorded'] = int(len(h))
        for c in ['val_macro_f1', 'val_accuracy', 'val_loss', 'train_macro_f1', 'train_accuracy']:
            if c in h.columns:
                idx = h[c].idxmin() if 'loss' in c else h[c].idxmax()
                row[f'best_{c}'] = float(h.loc[idx, c])
                if 'epoch' in h.columns:
                    row[f'best_{c}_epoch'] = int(h.loc[idx, 'epoch'])
    return row


def load_split_rows(run_dir: Path):
    rows = []
    for split_dir in list_split_dirs(run_dir):
        row = {'parent_run_dir': str(run_dir), 'run_name': run_dir.name, 'split': split_dir.name, 'split_dir': str(split_dir), 'run_type': infer_run_type(run_dir)}
        m = safe_json(split_dir / 'metrics.json')
        if m:
            row.update(flatten(m, 'metrics'))
        pred_path = split_dir / 'predictions.csv'
        if pred_path.exists():
            pred = read_table(pred_path)
            row['n_predictions'] = int(len(pred))
            if {'y_true', 'y_pred'}.issubset(pred.columns):
                row['accuracy_from_predictions'] = float((pred['y_true'] == pred['y_pred']).mean())
            if 'subject' in pred.columns:
                row['n_subjects'] = int(pred['subject'].nunique(dropna=True))
        pcm = split_dir / 'per_class_metrics.csv'
        if pcm.exists():
            row['has_per_class_metrics'] = True
        rows.append(row)
    return rows


def plot_bar(df, xcol, ycol, out_path, title, hue=None):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    if hue is None or hue not in df.columns:
        ax.bar(df[xcol].astype(str), df[ycol].astype(float))
    else:
        groups = list(pd.unique(df[hue].astype(str)))
        xvals = list(df[xcol].astype(str))
        xpos = np.arange(len(xvals))
        width = 0.8 / max(1, len(groups))
        for i, g in enumerate(groups):
            sub = df[df[hue].astype(str) == g]
            ys = []
            for xv in xvals:
                m = sub[sub[xcol].astype(str) == xv]
                ys.append(float(m.iloc[0][ycol]) if len(m) else np.nan)
            ax.bar(xpos + i * width, ys, width=width, label=g)
        ax.set_xticks(xpos + width * (len(groups) - 1) / 2)
        ax.set_xticklabels(xvals)
        ax.legend(frameon=False, fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ycol)
    ax.grid(axis='y', alpha=0.2)
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha('right')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def merge_prediction_tables(run_dir: Path):
    rows = []
    for split_dir in list_split_dirs(run_dir):
        pred_path = split_dir / 'predictions.csv'
        if pred_path.exists():
            pred = read_table(pred_path)
            pred['run_name'] = run_dir.name
            pred['split'] = split_dir.name
            rows.append(pred)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def merge_subject_tables(run_dir: Path):
    rows = []
    for split_dir in list_split_dirs(run_dir):
        subj = split_dir / 'subject_metrics.csv'
        if subj.exists():
            df = read_table(subj)
            df['run_name'] = run_dir.name
            df['split'] = split_dir.name
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def merge_class_tables(run_dir: Path):
    rows = []
    for split_dir in list_split_dirs(run_dir):
        pcm = split_dir / 'per_class_metrics.csv'
        if pcm.exists():
            df = read_table(pcm)
            df['run_name'] = run_dir.name
            df['split'] = split_dir.name
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def find_best_metric_column(run_df, split_df):
    for c in ['metrics.split_metrics.test.macro_f1', 'metrics.test_metrics.macro_f1', 'best_val_macro_f1', 'noise.results.real.macro_f1']:
        if c in run_df.columns:
            return ('run', c)
    for c in ['metrics.macro_f1', 'metrics.accuracy', 'accuracy_from_predictions']:
        if c in split_df.columns:
            return ('split', c)
    return (None, None)


def render_table_md(df, max_rows=20):
    if df is None or df.empty:
        return 'No data available.\n'
    return df.head(max_rows).to_markdown(index=False) + '\n'


def write_report(out_dir: Path, run_df: pd.DataFrame, split_df: pd.DataFrame, noise_df: pd.DataFrame, subject_df: pd.DataFrame, class_df: pd.DataFrame):
    lines = ['# Results Report', '']
    lines += ['## Overview', '']
    lines += [f'- Runs found: {len(run_df)}', f'- Split-level evaluations found: {len(split_df)}', '']

    lines += ['## Run overview', '']
    cols = [c for c in ['run_name', 'run_type', 'best_val_macro_f1', 'metrics.split_metrics.test.macro_f1', 'metrics.split_metrics.test.accuracy'] if c in run_df.columns]
    lines += [render_table_md(run_df[cols] if cols else run_df)]

    lines += ['## Split overview', '']
    cols = [c for c in ['run_name', 'run_type', 'split', 'metrics.accuracy', 'metrics.macro_f1', 'metrics.balanced_accuracy', 'n_predictions'] if c in split_df.columns]
    lines += [render_table_md(split_df[cols] if cols else split_df)]

    if not noise_df.empty:
        lines += ['## Noise control', '']
        lines += [render_table_md(noise_df)]

    if not class_df.empty:
        lines += ['## Per-class metrics', '']
        show = class_df[['run_name', 'split', 'label_name', 'precision', 'recall', 'f1', 'support']] if {'run_name', 'split', 'label_name', 'precision', 'recall', 'f1', 'support'}.issubset(class_df.columns) else class_df
        lines += [render_table_md(show)]

    if not subject_df.empty:
        lines += ['## Subject metrics', '']
        show = subject_df[['run_name', 'split', 'subject', 'n', 'accuracy']] if {'run_name', 'split', 'subject', 'n', 'accuracy'}.issubset(subject_df.columns) else subject_df
        lines += [render_table_md(show)]

    lines += ['## Output files', '']
    files = sorted([p.name for p in out_dir.iterdir() if p.is_file()])
    for f in files:
        lines.append(f'- {f}')
    lines.append('')

    (out_dir / 'results_report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    run_dirs = find_run_dirs(args.runs_dir)

    run_rows = [load_run_row(r) for r in run_dirs]
    run_df = pd.DataFrame(run_rows).sort_values(['run_type', 'run_name']) if run_rows else pd.DataFrame()
    run_df.to_csv(Path(args.output_dir) / 'runs_overview.csv', index=False)

    split_rows = []
    all_pred, all_subj, all_class = [], [], []
    noise_rows = []
    for r in run_dirs:
        split_rows.extend(load_split_rows(r))
        pred = merge_prediction_tables(r)
        if not pred.empty:
            pred.to_csv(Path(args.output_dir) / f'predictions_{r.name}.csv', index=False)
            all_pred.append(pred)
        subj = merge_subject_tables(r)
        if not subj.empty:
            all_subj.append(subj)
        cls = merge_class_tables(r)
        if not cls.empty:
            all_class.append(cls)
        noise_json = safe_json(r / 'noise_control_results.json')
        if noise_json and 'results' in noise_json:
            for cond, vals in noise_json['results'].items():
                row = {'run_name': r.name, 'condition': cond}
                row.update(vals)
                noise_rows.append(row)

    split_df = pd.DataFrame(split_rows).sort_values(['run_name', 'split']) if split_rows else pd.DataFrame()
    split_df.to_csv(Path(args.output_dir) / 'split_overview.csv', index=False)

    all_pred_df = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    if not all_pred_df.empty:
        all_pred_df.to_csv(Path(args.output_dir) / 'all_predictions.csv', index=False)

    subject_df = pd.concat(all_subj, ignore_index=True) if all_subj else pd.DataFrame()
    if not subject_df.empty:
        subject_df.to_csv(Path(args.output_dir) / 'all_subject_metrics.csv', index=False)
        plot_bar(subject_df.sort_values('accuracy', ascending=False).head(30), 'subject', 'accuracy', Path(args.output_dir) / 'top_subject_accuracy.png', 'Top subject accuracies', hue='run_name' if 'run_name' in subject_df.columns else None)

    class_df = pd.concat(all_class, ignore_index=True) if all_class else pd.DataFrame()
    if not class_df.empty:
        class_df.to_csv(Path(args.output_dir) / 'all_class_metrics.csv', index=False)
        if {'run_name', 'f1', 'label_name'}.issubset(class_df.columns):
            plot_bar(class_df[['run_name', 'f1', 'label_name']].rename(columns={'label_name': 'hue_lab'}), 'run_name', 'f1', Path(args.output_dir) / 'class_f1_by_run.png', 'Per-class F1 by run', hue='hue_lab')

    noise_df = pd.DataFrame(noise_rows)
    if not noise_df.empty:
        noise_df.to_csv(Path(args.output_dir) / 'noise_control_table.csv', index=False)
        for metric in ['accuracy', 'macro_f1', 'balanced_accuracy']:
            if metric in noise_df.columns:
                plot_bar(noise_df, 'run_name', metric, Path(args.output_dir) / f'noise_{metric}.png', f'Noise control {metric}', hue='condition')

    if not split_df.empty:
        for metric in ['metrics.accuracy', 'metrics.macro_f1', 'metrics.balanced_accuracy']:
            if metric in split_df.columns:
                plot_bar(split_df[split_df['split'] == 'test'].sort_values(metric, ascending=False), 'run_name', metric, Path(args.output_dir) / f'test_{metric.replace(".", "_")}.png', f'Test {metric} by run', hue='run_type')

    scope, best_col = find_best_metric_column(run_df, split_df)
    if scope == 'run' and best_col in run_df.columns:
        best_tbl = run_df[['run_name', 'run_type', best_col]].sort_values(best_col, ascending=False).rename(columns={best_col: 'score'})
        best_tbl.to_csv(Path(args.output_dir) / 'best_runs_table.csv', index=False)
    elif scope == 'split' and best_col in split_df.columns:
        best_tbl = split_df[['run_name', 'run_type', 'split', best_col]].sort_values(best_col, ascending=False).rename(columns={best_col: 'score'})
        best_tbl.to_csv(Path(args.output_dir) / 'best_runs_table.csv', index=False)

    write_report(Path(args.output_dir), run_df, split_df, noise_df, subject_df, class_df)

    summary = {
        'n_runs_found': int(len(run_dirs)),
        'n_split_rows': int(len(split_df)),
        'outputs': sorted([p.name for p in Path(args.output_dir).glob('*') if p.is_file()]),
    }
    with open(Path(args.output_dir) / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
