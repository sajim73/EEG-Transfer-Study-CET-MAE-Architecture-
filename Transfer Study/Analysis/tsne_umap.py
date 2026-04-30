#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
except Exception:
    umap = None

try:
    from sklearn.metrics import silhouette_score
except Exception:
    silhouette_score = None


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
    raise ValueError(f'Could not read tabular file: {path}')


def sanitize(x):
    return ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in str(x))


def load_npz(path):
    obj = np.load(path, allow_pickle=True)
    keys = set(obj.files)
    emb_key = None
    for cand in ['embeddings', 'X', 'features', 'z', 'pooled_embeddings']:
        if cand in keys:
            emb_key = cand
            break
    if emb_key is None:
        raise ValueError(f'No embedding key found in {path}; keys={sorted(keys)}')
    X = np.asarray(obj[emb_key], dtype=np.float32)
    meta = {}
    for k in ['y_true', 'y_pred', 'subject', 'sentence', 'paragraph_id', 'sentence_id', 'split', 'condition']:
        if k in keys:
            meta[k] = list(obj[k])
    meta_df = pd.DataFrame(meta) if meta else pd.DataFrame(index=np.arange(len(X)))
    return X, meta_df


def load_csv(path):
    df = read_table(path)
    emb_cols = [c for c in df.columns if str(c).startswith('emb_')]
    if not emb_cols:
        raise ValueError(f'CSV {path} has no emb_* columns')
    X = df[emb_cols].to_numpy(dtype=np.float32)
    meta = df.drop(columns=emb_cols)
    return X, meta


def load_embeddings(path):
    path = Path(path)
    if path.suffix.lower() == '.npz':
        return load_npz(path)
    if path.suffix.lower() in ['.csv', '.tsv']:
        return load_csv(path)
    raise ValueError(f'Unsupported embedding file type: {path}')


def pick_color_columns(df):
    cols = []
    for c in ['y_true', 'y_pred', 'subject', 'condition', 'split']:
        if c in df.columns and df[c].notna().any():
            cols.append(c)
    return cols or [None]


def plot_scatter(df, xcol, ycol, color_col, out_png, title):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    if color_col is None:
        ax.scatter(df[xcol], df[ycol], s=14, alpha=0.8)
    else:
        vals = df[color_col].fillna('NA').astype(str)
        uniq = list(pd.unique(vals))
        if len(uniq) > 20:
            vc = vals.value_counts()
            keep = set(vc.index[:19])
            vals = vals.map(lambda z: z if z in keep else 'OTHER')
            uniq = list(pd.unique(vals))
        cmap = plt.cm.get_cmap('tab20', max(2, len(uniq)))
        for i, u in enumerate(uniq):
            idx = vals == u
            ax.scatter(df.loc[idx, xcol], df.loc[idx, ycol], s=16, alpha=0.8, label=u, color=cmap(i))
        if len(uniq) <= 20:
            ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)


def label_stats(df, label_col, xcol, ycol):
    rows = []
    centroids = {}
    if label_col is None or label_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    for lab, g in df.groupby(label_col, dropna=False):
        rows.append({
            'label': lab,
            'n': int(len(g)),
            'centroid_x': float(g[xcol].mean()),
            'centroid_y': float(g[ycol].mean()),
            'std_x': float(g[xcol].std(ddof=0)),
            'std_y': float(g[ycol].std(ddof=0)),
        })
        centroids[lab] = np.array([g[xcol].mean(), g[ycol].mean()], dtype=np.float32)
    drows = []
    labs = list(centroids.keys())
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            drows.append({
                'label_a': labs[i],
                'label_b': labs[j],
                'centroid_distance': float(np.linalg.norm(centroids[labs[i]] - centroids[labs[j]]))
            })
    return pd.DataFrame(rows), pd.DataFrame(drows)


def reduce_embeddings(X, seed=42, pca_dim=50, perplexity=30, n_neighbors=15, min_dist=0.1):
    n_comp = min(pca_dim, X.shape[1], max(2, X.shape[0] - 1))
    pca = PCA(n_components=n_comp, random_state=seed)
    Xp = pca.fit_transform(X)
    perpl = min(perplexity, max(2, (X.shape[0] - 1) // 3))
    Xt = TSNE(n_components=2, random_state=seed, init='pca', learning_rate='auto', perplexity=perpl).fit_transform(Xp)
    Xu = None
    if umap is not None:
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=min(n_neighbors, max(2, X.shape[0] - 1)), min_dist=min_dist)
        Xu = reducer.fit_transform(Xp)
    return Xp, Xt, Xu, pca


def process_embedding_file(emb_path, out_dir, seed=42, pca_dim=50, perplexity=30, n_neighbors=15, min_dist=0.1):
    ensure_dir(out_dir)
    X, meta = load_embeddings(emb_path)
    if len(meta) != len(X):
        meta = meta.reset_index(drop=True)
        if len(meta) != len(X):
            raise ValueError(f'Metadata length mismatch for {emb_path}')
    Xp, Xt, Xu, pca = reduce_embeddings(X, seed=seed, pca_dim=pca_dim, perplexity=perplexity, n_neighbors=n_neighbors, min_dist=min_dist)

    pd.DataFrame({'component': np.arange(1, len(pca.explained_variance_ratio_) + 1), 'explained_variance_ratio': pca.explained_variance_ratio_, 'cumulative_explained_variance': np.cumsum(pca.explained_variance_ratio_)}).to_csv(Path(out_dir) / 'pca_explained_variance.csv', index=False)
    pca_df = pd.concat([meta.copy(), pd.DataFrame(Xp[:, :min(10, Xp.shape[1])], columns=[f'pca_{i+1}' for i in range(min(10, Xp.shape[1]))])], axis=1)
    pca_df.to_csv(Path(out_dir) / 'pca_embeddings.csv', index=False)

    tsne_df = meta.copy()
    tsne_df['tsne_1'] = Xt[:, 0]
    tsne_df['tsne_2'] = Xt[:, 1]
    tsne_df.to_csv(Path(out_dir) / 'tsne_coords.csv', index=False)

    if Xu is not None:
        umap_df = meta.copy()
        umap_df['umap_1'] = Xu[:, 0]
        umap_df['umap_2'] = Xu[:, 1]
        umap_df.to_csv(Path(out_dir) / 'umap_coords.csv', index=False)
    else:
        umap_df = None

    for c in pick_color_columns(meta):
        suffix = 'plain' if c is None else sanitize(c)
        plot_scatter(tsne_df, 'tsne_1', 'tsne_2', c, Path(out_dir) / f'tsne_{suffix}.png', f't-SNE colored by {c or "none"}')
        if umap_df is not None:
            plot_scatter(umap_df, 'umap_1', 'umap_2', c, Path(out_dir) / f'umap_{suffix}.png', f'UMAP colored by {c or "none"}')

    primary = 'y_true' if 'y_true' in meta.columns else None
    if primary is not None:
        s, d = label_stats(tsne_df, primary, 'tsne_1', 'tsne_2')
        if not s.empty:
            s.to_csv(Path(out_dir) / 'tsne_label_stats.csv', index=False)
        if not d.empty:
            d.to_csv(Path(out_dir) / 'tsne_centroid_distances.csv', index=False)
        if umap_df is not None:
            s2, d2 = label_stats(umap_df, primary, 'umap_1', 'umap_2')
            if not s2.empty:
                s2.to_csv(Path(out_dir) / 'umap_label_stats.csv', index=False)
            if not d2.empty:
                d2.to_csv(Path(out_dir) / 'umap_centroid_distances.csv', index=False)

    summary = {
        'source_embedding_file': str(emb_path),
        'n_samples': int(X.shape[0]),
        'embedding_dim': int(X.shape[1]),
        'pca_dim_used': int(Xp.shape[1]),
        'explained_variance_top10_sum': float(pca.explained_variance_ratio_[:10].sum()),
        'umap_available': bool(Xu is not None),
    }
    if primary is not None and silhouette_score is not None and len(pd.unique(meta[primary])) >= 2:
        try:
            summary['silhouette_tsne'] = float(silhouette_score(Xt, meta[primary].astype(str)))
        except Exception:
            pass
        if Xu is not None:
            try:
                summary['silhouette_umap'] = float(silhouette_score(Xu, meta[primary].astype(str)))
            except Exception:
                pass
    with open(Path(out_dir) / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def find_embedding_files(root):
    root = Path(root)
    files = []
    for p in root.rglob('embeddings.npz'):
        files.append(p)
    for p in root.rglob('*.npz'):
        if p.name != 'embeddings.npz' and 'embedding' in p.name.lower():
            files.append(p)
    dedup = []
    seen = set()
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            dedup.append(p)
            seen.add(rp)
    return sorted(dedup)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', default=None, help='Single embedding file')
    parser.add_argument('--runs-dir', default=None, help='Root directory to recurse for embeddings.npz files')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pca-dim', type=int, default=50)
    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--n-neighbors', type=int, default=15)
    parser.add_argument('--min-dist', type=float, default=0.1)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    targets = []
    if args.embeddings:
        targets.append(Path(args.embeddings))
    if args.runs_dir:
        targets.extend(find_embedding_files(args.runs_dir))
    if not targets:
        raise ValueError('Provide --embeddings or --runs-dir')

    summaries = []
    seen = set()
    for emb in targets:
        emb = Path(emb)
        if emb.resolve() in seen:
            continue
        seen.add(emb.resolve())
        rel_name = sanitize(emb.parent.as_posix().replace('/', '__'))
        out_subdir = Path(args.output_dir) / rel_name
        summary = process_embedding_file(emb, out_subdir, seed=args.seed, pca_dim=args.pca_dim, perplexity=args.perplexity, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
        summary['analysis_dir'] = str(out_subdir)
        summaries.append(summary)

    pd.DataFrame(summaries).to_csv(Path(args.output_dir) / 'embedding_analysis_index.csv', index=False)
    with open(Path(args.output_dir) / 'summary.json', 'w') as f:
        json.dump({'n_embedding_files': len(summaries), 'files': summaries}, f, indent=2)
    print(json.dumps({'n_embedding_files': len(summaries)}, indent=2))


if __name__ == '__main__':
    main()
