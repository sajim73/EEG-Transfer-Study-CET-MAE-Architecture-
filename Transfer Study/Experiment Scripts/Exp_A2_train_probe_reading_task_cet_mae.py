#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import logging
import math
import random
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from torch.utils.data import DataLoader, Dataset

import model_mae_bart as model_mae_bart_module
from model_mae_bart import Pooler

if hasattr(model_mae_bart_module, "CETMAE_project_late_bart"):
    CETMAEprojectlatebart = getattr(
        model_mae_bart_module, "CETMAE_project_late_bart"
    )
else:
    raise ImportError(
        "model_mae_bart.py does not define CETMAE_project_late_bart; "
        f"available names: {[n for n in dir(model_mae_bart_module) if 'CET' in n or 'bart' in n]}"
    )


# ---------------------------------------------------------------------
# Args / logging / utils
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="A2: Frozen reading-task probe for CET-MAE (ZuCo 2.0 NR vs TSR, LOSO by subject)"
    )

    parser.add_argument(
        "--npz-nr", "--nr-npz",
        dest="npz_nr",
        type=str,
        required=True,
        help="ZuCo 2.0 NPZ for Normal Reading (NR)",
    )

    parser.add_argument(
        "--npz-tsr", "--tsr-npz",
        dest="npz_tsr",
        type=str,
        required=True,
        help="ZuCo 2.0 NPZ for Task-Specific Reading (TSR)",
    )

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-epochs", type=int, default=10)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument("--bart-path", type=str, default="facebook/bart-large")
    parser.add_argument("--head-type", type=str, default="linear")
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Accepted for CLI consistency; A2 is always a frozen probe.",
    )
    parser.add_argument(
        "--input-is-pooled",
        action="store_true",
        help="Input NPZ already contains pooled sentence embeddings of shape (N, D).",
    )
    parser.add_argument(
        "--input-is-token-hidden",
        action="store_true",
        help="Input NPZ contains token-level hidden states of shape (N, L, D); the script pools them with model_mae_bart.Pooler.",
    )

    args = parser.parse_args()
    if not args.freeze_encoder:
        parser.error("A2 is a frozen-probe experiment; pass --freeze-encoder to run this script.")
    return args


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("a2_frozen_reading_probe_zuco2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    fh = logging.FileHandler(output_dir / "run.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bootstrap_subject_metric_ci(values, n_boot=1000, alpha=0.95, seed=42):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": None,
            "ci_lower": None,
            "ci_upper": None,
            "n_boot": int(n_boot),
        }
    rng = np.random.default_rng(seed)
    n = arr.size
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boots[i] = rng.choice(arr, size=n, replace=True).mean()
    lower_q = (1.0 - alpha) / 2.0
    upper_q = 1.0 - lower_q
    return {
        "mean": float(arr.mean()),
        "ci_lower": float(np.quantile(boots, lower_q)),
        "ci_upper": float(np.quantile(boots, upper_q)),
        "n_boot": int(n_boot),
    }


def sanitize_name(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def normalize_id(x):
    if pd.isna(x):
        return None
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    if isinstance(x, (np.floating, float)):
        if float(x).is_integer():
            return str(int(x))
        return str(x).strip()
    s = str(x).strip()
    if re.fullmatch(r"\d+\.0+", s):
        return s.split(".")[0]
    return s


def normalize_text(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def first_existing_key(keys, candidates):
    keyset = {k.lower(): k for k in keys}
    for cand in candidates:
        if cand.lower() in keyset:
            return keyset[cand.lower()]
    return None


def to_numpy_array(x):
    arr = np.asarray(x)
    if arr.dtype == object:
        try:
            arr = np.stack(arr)
        except Exception:
            arr = np.array(list(arr))
    return arr


def ensure_2d_mask(mask, eeg):
    if mask is None:
        if eeg.ndim == 3:
            inferred = np.any(np.abs(eeg) > 0, axis=-1).astype(np.int64)
        elif eeg.ndim == 2:
            inferred = np.ones((eeg.shape[0], 1), dtype=np.int64)
        else:
            raise ValueError(f"Unsupported EEG shape for mask inference: {eeg.shape}")
        return inferred

    mask = to_numpy_array(mask)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)
    if mask.ndim == 1:
        mask = mask[:, None]
    return mask.astype(np.int64)


def pad_or_trim_mask(mask, seq_len):
    if mask.shape[1] == seq_len:
        return mask
    if mask.shape[1] > seq_len:
        return mask[:, :seq_len]
    pad = np.zeros((mask.shape[0], seq_len - mask.shape[1]), dtype=mask.dtype)
    return np.concatenate([mask, pad], axis=1)



# ---------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------
def load_npz_data(npz_path, logger):
    npz = np.load(npz_path, allow_pickle=True)
    keys = list(npz.files)
    logger.info(f"NPZ keys found in {npz_path}: {keys}")

    eeg_key = first_existing_key(
        keys,
        [
            "normalized_input_embeddings",
            "input_embeddings",
            "eeg",
            "embeddings",
            "features",
            "x",
            "X",
            "eeg_features",
            "eeg_embeddings",
            "hidden_states",
            "token_embeddings",
            "sentence_embeddings",
        ],
    )
    if eeg_key is None:
        raise KeyError(
            f"Could not find EEG/features array in {npz_path}. Available keys: {keys}"
        )

    mask_key = first_existing_key(
        keys,
        [
            "input_attn_mask",
            "attention_mask",
            "attn_mask",
            "mask",
            "eeg_attention_mask",
            "eeg_mask",
            "input_mask",
        ],
    )
    invert_mask_key = first_existing_key(keys, ["input_attn_mask_invert"])
    subject_key = first_existing_key(
        keys,
        [
            "subject_id",
            "subject",
            "subjects",
            "participant",
            "participant_id",
            "subj",
        ],
    )
    text_key = first_existing_key(
        keys,
        [
            "target_string",
            "sentence_text",
            "sentence",
            "text",
        ],
    )

    eeg = to_numpy_array(npz[eeg_key]).astype(np.float32)

    if mask_key is not None:
        mask = ensure_2d_mask(npz[mask_key], eeg)
    elif invert_mask_key is not None:
        inv_mask = to_numpy_array(npz[invert_mask_key]).astype(np.int64)
        if inv_mask.ndim == 3 and inv_mask.shape[-1] == 1:
            inv_mask = np.squeeze(inv_mask, axis=-1)
        if inv_mask.ndim == 1:
            inv_mask = inv_mask[:, None]
        mask = 1 - inv_mask
    else:
        mask = ensure_2d_mask(None, eeg)

    if eeg.ndim == 3:
        mask = pad_or_trim_mask(mask, eeg.shape[1])
    elif eeg.ndim == 2:
        if mask.shape[1] != 1:
            mask = np.ones((eeg.shape[0], 1), dtype=np.int64)
    else:
        raise ValueError(f"Unsupported EEG array shape: {eeg.shape}")

    if subject_key is None:
        raise KeyError(
            f"Could not find subject IDs in {npz_path}. Available keys: {keys}"
        )
    subjects = [normalize_id(x) for x in to_numpy_array(npz[subject_key]).reshape(-1)]

    if text_key is None:
        raise KeyError(
            f"Could not find sentence text key in NPZ. Available keys: {keys}"
        )
    texts = [normalize_text(x) for x in np.array(npz[text_key]).astype(str)]

    logger.info(f"Loaded EEG/features from key '{eeg_key}' with shape {tuple(eeg.shape)}")
    logger.info(f"Loaded attention mask shape: {tuple(mask.shape)}")
    logger.info(f"Text key: {text_key}; unique texts: {len(set(texts))}")

    return {
        "eeg": eeg,
        "mask": mask,
        "subject": np.array(subjects, dtype=object),
        "text": np.array(texts, dtype=object),
        "keys": keys,
    }


def subset_npz_by_subjects(npz_data, keep_subjects):
    keep_subjects = set(map(str, keep_subjects))
    subject_arr = np.array([str(x) for x in npz_data["subject"]], dtype=object)
    keep_mask = np.array([s in keep_subjects for s in subject_arr], dtype=bool)
    return {
        "eeg": npz_data["eeg"][keep_mask],
        "mask": npz_data["mask"][keep_mask],
        "subject": npz_data["subject"][keep_mask],
        "text": npz_data["text"][keep_mask],
        "keys": npz_data["keys"],
    }, keep_mask


def build_task_dataframe(npz_data, task_name):
    n = len(npz_data["eeg"])
    return pd.DataFrame(
        {
            "row_idx": np.arange(n),
            "subject": npz_data["subject"].astype(str),
            "text": npz_data["text"].astype(str),
            "task_name": task_name,
        }
    )


def combine_nr_tsr(npz_nr, npz_tsr, logger):
    nr_subjects = sorted(set(np.array(npz_nr["subject"]).astype(str).tolist()))
    tsr_subjects = sorted(set(np.array(npz_tsr["subject"]).astype(str).tolist()))
    common_subjects = sorted(set(nr_subjects).intersection(tsr_subjects))

    if len(common_subjects) < 2:
        raise ValueError(
            f"Need at least 2 common subjects for LOSO. Found {len(common_subjects)}."
        )

    logger.info(f"NR subjects: {nr_subjects}")
    logger.info(f"TSR subjects: {tsr_subjects}")
    logger.info(f"Using common subjects only ({len(common_subjects)}): {common_subjects}")

    npz_nr, nr_keep_mask = subset_npz_by_subjects(npz_nr, common_subjects)
    npz_tsr, tsr_keep_mask = subset_npz_by_subjects(npz_tsr, common_subjects)

    logger.info(
        f"After common-subject filtering | NR: {int(nr_keep_mask.sum())} samples | "
        f"TSR: {int(tsr_keep_mask.sum())} samples"
    )

    if npz_nr["eeg"].ndim != npz_tsr["eeg"].ndim:
        raise ValueError(
            f"NR and TSR EEG arrays have different ranks: {npz_nr['eeg'].ndim} vs {npz_tsr['eeg'].ndim}"
        )

    if npz_nr["eeg"].ndim == 3:
        if npz_nr["eeg"].shape[1:] != npz_tsr["eeg"].shape[1:]:
            raise ValueError(
                f"NR and TSR token shapes differ: {npz_nr['eeg'].shape[1:]} vs {npz_tsr['eeg'].shape[1:]}"
            )
        if npz_nr["mask"].shape[1] != npz_tsr["mask"].shape[1]:
            raise ValueError(
                f"NR and TSR mask lengths differ: {npz_nr['mask'].shape[1]} vs {npz_tsr['mask'].shape[1]}"
            )
    elif npz_nr["eeg"].ndim == 2:
        if npz_nr["eeg"].shape[1] != npz_tsr["eeg"].shape[1]:
            raise ValueError(
                f"NR and TSR feature dims differ: {npz_nr['eeg'].shape[1]} vs {npz_tsr['eeg'].shape[1]}"
            )
    else:
        raise ValueError(f"Unsupported EEG rank: {npz_nr['eeg'].ndim}")

    df_nr = build_task_dataframe(npz_nr, "NR")
    df_tsr = build_task_dataframe(npz_tsr, "TSR")
    df_nr["label"] = 0
    df_tsr["label"] = 1

    eeg = np.concatenate([npz_nr["eeg"], npz_tsr["eeg"]], axis=0).astype(np.float32)
    mask = np.concatenate([npz_nr["mask"], npz_tsr["mask"]], axis=0).astype(np.int64)
    meta = pd.concat([df_nr, df_tsr], axis=0, ignore_index=True)

    logger.info(f"Combined NR+TSR dataset size: {len(meta)}")
    logger.info(f"Task counts: {meta['task_name'].value_counts().to_dict()}")
    logger.info(f"Subjects in final dataset: {sorted(meta['subject'].astype(str).unique().tolist())}")

    return eeg, mask, meta, common_subjects


# ---------------------------------------------------------------------
# Checkpoint loading and frozen encoder
# ---------------------------------------------------------------------
def strip_prefixes(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ["module.", "model.", "backbone."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned


def load_checkpoint_state(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")
    for key in ["state_dict", "model_state_dict", "model", "net"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            return strip_prefixes(ckpt[key])
    return strip_prefixes(ckpt)


class FrozenCETMAEEncoder(nn.Module):
    def __init__(self, checkpoint_path, bart_path, device, logger):
        super().__init__()
        base = CETMAEprojectlatebart(pretrain_path=bart_path, device=str(device))
        state = load_checkpoint_state(checkpoint_path)
        missing, unexpected = base.load_state_dict(state, strict=False)

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Missing keys count: {len(missing)}")
        logger.info(f"Unexpected keys count: {len(unexpected)}")
        if len(missing) > 0:
            logger.info(f"Missing key sample: {missing[:10]}")
        if len(unexpected) > 0:
            logger.info(f"Unexpected key sample: {unexpected[:10]}")

        self.pos_embed_e = base.pos_embed_e
        self.e_branch = base.e_branch
        self.fc_eeg = base.fc_eeg
        self.act = base.act
        self.unify_branch = getattr(base, "unify_branch", None)

        self.input_dim = int(self.fc_eeg.in_features)
        self.output_dim = int(self.fc_eeg.out_features)

        for module in [self.pos_embed_e, self.e_branch, self.fc_eeg]:
            for p in module.parameters():
                p.requires_grad = False
        if self.unify_branch is not None:
            for p in self.unify_branch.parameters():
                p.requires_grad = False

        self.eval()

    def encode_raw_tokens(self, eeg, attention_mask):
        pad_mask = attention_mask == 0
        x = self.pos_embed_e(eeg)
        x = self.e_branch(x, src_key_padding_mask=pad_mask.bool())
        x = self.act(self.fc_eeg(x))
        if self.unify_branch is not None:
            try:
                x = self.unify_branch(
                    x, src_key_padding_mask=pad_mask.bool(), modality="e"
                )
            except TypeError:
                x = self.unify_branch(x, src_key_padding_mask=pad_mask.bool())
        pooled = Pooler(x, attention_mask.float())
        return pooled

    def mean_pool_hidden_tokens(self, x, attention_mask):
        return Pooler(x, attention_mask.float())


def extract_features(encoder, eeg, mask, args, device, logger):
    n = len(eeg)
    features = []
    encoder = encoder.to(device)
    encoder.eval()

    for start in range(0, n, args.feature_batch_size):
        end = min(start + args.feature_batch_size, n)

        eeg_b = torch.from_numpy(eeg[start:end]).float().to(device)
        mask_b = torch.from_numpy(mask[start:end]).long().to(device)

        with torch.no_grad():
            if args.input_is_pooled or eeg_b.ndim == 2:
                feat_b = eeg_b
            elif args.input_is_token_hidden:
                feat_b = encoder.mean_pool_hidden_tokens(eeg_b, mask_b)
            else:
                if eeg_b.ndim != 3:
                    raise ValueError(
                        f"Expected 3D raw EEG for encoder extraction, got shape {tuple(eeg_b.shape)}"
                    )
                if eeg_b.size(-1) == encoder.input_dim:
                    feat_b = encoder.encode_raw_tokens(eeg_b, mask_b)
                elif eeg_b.size(-1) == encoder.output_dim:
                    feat_b = encoder.mean_pool_hidden_tokens(eeg_b, mask_b)
                else:
                    raise ValueError(
                        f"Could not infer input mode. Last dim={eeg_b.size(-1)}, "
                        f"encoder input_dim={encoder.input_dim}, encoder output_dim={encoder.output_dim}. "
                        f"Use --input-is-pooled or --input-is-token-hidden if needed."
                    )

        features.append(feat_b.detach().cpu().numpy().astype(np.float32))

        batch_idx = start // args.feature_batch_size + 1
        if start == 0 or end == n or batch_idx % 20 == 0:
            logger.info(f"Feature extraction progress: {end}/{n}")

    features = np.concatenate(features, axis=0)
    logger.info(f"Extracted pooled feature matrix shape: {features.shape}")
    return features


# ---------------------------------------------------------------------
# Probe dataset / model / metrics
# ---------------------------------------------------------------------
class FeatureDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.indices = torch.from_numpy(indices).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.indices[idx]


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.head = nn.Linear(input_dim, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(x)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def split_train_val_subjects(trainval_subjects, val_ratio, seed):
    trainval_subjects = list(sorted(trainval_subjects))
    if len(trainval_subjects) < 2:
        raise ValueError("Need at least 2 train/val subjects after holding one out for test.")
    rng = random.Random(seed)
    rng.shuffle(trainval_subjects)
    n_val = max(1, int(round(len(trainval_subjects) * val_ratio)))
    n_val = min(n_val, len(trainval_subjects) - 1)
    val_subjects = sorted(trainval_subjects[:n_val])
    train_subjects = sorted(trainval_subjects[n_val:])
    return train_subjects, val_subjects


def run_epoch(model, loader, optimizer, device, train):
    criterion = nn.CrossEntropyLoss()
    model.train() if train else model.eval()

    running_loss = 0.0
    all_y = []
    all_pred = []
    all_prob = []
    all_idx = []

    for x, y, idx in loader:
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * x.size(0)
        all_y.append(y.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())
        all_prob.append(probs.detach().cpu().numpy())
        all_idx.append(idx.detach().cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    prob = np.concatenate(all_prob)
    idx = np.concatenate(all_idx)

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(running_loss / len(y_true))
    return metrics, y_true, y_pred, prob, idx


def evaluate_split(model, loader, device):
    metrics, y_true, y_pred, prob, idx = run_epoch(
        model, loader, optimizer=None, device=device, train=False
    )
    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": prob,
        "indices": idx,
    }


def summarize_metrics(df, metric_names):
    summary = {}
    for m in metric_names:
        summary[m] = {
            "mean": float(df[m].mean()),
            "std": float(df[m].std(ddof=1)) if len(df) > 1 else 0.0,
        }
    return summary


# ---------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------
def plot_training_curves(history, out_path, title):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val loss", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Val acc", linewidth=2)
    axes[1].plot(epochs, history["train_bal_acc"], label="Train bal acc", linewidth=2, linestyle="--")
    axes[1].plot(epochs, history["val_bal_acc"], label="Val bal acc", linewidth=2, linestyle="--")
    axes[1].plot(epochs, history["train_f1"], label="Train macro-F1", linewidth=2)
    axes[1].plot(epochs, history["val_f1"], label="Val macro-F1", linewidth=2)
    axes[1].set_title("Accuracy / Balanced Acc / Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, class_names, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(meta_df, out_path, title):
    dist = meta_df.groupby(["split", "task_name"]).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=dist, x="split", y="count", hue="task_name", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Split")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    dist.to_csv(out_path.with_suffix(".csv"), index=False)


def plot_subject_task_counts(meta_df, out_path, title):
    cnt = meta_df.groupby(["subject", "task_name"]).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * cnt["subject"].nunique()), 4.8))
    sns.barplot(data=cnt, x="subject", y="count", hue="task_name", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Samples")
    ax.grid(axis="y", alpha=0.25)
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
        tick.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    cnt.to_csv(out_path.with_suffix(".csv"), index=False)


def plot_probability_histogram(y_true, prob, class_names, out_path, title):
    if prob is None or prob.shape[1] != 2:
        return
    scores = prob[:, 1]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for lab in [0, 1]:
        idx = np.asarray(y_true) == lab
        ax.hist(scores[idx], bins=20, alpha=0.55, label=class_names[lab])
    ax.set_xlabel(f"Predicted probability of {class_names[1]}")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr_reliability(y_true, prob, out_prefix, positive_name="TSR"):
    if prob is None or prob.shape[1] != 2:
        return None

    scores = prob[:, 1]

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        str(out_prefix) + "_roc_curve.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(str(out_prefix) + "_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    pd.DataFrame({"precision": precision, "recall": recall}).to_csv(
        str(out_prefix) + "_pr_curve.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(recall, precision, label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall curve ({positive_name} positive)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(str(out_prefix) + "_pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    bins = np.linspace(0, 1, 11)
    bin_ids = np.digitize(scores, bins) - 1
    rows = []
    for b in range(10):
        idx = bin_ids == b
        if idx.sum() == 0:
            continue
        rows.append(
            {
                "bin_left": float(bins[b]),
                "bin_right": float(bins[b + 1]),
                "mean_pred": float(scores[idx].mean()),
                "empirical_pos_rate": float(np.asarray(y_true)[idx].mean()),
                "count": int(idx.sum()),
            }
        )
    rel = pd.DataFrame(rows)
    rel.to_csv(str(out_prefix) + "_reliability.csv", index=False)

    if not rel.empty:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.plot(rel["mean_pred"], rel["empirical_pos_rate"], marker="o")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed positive rate")
        ax.set_title("Reliability diagram")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(str(out_prefix) + "_reliability.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    return {"roc_auc": float(roc_auc), "average_precision": float(ap)}


def pca_2d(X):
    X = np.asarray(X, dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = U[:, :2] * S[:2]
    if coords.shape[1] == 1:
        coords = np.concatenate(
            [coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)], axis=1
        )
    return coords


def save_pca_plots(features, meta_df, out_prefix, color_columns):
    if len(features) < 2:
        return

    coords = pca_2d(features)
    out_df = meta_df.copy()
    out_df["pca1"] = coords[:, 0]
    out_df["pca2"] = coords[:, 1]
    out_df.to_csv(str(out_prefix) + "_pca2.csv", index=False)

    for col in color_columns:
        if col not in out_df.columns:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        vals = out_df[col].astype(str)
        uniq = vals.unique().tolist()
        if len(uniq) > 20:
            vc = vals.value_counts()
            keep = set(vc.index[:19])
            vals = vals.map(lambda z: z if z in keep else "OTHER")

        palette = sns.color_palette("tab20", n_colors=len(vals.unique()))
        for i, u in enumerate(vals.unique().tolist()):
            idx = vals == u
            ax.scatter(
                out_df.loc[idx, "pca1"],
                out_df.loc[idx, "pca2"],
                s=20,
                alpha=0.75,
                label=u,
                color=palette[i],
            )

        ax.set_xlabel("PCA-1")
        ax.set_ylabel("PCA-2")
        ax.set_title(f"PCA of pooled embeddings colored by {col}")
        ax.grid(alpha=0.2)
        if len(vals.unique()) <= 20:
            ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(str(out_prefix) + f"_pca_by_{col}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def save_eval_artifacts(eval_out, meta_subset, features_subset, split_dir, class_names):
    split_dir.mkdir(parents=True, exist_ok=True)

    metrics = dict(eval_out["metrics"])
    metrics["n"] = int(len(eval_out["y_true"]))
    save_json(split_dir / "metrics.json", metrics)

    pred_df = meta_subset.copy().reset_index(drop=True)
    pred_df["true_label_id"] = eval_out["y_true"]
    pred_df["pred_label_id"] = eval_out["y_pred"]
    pred_df["true_label_name"] = pred_df["true_label_id"].map({0: class_names[0], 1: class_names[1]})
    pred_df["pred_label_name"] = pred_df["pred_label_id"].map({0: class_names[0], 1: class_names[1]})

    if eval_out["probabilities"] is not None:
        pred_df[f"prob_{class_names[0]}"] = eval_out["probabilities"][:, 0]
        pred_df[f"prob_{class_names[1]}"] = eval_out["probabilities"][:, 1]

    pred_df.to_csv(split_dir / "predictions.csv", index=False)

    cm = confusion_matrix(eval_out["y_true"], eval_out["y_pred"], labels=[0, 1])
    pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in class_names],
        columns=[f"pred_{c}" for c in class_names],
    ).to_csv(split_dir / "confusion_matrix.csv")

    plot_confusion(cm, class_names, split_dir / "confusion_matrix.png", f"Confusion matrix ({split_dir.name})")
    plot_probability_histogram(
        eval_out["y_true"],
        eval_out["probabilities"],
        class_names,
        split_dir / "probability_histogram.png",
        f"Probability histogram ({split_dir.name})",
    )

    ranking = plot_roc_pr_reliability(
        eval_out["y_true"],
        eval_out["probabilities"],
        split_dir / split_dir.name,
        positive_name=class_names[1],
    )
    if ranking is not None:
        save_json(split_dir / "ranking_metrics.json", ranking)

    np.savez_compressed(
        split_dir / "embeddings.npz",
        embeddings=features_subset.astype(np.float32),
        y_true=np.asarray(eval_out["y_true"]),
        y_pred=np.asarray(eval_out["y_pred"]),
        subject=np.asarray(meta_subset["subject"].astype(str)),
        text=np.asarray(meta_subset["text"].astype(str)),
        task_name=np.asarray(meta_subset["task_name"].astype(str)),
        split=np.asarray(meta_subset["split"].astype(str)),
    )

    save_pca_plots(
        features_subset,
        pred_df,
        split_dir / "embeddings",
        color_columns=["true_label_name", "pred_label_name", "subject"],
    )


def save_subject_accuracy_plot(preds_df, out_csv, out_png):
    rows = []
    for subject, g in preds_df.groupby("held_out_subject"):
        rows.append(
            {
                "held_out_subject": subject,
                "n": int(len(g)),
                "accuracy": float((g["true_label_id"] == g["pred_label_id"]).mean()),
            }
        )
    subj_df = pd.DataFrame(rows).sort_values("held_out_subject")
    subj_df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(subj_df)), 4.2))
    ax.bar(subj_df["held_out_subject"].astype(str), subj_df["accuracy"])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Held-out subject")
    ax.set_ylabel("Accuracy")
    ax.set_title("Subject-wise test accuracy")
    ax.grid(axis="y", alpha=0.25)
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
        tick.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)
    set_seed(args.seed)

    if args.head_type.lower() != "linear":
        raise ValueError("This script currently implements only --head-type linear.")

    logger.info("Starting A2 frozen reading-task probe (ZuCo 2.0 NR vs TSR)")
    logger.info(json.dumps(vars(args), indent=2))

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    npz_nr = load_npz_data(args.npz_nr, logger)
    npz_tsr = load_npz_data(args.npz_tsr, logger)
    eeg, mask, meta, common_subjects = combine_nr_tsr(npz_nr, npz_tsr, logger)

    plot_subject_task_counts(
        meta,
        output_dir / "dataset_subject_task_counts.png",
        "ZuCo 2.0 subject/task sample counts",
    )
    save_json(output_dir / "label_map.json", {"NR": 0, "TSR": 1})
    meta.to_csv(output_dir / "combined_metadata.csv", index=False)

    encoder = FrozenCETMAEEncoder(args.checkpoint, args.bart_path, device, logger)
    features = extract_features(encoder, eeg, mask, args, device, logger)

    np.savez_compressed(
        output_dir / "all_pooled_features.npz",
        features=features.astype(np.float32),
        labels=meta["label"].values.astype(np.int64),
        subjects=meta["subject"].astype(str).values,
        task_name=meta["task_name"].astype(str).values,
        text=meta["text"].astype(str).values,
    )

    subjects = meta["subject"].astype(str).values
    labels = meta["label"].values.astype(np.int64)
    class_names = ["NR", "TSR"]
    subjects_unique = sorted(np.unique(subjects).tolist())

    logger.info(f"Running leave-one-subject-out CV over {len(subjects_unique)} subjects.")

    all_fold_rows = []
    all_pred_rows = []
    all_test_features = []
    all_test_meta = []
    all_test_y = []
    all_test_pred = []
    all_test_prob = []
    aggregate_cm = np.zeros((2, 2), dtype=np.int64)

    for fold_idx, held_out_subject in enumerate(subjects_unique, start=1):
        logger.info("=" * 88)
        logger.info(f"Fold {fold_idx}/{len(subjects_unique)} | held-out subject: {held_out_subject}")

        test_mask = subjects == held_out_subject
        trainval_subjects = [s for s in subjects_unique if s != held_out_subject]
        train_subjects, val_subjects = split_train_val_subjects(
            trainval_subjects, args.val_ratio, args.seed + fold_idx
        )

        train_mask = np.isin(subjects, train_subjects)
        val_mask = np.isin(subjects, val_subjects)

        fold_meta = meta.copy()
        fold_meta["split"] = np.where(test_mask, "test", np.where(val_mask, "val", "train"))

        fold_dir = output_dir / f"fold_{fold_idx:02d}_{sanitize_name(held_out_subject)}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_meta.to_csv(fold_dir / "split_manifest.csv", index=False)
        plot_class_distribution(
            fold_meta,
            fold_dir / "class_distribution.png",
            f"Class distribution by split (fold {fold_idx})",
        )

        X_train = features[train_mask]
        y_train = labels[train_mask]
        idx_train = np.where(train_mask)[0]

        X_val = features[val_mask]
        y_val = labels[val_mask]
        idx_val = np.where(val_mask)[0]

        X_test = features[test_mask]
        y_test = labels[test_mask]
        idx_test = np.where(test_mask)[0]

        logger.info(
            f"Train subjects ({len(train_subjects)}): {train_subjects} | "
            f"Val subjects ({len(val_subjects)}): {val_subjects} | "
            f"Test subject: {held_out_subject}"
        )
        logger.info(
            f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}"
        )
        logger.info(f"Train label counts: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        logger.info(f"Val label counts: {dict(zip(*np.unique(y_val, return_counts=True)))}")
        logger.info(f"Test label counts: {dict(zip(*np.unique(y_test, return_counts=True)))}")

        train_loader = DataLoader(
            FeatureDataset(X_train, y_train, idx_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            FeatureDataset(X_val, y_val, idx_val),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            FeatureDataset(X_test, y_test, idx_test),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model = LinearProbe(input_dim=features.shape[1], num_classes=2).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_f1": [],
            "val_f1": [],
            "train_bal_acc": [],
            "val_bal_acc": [],
        }

        best_state = None
        best_val_f1 = -math.inf
        best_epoch = -1

        for epoch in range(1, args.epochs + 1):
            train_metrics, *_ = run_epoch(model, train_loader, optimizer, device, train=True)
            val_metrics, *_ = run_epoch(model, val_loader, optimizer=None, device=device, train=False)

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["train_f1"].append(train_metrics["macro_f1"])
            history["val_f1"].append(val_metrics["macro_f1"])
            history["train_bal_acc"].append(train_metrics["balanced_accuracy"])
            history["val_bal_acc"].append(val_metrics["balanced_accuracy"])

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

            if epoch == 1 or epoch % args.log_every_epochs == 0 or epoch == args.epochs:
                logger.info(
                    f"[Fold {fold_idx} | Epoch {epoch:03d}] "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"train_acc={train_metrics['accuracy']:.4f} "
                    f"train_f1={train_metrics['macro_f1']:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_acc={val_metrics['accuracy']:.4f} "
                    f"val_f1={val_metrics['macro_f1']:.4f}"
                )

        if best_state is None:
            raise RuntimeError("No best probe state was captured during training.")

        torch.save(model.state_dict(), fold_dir / "last_probe.pt")
        torch.save(best_state, fold_dir / "best_probe.pt")

        pd.DataFrame(history).assign(
            epoch=np.arange(1, args.epochs + 1)
        ).to_csv(fold_dir / "history.csv", index=False)

        plot_training_curves(
            history,
            fold_dir / "training_curves.png",
            f"A2 Frozen Probe - Held-out Subject {held_out_subject} (best epoch {best_epoch})",
        )

        model.load_state_dict(best_state)

        for split_name, loader, mask_now in [
            ("train", train_loader, train_mask),
            ("val", val_loader, val_mask),
            ("test", test_loader, test_mask),
        ]:
            eval_out = evaluate_split(model, loader, device)
            meta_subset = fold_meta.loc[mask_now].copy().reset_index(drop=True)
            features_subset = features[mask_now]

            save_eval_artifacts(
                eval_out, meta_subset, features_subset, fold_dir / split_name, class_names
            )

            if split_name == "test":
                cm = confusion_matrix(eval_out["y_true"], eval_out["y_pred"], labels=[0, 1])
                aggregate_cm += cm

                logger.info(
                    f"[Fold {fold_idx} TEST] subject={held_out_subject} "
                    f"acc={eval_out['metrics']['accuracy']:.4f} "
                    f"bal_acc={eval_out['metrics']['balanced_accuracy']:.4f} "
                    f"macro_f1={eval_out['metrics']['macro_f1']:.4f}"
                )

                fold_row = {
                    "fold": fold_idx,
                    "held_out_subject": held_out_subject,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_f1,
                    "test_accuracy": eval_out["metrics"]["accuracy"],
                    "test_balanced_accuracy": eval_out["metrics"]["balanced_accuracy"],
                    "test_macro_f1": eval_out["metrics"]["macro_f1"],
                    "n_test": len(eval_out["y_true"]),
                }
                all_fold_rows.append(fold_row)

                pred_meta = meta_subset.copy().reset_index(drop=True)
                pred_meta["fold"] = fold_idx
                pred_meta["held_out_subject"] = held_out_subject
                pred_meta["true_label_id"] = eval_out["y_true"]
                pred_meta["pred_label_id"] = eval_out["y_pred"]
                pred_meta["true_label_name"] = pred_meta["true_label_id"].map(
                    {0: class_names[0], 1: class_names[1]}
                )
                pred_meta["pred_label_name"] = pred_meta["pred_label_id"].map(
                    {0: class_names[0], 1: class_names[1]}
                )
                pred_meta["prob_NR"] = eval_out["probabilities"][:, 0]
                pred_meta["prob_TSR"] = eval_out["probabilities"][:, 1]

                all_pred_rows.extend(pred_meta.to_dict("records"))
                all_test_features.append(features_subset)
                all_test_meta.append(pred_meta)
                all_test_y.append(eval_out["y_true"])
                all_test_pred.append(eval_out["y_pred"])
                all_test_prob.append(eval_out["probabilities"])

        save_json(
            fold_dir / "metrics.json",
            {
                "fold": fold_idx,
                "held_out_subject": held_out_subject,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_f1,
            },
        )

    folds_df = pd.DataFrame(all_fold_rows)
    preds_df = pd.DataFrame(all_pred_rows)

    folds_df.to_csv(output_dir / "per_subject_results.csv", index=False)
    preds_df.to_csv(output_dir / "all_predictions.csv", index=False)

    plot_confusion(
        aggregate_cm,
        class_names,
        output_dir / "overall_confusion_matrix.png",
        "Overall confusion matrix across LOSO folds",
    )

    save_subject_accuracy_plot(
        preds_df,
        output_dir / "subject_accuracy.csv",
        output_dir / "subject_accuracy.png",
    )

    overall_y = np.concatenate(all_test_y)
    overall_pred = np.concatenate(all_test_pred)
    overall_prob = np.concatenate(all_test_prob)
    overall_features = np.concatenate(all_test_features, axis=0)
    overall_meta = pd.concat(all_test_meta, ignore_index=True)

    plot_probability_histogram(
        overall_y,
        overall_prob,
        class_names,
        output_dir / "overall_probability_histogram.png",
        "Overall probability histogram",
    )

    ranking = plot_roc_pr_reliability(
        overall_y,
        overall_prob,
        output_dir / "overall",
        positive_name="TSR",
    )
    if ranking is not None:
        save_json(output_dir / "overall_ranking_metrics.json", ranking)

    save_pca_plots(
        overall_features,
        overall_meta,
        output_dir / "overall_embeddings",
        color_columns=[
            "true_label_name",
            "pred_label_name",
            "subject",
            "held_out_subject",
        ],
    )

    summary = summarize_metrics(
        folds_df.rename(
            columns={
                "test_accuracy": "accuracy",
                "test_balanced_accuracy": "balanced_accuracy",
                "test_macro_f1": "macro_f1",
            }
        ),
        metric_names=["accuracy", "balanced_accuracy", "macro_f1"],
    )
    summary["accuracy"]["bootstrap_ci"] = bootstrap_subject_metric_ci(
        folds_df["test_accuracy"].to_numpy(), n_boot=1000, seed=args.seed
    )
    summary["balanced_accuracy"]["bootstrap_ci"] = bootstrap_subject_metric_ci(
        folds_df["test_balanced_accuracy"].to_numpy(), n_boot=1000, seed=args.seed
    )
    summary["macro_f1"]["bootstrap_ci"] = bootstrap_subject_metric_ci(
        folds_df["test_macro_f1"].to_numpy(), n_boot=1000, seed=args.seed
    )

    save_json(
        output_dir / "summary.json",
        {
            "dataset": "ZuCo 2.0",
            "task_definition": {
                "negative_class": "NR",
                "positive_class": "TSR",
                "note": "This experiment uses within-ZuCo-2.0 Normal Reading vs Task-Specific Reading, not ZuCo-2.0 NR vs ZuCo-1.0 SR."
            },
            "num_subjects": len(subjects_unique),
            "subjects": subjects_unique,
            "common_subjects_used": common_subjects,
            "class_names": class_names,
            "summary_mean_std": summary,
            "aggregate_confusion_matrix": aggregate_cm.tolist(),
        },
    )

    logger.info("=" * 88)
    logger.info("Finished A2 frozen reading-task probe (ZuCo 2.0 NR vs TSR)")
    logger.info(f"Per-subject results saved to: {output_dir / 'per_subject_results.csv'}")
    logger.info(f"All predictions saved to: {output_dir / 'all_predictions.csv'}")
    logger.info(f"Summary saved to: {output_dir / 'summary.json'}")
    logger.info(
        "Final LOSO results | "
        f"accuracy={summary['accuracy']['mean']:.4f}±{summary['accuracy']['std']:.4f} | "
        f"balanced_accuracy={summary['balanced_accuracy']['mean']:.4f}±{summary['balanced_accuracy']['std']:.4f} | "
        f"macro_f1={summary['macro_f1']['mean']:.4f}±{summary['macro_f1']['std']:.4f}"
    )


if __name__ == "__main__":
    main()