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
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Robust import of CETMAE_project_late_bart and Pooler
# ---------------------------------------------------------------------
import model_mae_bart as model_mae_bart_module

try:
    from model_mae_bart import Pooler
except ImportError as e:
    raise ImportError(
        "Could not import Pooler from model_mae_bart.py. "
        "Please verify that model_mae_bart.py defines Pooler."
    ) from e

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
# Argument parsing, logging, and utilities
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="A2: Frozen relation-control probe for CET-MAE (LOSO by subject, trial-level)"
    )
    parser.add_argument("--embeddings-npz", type=str, required=True)
    parser.add_argument("--labels-csv", type=str, required=True)
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
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument("--bart-path", type=str, default="facebook/bart-large")

    parser.add_argument(
        "--sentence-col",
        type=str,
        default="sentence",
        help="Sentence/text column in the A2 CSV.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="control",
        help="Binary label column in the A2 CSV.",
    )
    parser.add_argument(
        "--subject-col",
        type=str,
        default=None,
        help="Optional subject column in labels CSV; if missing, use NPZ subject_id.",
    )

    parser.add_argument(
        "--input-is-pooled",
        action="store_true",
        help="NPZ already contains pooled 1024-d sentence embeddings",
    )
    parser.add_argument(
        "--input-is-token-hidden",
        action="store_true",
        help="NPZ contains token-level hidden states (N,L,1024); only mean-pool them",
    )

    return parser.parse_args()


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("a2_frozen_probe")
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
# NPZ + labels loading, alignment by text
# ---------------------------------------------------------------------
def load_npz_data(npz_path, logger):
    npz = np.load(npz_path, allow_pickle=True)
    keys = list(npz.files)
    logger.info(f"NPZ keys found: {keys}")

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
        mask_raw = npz[mask_key]
        mask = ensure_2d_mask(mask_raw, eeg)
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

    subjects = None
    if subject_key is not None:
        subjects = [normalize_id(x) for x in to_numpy_array(npz[subject_key]).reshape(-1)]

    if text_key is None:
        raise KeyError(
            f"Could not find sentence text key in NPZ. Available keys: {keys}"
        )

    texts = [normalize_text(x) for x in np.array(npz[text_key]).astype(str)]

    logger.info(
        f"Loaded EEG/features array from key '{eeg_key}' with shape {tuple(eeg.shape)}"
    )
    logger.info(f"Loaded attention mask shape: {tuple(mask.shape)}")
    logger.info(f"Subject IDs present: {subjects is not None}")
    logger.info(f"Text key: {text_key}; unique texts: {len(set(texts))}")

    return {
        "eeg": eeg,
        "mask": mask,
        "subject": subjects,
        "text": np.array(texts, dtype=object),
        "keys": keys,
    }


def normalize_control_label(x):
    if pd.isna(x):
        return "non_control"

    s = str(x).strip().lower()

    if s in {"", "nan", "none", "null", "0", "false", "no", "non_control", "non-control", "noncontrol"}:
        return "non_control"
    if s in {"control", "1", "true", "yes"}:
        return "control"

    return None


def load_labels(csv_path, sentence_col, label_col, subject_col, logger):
    last_err = None
    for enc in ["utf-8", "cp1252", "latin-1"]:
        try:
            df = pd.read_csv(csv_path, comment="#", sep=";", encoding=enc)
            logger.info(
                f"Loaded labels CSV with shape {df.shape} from {csv_path} using encoding={enc}"
            )
            break
        except UnicodeDecodeError as e:
            last_err = e
    else:
        raise last_err

    logger.info(f"CSV columns: {list(df.columns)}")

    if sentence_col not in df.columns:
        alt = first_existing_key(df.columns, [sentence_col, "sentence", "text"])
        if alt is None:
            raise KeyError(
                f"Could not find sentence column '{sentence_col}' in CSV columns: {list(df.columns)}"
            )
        sentence_col = alt

    if label_col not in df.columns:
        alt = first_existing_key(df.columns, [label_col, "control", "label"])
        if alt is None:
            raise KeyError(
                f"Could not find label column '{label_col}' in CSV columns: {list(df.columns)}"
            )
        label_col = alt

    if subject_col is None:
        subject_col = first_existing_key(
            df.columns,
            ["subject", "subject_id", "participant", "participant_id", "subj"],
        )
    elif subject_col not in df.columns:
        alt = first_existing_key(
            df.columns,
            [subject_col, "subject", "subject_id", "participant", "participant_id", "subj"],
        )
        subject_col = alt

    out = pd.DataFrame(
        {
            "text": df[sentence_col].map(normalize_text),
            "control_raw": df[label_col].map(normalize_control_label),
        }
    )

    if subject_col is not None and subject_col in df.columns:
        out["subject"] = df[subject_col].map(normalize_id)

    out = out.dropna(subset=["text", "control_raw"]).copy()

    conflict_check = (
        out.groupby("text")["control_raw"].nunique(dropna=True).reset_index(name="n")
    )
    conflicts = conflict_check[conflict_check["n"] > 1]
    if len(conflicts) > 0:
        raise ValueError(
            f"Found {len(conflicts)} sentence texts with conflicting A2 labels."
        )

    out = out.drop_duplicates(subset=["text"]).copy()

    logger.info(
        f"Labels DataFrame after cleaning: {out.shape[0]} rows, "
        f"{out['text'].nunique()} unique texts"
    )
    logger.info(
        f"Unique control_raw values in labels: {sorted(out['control_raw'].dropna().unique().tolist())}"
    )

    return out


def build_merged_table(npz_data, labels_df, logger):
    n = len(npz_data["eeg"])
    meta = pd.DataFrame(
        {
            "row_idx": np.arange(n),
            "subject_npz": npz_data["subject"],
            "text": npz_data["text"].astype(str),
        }
    )

    merged = meta.merge(labels_df, on="text", how="left")

    n_labeled = merged["control_raw"].notna().sum()
    n_unlabeled = merged["control_raw"].isna().sum()
    logger.info(
        f"After text-merge: labeled trials = {n_labeled}, unlabeled trials = {n_unlabeled}"
    )

    if n_labeled == 0:
        unique_npz_texts = meta["text"].nunique()
        unique_label_texts = labels_df["text"].nunique()
        logger.error(
            "No trials matched any labels after text-merge. "
            f"Unique NPZ texts = {unique_npz_texts}, unique label texts = {unique_label_texts}."
        )
        raise ValueError("Merged trial-level dataset is empty (no labeled trials).")

    merged = merged[merged["control_raw"].notna()].copy()

    if "subject" not in merged.columns or merged["subject"].isna().all():
        merged["subject"] = merged["subject_npz"]

    if merged["subject"].isna().any():
        missing_subj = int(merged["subject"].isna().sum())
        raise ValueError(
            "Subject IDs are required for LOSO. "
            f"Could not build a complete subject column. Missing subjects in {missing_subj} rows."
        )

    logger.info(
        f"Unique control_raw values before label mapping: {sorted(merged['control_raw'].dropna().unique().tolist())}"
    )

    class_names = ["non_control", "control"]
    label_to_id = {"non_control": 0, "control": 1}

    merged["label"] = merged["control_raw"].map(label_to_id)
    merged = merged.dropna(subset=["label"]).copy()
    merged["label"] = merged["label"].astype(int)

    if merged.empty:
        raise ValueError(
            "Merged dataset became empty after label mapping. "
            "Check normalize_control_label and label_to_id."
        )

    logger.info(f"Final merged trial-level dataset size: {len(merged)}")
    logger.info(
        f"Subjects: {sorted(merged['subject'].astype(str).unique().tolist())}"
    )
    logger.info(
        f"Label counts: {merged['control_raw'].value_counts().to_dict()}"
    )

    return merged.reset_index(drop=True), class_names


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
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return strip_prefixes(ckpt[key])
    if isinstance(ckpt, dict):
        return strip_prefixes(ckpt)
    raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")


class FrozenEEGEncoder(nn.Module):
    def __init__(self, checkpoint_path, bart_path, device, logger):
        super().__init__()
        model = CETMAEprojectlatebart(pretrain_path=bart_path, device=str(device))
        state = load_checkpoint_state(checkpoint_path)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Missing keys count: {len(missing)}")
        logger.info(f"Unexpected keys count: {len(unexpected)}")
        if len(missing) > 0:
            logger.info(f"Missing key sample: {missing[:10]}")
        if len(unexpected) > 0:
            logger.info(f"Unexpected key sample: {unexpected[:10]}")

        self.posembede = model.pos_embed_e
        self.ebranch = model.e_branch
        self.fceeg = model.fc_eeg
        self.act = model.act
        self.unifybranch = model.unify_branch

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, eeg, attention_mask):
        if eeg.ndim != 3:
            raise ValueError(f"Expected EEG input of shape (N,L,D), got {tuple(eeg.shape)}")

        pad_mask = attention_mask == 0

        x = self.posembede(eeg)
        x = self.ebranch(x, src_key_padding_mask=pad_mask.bool())
        x = self.act(self.fceeg(x))
        x = self.unifybranch(x, src_key_padding_mask=pad_mask.bool(), modality="e")

        pooled = Pooler(x, attention_mask.float())
        return pooled


def masked_mean_pool(x, mask):
    denom = mask.sum(axis=1, keepdims=True).clip(min=1.0)
    return (x * mask[:, :, None]).sum(axis=1) / denom


def extract_features(npz_data, args, device, output_dir, logger):
    eeg = npz_data["eeg"]
    mask = npz_data["mask"].astype(np.float32)

    if len(eeg) == 0:
        raise ValueError("No EEG rows were provided to extract_features().")

    if args.input_is_pooled or (eeg.ndim == 2 and eeg.shape[1] == 1024):
        logger.info("Detected pooled 1024-d sentence features. Using them directly.")
        feats = eeg.astype(np.float32)

    elif args.input_is_token_hidden or (eeg.ndim == 3 and eeg.shape[-1] == 1024):
        logger.info(
            "Detected token-level 1024-d hidden states. Applying mean-pooling only."
        )
        feats = masked_mean_pool(eeg.astype(np.float32), mask.astype(np.float32)).astype(
            np.float32
        )

    else:
        logger.info(
            "Extracting frozen CET-MAE features from raw EEG token features using checkpoint encoder."
        )
        encoder = FrozenEEGEncoder(args.checkpoint, args.bart_path, device, logger).to(device)
        encoder.eval()

        eeg_tensor = torch.from_numpy(eeg).float()
        mask_tensor = torch.from_numpy(mask).long()

        feats_list = []
        loader = DataLoader(
            list(range(len(eeg_tensor))),
            batch_size=args.feature_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        with torch.no_grad():
            pbar = tqdm(loader, desc="Feature extraction", leave=True, dynamic_ncols=True)
            for batch_ids in pbar:
                batch_eeg = eeg_tensor[batch_ids].to(device)
                batch_mask = mask_tensor[batch_ids].to(device)
                batch_feats = encoder(batch_eeg, batch_mask)
                feats_list.append(batch_feats.cpu())
                done = int(sum(x.shape[0] for x in feats_list))
                pbar.set_postfix({"done": done})

        if len(feats_list) == 0:
            raise ValueError("Feature extraction produced no batches.")

        feats = torch.cat(feats_list, dim=0).numpy().astype(np.float32)

    np.save(output_dir / "pooled_features.npy", feats)
    logger.info(
        f"Saved pooled features to {output_dir / 'pooled_features.npy'} with shape {feats.shape}"
    )

    return feats


# ---------------------------------------------------------------------
# Dataset, model head, metrics
# ---------------------------------------------------------------------
class FeatureDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class LinearProbe(nn.Module):
    def __init__(self, in_dim=1024, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def safe_train_val_split(y, val_ratio, seed):
    idx = np.arange(len(y))
    _, counts = np.unique(y, return_counts=True)
    stratify = y if np.all(counts >= 2) else None
    return train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=stratify)


def run_epoch(model, loader, optimizer, device, train, logger, log_every, epoch_idx):
    criterion = nn.CrossEntropyLoss()
    if train:
        model.train()
        desc = f"Train epoch {epoch_idx}"
    else:
        model.eval()
        desc = f"Val epoch {epoch_idx}"

    running_loss = 0.0
    all_y = []
    all_pred = []

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for step, (x, y) in enumerate(pbar, start=1):
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits, dim=1)
        running_loss += loss.item() * x.size(0)

        all_y.append(y.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())

        if step % log_every == 0 or step == len(loader):
            cur_y = np.concatenate(all_y)
            cur_pred = np.concatenate(all_pred)
            cur_acc = accuracy_score(cur_y, cur_pred)
            pbar.set_postfix(
                loss=f"{running_loss / len(cur_y):.4f}", acc=f"{cur_acc:.4f}"
            )

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(running_loss / len(y_true))
    return metrics


def evaluate_test(model, loader, device):
    model.eval()
    all_y = []
    all_pred = []
    all_logits = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_y.append(y.numpy())
            all_pred.append(preds.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    logits = np.concatenate(all_logits)
    metrics = compute_metrics(y_true, y_pred)
    return y_true, y_pred, logits, metrics


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
    axes[1].plot(epochs, history["train_f1"], label="Train macro-F1", linewidth=2)
    axes[1].plot(epochs, history["val_f1"], label="Val macro-F1", linewidth=2)
    axes[1].set_title("Accuracy / Macro-F1")
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


def summarize_metrics(df, metric_names):
    summary = {}
    for m in metric_names:
        summary[m] = {
            "mean": float(df[m].mean()),
            "std": float(df[m].std(ddof=1)) if len(df) > 1 else 0.0,
        }
    return summary


# ---------------------------------------------------------------------
# Main A2 LOSO pipeline
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)
    set_seed(args.seed)

    logger.info("Starting A2 frozen relation-control probe (trial-level, text-joined)")
    logger.info(json.dumps(vars(args), indent=2))

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    npz_data = load_npz_data(args.embeddings_npz, logger)
    labels_df = load_labels(
        args.labels_csv, args.sentence_col, args.label_col, args.subject_col, logger
    )
    merged, class_names = build_merged_table(npz_data, labels_df, logger)

    eeg = npz_data["eeg"][merged["row_idx"].values]
    mask = npz_data["mask"][merged["row_idx"].values]
    subjects = merged["subject"].astype(str).values
    labels = merged["label"].values.astype(np.int64)
    texts = merged["text"].astype(str).values

    filtered_npz_data = {
        "eeg": eeg,
        "mask": mask,
    }

    features = extract_features(filtered_npz_data, args, device, output_dir, logger)

    if features.ndim != 2:
        raise ValueError(f"Expected pooled feature matrix of shape (N,D), got {features.shape}")

    if features.shape[1] != 1024:
        logger.warning(
            f"Feature dim is {features.shape[1]}, not 1024. The linear head will use this dim as-is."
        )

    subjects_unique = sorted(np.unique(subjects).tolist())
    logger.info(f"Running leave-one-subject-out CV over {len(subjects_unique)} subjects.")

    all_fold_rows = []
    all_pred_rows = []
    aggregate_cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)

    for fold_idx, held_out_subject in enumerate(subjects_unique, start=1):
        logger.info("=" * 88)
        logger.info(
            f"Fold {fold_idx}/{len(subjects_unique)} | held-out subject: {held_out_subject}"
        )

        test_mask = subjects == held_out_subject
        trainval_mask = ~test_mask

        X_trainval = features[trainval_mask]
        y_trainval = labels[trainval_mask]
        X_test = features[test_mask]
        y_test = labels[test_mask]
        test_texts = texts[test_mask]

        train_idx, val_idx = safe_train_val_split(
            y_trainval, args.val_ratio, args.seed + fold_idx
        )

        X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
        X_val, y_val = X_trainval[val_idx], y_trainval[val_idx]

        logger.info(
            f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}"
        )
        logger.info(
            f"Train label counts: {dict(zip(*np.unique(y_train, return_counts=True)))}"
        )
        logger.info(
            f"Val label counts: {dict(zip(*np.unique(y_val, return_counts=True)))}"
        )
        logger.info(
            f"Test label counts: {dict(zip(*np.unique(y_test, return_counts=True)))}"
        )

        train_loader = DataLoader(
            FeatureDataset(X_train, y_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            FeatureDataset(X_val, y_val),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            FeatureDataset(X_test, y_test),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model = LinearProbe(in_dim=features.shape[1], num_classes=len(class_names)).to(device)
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
        }

        best_state = None
        best_val_f1 = -math.inf
        best_epoch = -1

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, optimizer, device, True, logger, args.log_every, epoch
            )
            val_metrics = run_epoch(
                model, val_loader, optimizer, device, False, logger, args.log_every, epoch
            )

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["train_f1"].append(train_metrics["macro_f1"])
            history["val_f1"].append(val_metrics["macro_f1"])

            logger.info(
                f"[Fold {fold_idx} | Epoch {epoch:03d}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"train_f1={train_metrics['macro_f1']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['macro_f1']:.4f}"
            )

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                logger.info(
                    f"New best model at epoch {epoch} with val_macro_f1={best_val_f1:.4f}"
                )

        model.load_state_dict(best_state)

        fold_dir = output_dir / f"fold_{fold_idx:02d}_{sanitize_name(held_out_subject)}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        torch.save(best_state, fold_dir / "best_linear_probe.pt")
        plot_training_curves(
            history,
            fold_dir / "training_curves.png",
            f"A2 Frozen Probe - Subject {held_out_subject} (best epoch {best_epoch})",
        )

        y_true, y_pred, logits, test_metrics = evaluate_test(model, test_loader, device)
        cm = confusion_matrix(
            y_true, y_pred, labels=list(range(len(class_names)))
        )

        aggregate_cm += cm

        plot_confusion(
            cm,
            class_names,
            fold_dir / "confusion_matrix.png",
            f"Confusion Matrix - Held-out Subject {held_out_subject}",
        )

        logger.info(
            f"[Fold {fold_idx} TEST] subject={held_out_subject} "
            f"acc={test_metrics['accuracy']:.4f} "
            f"bal_acc={test_metrics['balanced_accuracy']:.4f} "
            f"macro_f1={test_metrics['macro_f1']:.4f}"
        )

        fold_row = {
            "fold": fold_idx,
            "held_out_subject": held_out_subject,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "n_test": len(y_true),
        }
        all_fold_rows.append(fold_row)

        for i in range(len(y_true)):
            all_pred_rows.append(
                {
                    "fold": fold_idx,
                    "held_out_subject": held_out_subject,
                    "text": test_texts[i],
                    "true_label_id": int(y_true[i]),
                    "pred_label_id": int(y_pred[i]),
                    "true_label_name": class_names[int(y_true[i])],
                    "pred_label_name": class_names[int(y_pred[i])],
                }
            )

        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "fold_info": fold_row,
                    "class_names": class_names,
                    "confusion_matrix": cm.tolist(),
                    "history": history,
                },
                f,
                indent=2,
            )

    folds_df = pd.DataFrame(all_fold_rows)
    preds_df = pd.DataFrame(all_pred_rows)

    folds_df.to_csv(output_dir / "per_subject_results.csv", index=False)
    preds_df.to_csv(output_dir / "all_predictions.csv", index=False)

    plot_confusion(
        aggregate_cm,
        class_names,
        output_dir / "overall_confusion_matrix.png",
        "Overall Confusion Matrix Across LOSO Folds",
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

    with open(output_dir / "summary.json", "w") as f:
        json.dump(
            {
                "num_subjects": len(subjects_unique),
                "class_names": class_names,
                "summary_mean_std": summary,
                "aggregate_confusion_matrix": aggregate_cm.tolist(),
            },
            f,
            indent=2,
        )

    logger.info("=" * 88)
    logger.info("Finished A2 frozen relation-control probe")
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
