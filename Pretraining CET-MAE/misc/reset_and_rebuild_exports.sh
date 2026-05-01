#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/sajim/CET-MAE"
TRANSFER_DIR="$ROOT/transfer_study"
EXPORTS_DIR="$TRANSFER_DIR/exports"

echo "==> Deleting old transfer exports"
rm -rf "$EXPORTS_DIR/embeddings_SR_pretrained_unlabeled"
rm -rf "$EXPORTS_DIR/embeddings_NR_pretrained_unlabeled"

mkdir -p "$EXPORTS_DIR/embeddings_SR_pretrained_unlabeled"
mkdir -p "$EXPORTS_DIR/embeddings_NR_pretrained_unlabeled"

echo "==> Rebuilding clean SR/NR exports (no split at export time)"

cd "$ROOT"

python3 - <<'PY'
import os
import csv
import json
import h5py
import torch
import numpy as np
import scipy.io as io
import importlib.util
from glob import glob
from pathlib import Path

ROOT = Path("/home/sajim/CET-MAE").resolve()
os.chdir(ROOT)

TRANSFER_DIR = ROOT / "transfer_study"
EXPORTS_DIR = TRANSFER_DIR / "exports"

SR_OUT = EXPORTS_DIR / "embeddings_SR_pretrained_unlabeled"
NR_OUT = EXPORTS_DIR / "embeddings_NR_pretrained_unlabeled"

BANDS = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']
EEG_TYPE = "GD"
MAX_LEN = 58
DIM = 105
FULL_DIM = DIM * len(BANDS)

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod_v1 = load_module("data2pickle_v1_mod", ROOT / "data2pickle_v1.py")
dh = load_module("data_loading_helpers_modified_mod", ROOT / "data_loading_helpers_modified.py")

def get_attr_any(obj, names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AttributeError(f"None of these attributes found: {names}")

load_matlab_string = get_attr_any(dh, ["load_matlab_string", "loadmatlabstring"])
extract_word_level_data = get_attr_any(dh, ["extract_word_level_data", "extractwordleveldata"])

def resolve_local_model_dir(*candidates):
    for c in candidates:
        p = Path(c)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(
        "Could not find local tokenizer/model directory. Tried: "
        + ", ".join(str((ROOT / c).resolve()) if not Path(c).is_absolute() else str(Path(c)) for c in candidates)
    )

def tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def is_missing_word_data(word_data):
    if word_data is None:
        return True
    if isinstance(word_data, float):
        return True
    if isinstance(word_data, np.ndarray) and word_data.size == 0:
        return True
    return False

def scalar_int(x, default=0):
    if x is None:
        return default
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return default
        return int(np.asarray(x).reshape(-1)[0])
    try:
        return int(x)
    except Exception:
        return default

def safe_seq_len(seq_len_value):
    if isinstance(seq_len_value, list):
        return len(seq_len_value)
    if isinstance(seq_len_value, tuple):
        return len(seq_len_value)
    if isinstance(seq_len_value, np.ndarray):
        return len(seq_len_value.reshape(-1).tolist())
    try:
        return int(seq_len_value)
    except Exception:
        return 0

def normalize_1d(arr):
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    std = arr.std()
    if std == 0 or np.isnan(std) or np.isinf(std):
        std = 1.0
    mean = arr.mean()
    return (arr - mean) / std

def normalize_2d(mat):
    mat = np.asarray(mat, dtype=np.float32)
    flat = mat.reshape(-1)
    std = flat.std()
    if std == 0 or np.isnan(std) or np.isinf(std):
        std = 1.0
    mean = flat.mean()
    return (mat - mean) / std

def finalize_and_save(records, out_dir, dataset_name):
    if not records:
        raise RuntimeError(f"No records created for {dataset_name}")

    input_embeddings = np.stack([r["input_embeddings"] for r in records], axis=0)
    normalized_input_embeddings = np.stack([r["normalized_input_embeddings"] for r in records], axis=0)
    input_attn_mask = np.stack([r["input_attn_mask"] for r in records], axis=0)
    input_attn_mask_invert = np.stack([r["input_attn_mask_invert"] for r in records], axis=0)
    seq_len = np.asarray([r["seq_len"] for r in records], dtype=np.int64)
    target_string = np.asarray([r["target_string"] for r in records], dtype=object)
    subject_id = np.asarray([r["subject_id"] for r in records], dtype=object)
    sentence_index = np.asarray([r["sentence_index"] for r in records], dtype=np.int64)
    task_name = np.asarray([r["task_name"] for r in records], dtype=object)
    source_file = np.asarray([r["source_file"] for r in records], dtype=object)

    np.savez_compressed(
        out_dir / "embeddings.npz",
        input_embeddings=input_embeddings,
        normalized_input_embeddings=normalized_input_embeddings,
        input_attn_mask=input_attn_mask,
        input_attn_mask_invert=input_attn_mask_invert,
        seq_len=seq_len,
        target_string=target_string,
        subject_id=subject_id,
        sentence_index=sentence_index,
        task_name=task_name,
        source_file=source_file,
    )

    with open(out_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_idx", "subject_id", "sentence_index", "task_name", "source_file", "target_string", "seq_len"])
        for i, r in enumerate(records):
            writer.writerow([
                i, r["subject_id"], r["sentence_index"], r["task_name"],
                r["source_file"], r["target_string"], r["seq_len"]
            ])

    summary = {
        "dataset_name": dataset_name,
        "n_samples": len(records),
        "input_shape": list(input_embeddings.shape),
        "note": "Full export with subject metadata; no train/val/test split applied here."
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] {dataset_name}: {len(records)} samples -> {out_dir/'embeddings.npz'}")

def build_sr_export():
    sr_dir = ROOT / "zuco_dataset" / "task1-SR" / "Matlab_files"
    mat_files = sorted(glob(str(sr_dir / "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No SR .mat files found in {sr_dir}")

    bart_dir = resolve_local_model_dir("models/huggingface/bart-large", "./models/huggingface/bart-large")
    tokenizer = mod_v1.BartTokenizer.from_pretrained(str(bart_dir), local_files_only=True)

    records = []

    for mat_file in mat_files:
        subject_id = os.path.basename(mat_file).split("_")[0].replace("results", "").strip()
        matdata = io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)["sentenceData"]

        for sent_idx, sent in enumerate(matdata):
            word_data = sent.word
            if is_missing_word_data(word_data):
                continue

            sent_obj = {"content": sent.content}
            sent_obj["sentence_level_EEG"] = {
                "mean_t1": sent.mean_t1,
                "mean_t2": sent.mean_t2,
                "mean_a1": sent.mean_a1,
                "mean_a2": sent.mean_a2,
                "mean_b1": sent.mean_b1,
                "mean_b2": sent.mean_b2,
                "mean_g1": sent.mean_g1,
                "mean_g2": sent.mean_g2,
            }

            sent_obj["word"] = []
            word_tokens_has_fixation = []
            word_tokens_with_mask = []
            word_tokens_all = []

            words_iter = word_data if isinstance(word_data, (list, tuple, np.ndarray)) else [word_data]

            for word in words_iter:
                word_obj = {"content": word.content}
                word_tokens_all.append(word.content)

                nfix = scalar_int(getattr(word, "nFixations", 0), default=0)
                word_obj["nFixations"] = nfix

                if nfix > 0:
                    word_obj["word_level_EEG"] = {
                        "FFD": {
                            "FFD_t1": word.FFD_t1, "FFD_t2": word.FFD_t2, "FFD_a1": word.FFD_a1, "FFD_a2": word.FFD_a2,
                            "FFD_b1": word.FFD_b1, "FFD_b2": word.FFD_b2, "FFD_g1": word.FFD_g1, "FFD_g2": word.FFD_g2,
                        },
                        "TRT": {
                            "TRT_t1": word.TRT_t1, "TRT_t2": word.TRT_t2, "TRT_a1": word.TRT_a1, "TRT_a2": word.TRT_a2,
                            "TRT_b1": word.TRT_b1, "TRT_b2": word.TRT_b2, "TRT_g1": word.TRT_g1, "TRT_g2": word.TRT_g2,
                        },
                        "GD": {
                            "GD_t1": word.GD_t1, "GD_t2": word.GD_t2, "GD_a1": word.GD_a1, "GD_a2": word.GD_a2,
                            "GD_b1": word.GD_b1, "GD_b2": word.GD_b2, "GD_g1": word.GD_g1, "GD_g2": word.GD_g2,
                        }
                    }
                    sent_obj["word"].append(word_obj)
                    word_tokens_has_fixation.append(word.content)
                    word_tokens_with_mask.append(word.content)
                else:
                    word_tokens_with_mask.append("[MASK]")

            if len(sent_obj["word"]) == 0:
                continue

            sent_obj["word_tokens_has_fixation"] = word_tokens_has_fixation
            sent_obj["word_tokens_with_mask"] = word_tokens_with_mask
            sent_obj["word_tokens_all"] = word_tokens_all

            sample = mod_v1.get_input_sample(
                sent_obj,
                tokenizer=tokenizer,
                eeg_type=EEG_TYPE,
                bands=BANDS,
                max_len=MAX_LEN,
                dim=DIM,
            )
            if sample is None:
                continue

            records.append({
                "input_embeddings": tensor_to_numpy(sample["input_embeddings"]).astype(np.float32),
                "normalized_input_embeddings": tensor_to_numpy(sample["normalized_input_embeddings"]).astype(np.float32),
                "input_attn_mask": tensor_to_numpy(sample["input_attn_mask"]).astype(np.int64),
                "input_attn_mask_invert": tensor_to_numpy(sample["input_attn_mask_invert"]).astype(np.int64),
                "seq_len": safe_seq_len(sample["seq_len"]),
                "target_string": str(sample["target_string"]),
                "subject_id": subject_id,
                "sentence_index": int(sent_idx),
                "task_name": "task1-SR",
                "source_file": os.path.basename(mat_file),
            })

    finalize_and_save(records, SR_OUT, "SR")

def valid_sentence_band(x):
    arr = np.asarray(x).reshape(-1)
    return arr.size == DIM and not np.isnan(arr).any() and not np.isinf(arr).any()

def valid_band_105(x):
    arr = np.asarray(x).reshape(-1)
    return arr.size == DIM and not np.isnan(arr).any() and not np.isinf(arr).any()

def build_nr_sample_manual(sent_string, sent_eeg_dict, word_items):
    sent_features = []
    for band in BANDS:
        key = f"mean{band}"
        arr = np.asarray(sent_eeg_dict[key]).reshape(-1)
        if not valid_band_105(arr):
            return None
        sent_features.append(arr.astype(np.float32))

    sent_vec = np.concatenate(sent_features, axis=0).astype(np.float32)

    word_vecs = []

    for _, data_dict in sorted(word_items.items(), key=lambda kv: kv[0]):
        if "content" not in data_dict or "GD_EEG" not in data_dict:
            continue

        gd = data_dict["GD_EEG"]
        if len(gd) != 8:
            continue

        gd_bands = []
        ok = True
        for band_arr in gd:
            arr = np.asarray(band_arr).reshape(-1)
            if not valid_band_105(arr):
                ok = False
                break
            gd_bands.append(arr.astype(np.float32))

        if not ok:
            continue

        word_text = str(data_dict["content"]).strip()
        if len(word_text) == 0:
            continue

        word_vec = np.concatenate(gd_bands, axis=0).astype(np.float32)
        if word_vec.size != FULL_DIM:
            continue

        word_vecs.append(word_vec)

    if len(word_vecs) == 0:
        return None

    word_vecs.append(sent_vec)
    nonnorm = np.stack(word_vecs, axis=0).astype(np.float32)
    norm = normalize_2d(nonnorm).astype(np.float32)

    seq_len = nonnorm.shape[0]
    if seq_len > MAX_LEN:
        nonnorm = nonnorm[:MAX_LEN]
        norm = norm[:MAX_LEN]
        seq_len = MAX_LEN

    if seq_len < MAX_LEN:
        pad = np.zeros((MAX_LEN - seq_len, FULL_DIM), dtype=np.float32)
        nonnorm = np.concatenate([nonnorm, pad], axis=0)
        norm = np.concatenate([norm, pad], axis=0)

    input_attn_mask = np.zeros((MAX_LEN,), dtype=np.int64)
    input_attn_mask[:seq_len] = 1

    input_attn_mask_invert = np.ones((MAX_LEN,), dtype=np.int64)
    input_attn_mask_invert[:seq_len] = 0

    return {
        "input_embeddings": nonnorm,
        "normalized_input_embeddings": norm,
        "input_attn_mask": input_attn_mask,
        "input_attn_mask_invert": input_attn_mask_invert,
        "seq_len": seq_len,
        "target_string": str(sent_string),
    }

def build_nr_export():
    nr_dir = ROOT / "zuco_dataset" / "task2-NR-2.0" / "Matlab_files"
    mat_files = sorted(glob(str(nr_dir / "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No NR .mat files found in {nr_dir}")

    records = []
    kept_sentences = 0
    skipped_no_words = 0
    skipped_bad_sentence_eeg = 0
    skipped_bad_sample = 0

    for mat_file in mat_files:
        subject_id = os.path.basename(mat_file).split("_")[0].replace("results", "").strip()

        with h5py.File(mat_file, "r") as f:
            sentence_data = f["sentenceData"]
            mean_t1_objs = sentence_data["mean_t1"]
            mean_t2_objs = sentence_data["mean_t2"]
            mean_a1_objs = sentence_data["mean_a1"]
            mean_a2_objs = sentence_data["mean_a2"]
            mean_b1_objs = sentence_data["mean_b1"]
            mean_b2_objs = sentence_data["mean_b2"]
            mean_g1_objs = sentence_data["mean_g1"]
            mean_g2_objs = sentence_data["mean_g2"]
            raw_data = sentence_data["rawData"]
            content_data = sentence_data["content"]
            word_data = sentence_data["word"]

            for sent_idx in range(len(raw_data)):
                content_ref = content_data[sent_idx][0]
                sent_string = load_matlab_string(f[content_ref])

                sent_eeg_dict = {
                    "mean_t1": np.squeeze(f[mean_t1_objs[sent_idx][0]][()]),
                    "mean_t2": np.squeeze(f[mean_t2_objs[sent_idx][0]][()]),
                    "mean_a1": np.squeeze(f[mean_a1_objs[sent_idx][0]][()]),
                    "mean_a2": np.squeeze(f[mean_a2_objs[sent_idx][0]][()]),
                    "mean_b1": np.squeeze(f[mean_b1_objs[sent_idx][0]][()]),
                    "mean_b2": np.squeeze(f[mean_b2_objs[sent_idx][0]][()]),
                    "mean_g1": np.squeeze(f[mean_g1_objs[sent_idx][0]][()]),
                    "mean_g2": np.squeeze(f[mean_g2_objs[sent_idx][0]][()]),
                }

                if not all(valid_sentence_band(v) for v in sent_eeg_dict.values()):
                    skipped_bad_sentence_eeg += 1
                    continue

                word_ref = f[word_data[sent_idx][0]]
                word_items, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask = extract_word_level_data(f, word_ref)

                if word_items == {} or len(word_tokens_all) == 0:
                    skipped_no_words += 1
                    continue

                sample = build_nr_sample_manual(sent_string, sent_eeg_dict, word_items)
                if sample is None:
                    skipped_bad_sample += 1
                    continue

                kept_sentences += 1

                records.append({
                    "input_embeddings": sample["input_embeddings"].astype(np.float32),
                    "normalized_input_embeddings": sample["normalized_input_embeddings"].astype(np.float32),
                    "input_attn_mask": sample["input_attn_mask"].astype(np.int64),
                    "input_attn_mask_invert": sample["input_attn_mask_invert"].astype(np.int64),
                    "seq_len": int(sample["seq_len"]),
                    "target_string": str(sample["target_string"]),
                    "subject_id": subject_id,
                    "sentence_index": int(sent_idx),
                    "task_name": "task2-NR-2.0",
                    "source_file": os.path.basename(mat_file),
                })

    print(f"[NR] kept_sentences={kept_sentences}")
    print(f"[NR] skipped_no_words={skipped_no_words}")
    print(f"[NR] skipped_bad_sentence_eeg={skipped_bad_sentence_eeg}")
    print(f"[NR] skipped_bad_sample={skipped_bad_sample}")

    finalize_and_save(records, NR_OUT, "NR")

build_sr_export()
build_nr_export()

print("\nDone.")
PY

echo "==> Finished"
echo "SR export: $EXPORTS_DIR/embeddings_SR_pretrained_unlabeled/embeddings.npz"
echo "NR export: $EXPORTS_DIR/embeddings_NR_pretrained_unlabeled/embeddings.npz"
