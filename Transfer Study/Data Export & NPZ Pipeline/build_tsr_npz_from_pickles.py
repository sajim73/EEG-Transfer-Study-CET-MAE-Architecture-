#!/usr/bin/env python3
import os
import sys
import csv
import json
import shutil
import pickle
import numpy as np
from glob import glob
from pathlib import Path


ROOT = Path("/home/sajim/CET-MAE").resolve()
os.chdir(ROOT)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRANSFER_DIR = ROOT / "transfer_study"
EXPORTS_DIR = TRANSFER_DIR / "exports"
TSR_OUT = EXPORTS_DIR / "embeddings_TSR_pretrained_unlabeled"

# Change this only if your TSR pickles are somewhere else.
TSR_PICKLE_ROOT = ROOT / "datasets" / "data_word_sentence_5"

TASK_NAME = "task3-TSR-2.0"


def tensor_to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def parse_pickle_metadata(pickle_path: Path):
    split_name = pickle_path.parent.name
    stem = pickle_path.stem

    # Example:
    # v2-task3-TSR-2.0-ZAB-17
    prefix, subject_id, sentence_index = stem.rsplit("-", 2)

    return {
        "split_name": split_name,
        "subject_id": subject_id,
        "sentence_index": int(sentence_index),
        "source_prefix": prefix,
        "pickle_file": pickle_path.name,
    }


def finalize_and_save(records, out_dir):
    if not records:
        raise RuntimeError("No TSR pickle records found.")

    input_embeddings = np.stack([r["input_embeddings"] for r in records], axis=0)
    normalized_input_embeddings = np.stack([r["normalized_input_embeddings"] for r in records], axis=0)
    input_attn_mask = np.stack([r["input_attn_mask"] for r in records], axis=0)
    input_attn_mask_invert = np.stack([r["input_attn_mask_invert"] for r in records], axis=0)

    seq_len = np.asarray([r["seq_len"] for r in records], dtype=np.int64)
    target_string = np.asarray([r["target_string"] for r in records], dtype=object)
    subject_id = np.asarray([r["subject_id"] for r in records], dtype=object)
    sentence_index = np.asarray([r["sentence_index"] for r in records], dtype=np.int64)
    task_name = np.asarray([r["task_name"] for r in records], dtype=object)
    split_name = np.asarray([r["split_name"] for r in records], dtype=object)
    pickle_file = np.asarray([r["pickle_file"] for r in records], dtype=object)
    source_prefix = np.asarray([r["source_prefix"] for r in records], dtype=object)

    savez_kwargs = dict(
        input_embeddings=input_embeddings,
        normalized_input_embeddings=normalized_input_embeddings,
        input_attn_mask=input_attn_mask,
        input_attn_mask_invert=input_attn_mask_invert,
        seq_len=seq_len,
        target_string=target_string,
        subject_id=subject_id,
        sentence_index=sentence_index,
        task_name=task_name,
        split_name=split_name,
        pickle_file=pickle_file,
        source_prefix=source_prefix,
    )

    if all("target_id_has_fixation" in r for r in records):
        savez_kwargs["target_id_has_fixation"] = np.stack(
            [r["target_id_has_fixation"] for r in records], axis=0
        )

    if all("seq_len_has_fixation" in r for r in records):
        savez_kwargs["seq_len_has_fixation"] = np.asarray(
            [r["seq_len_has_fixation"] for r in records], dtype=np.int64
        )

    if all("word_tokens_has_fixation" in r for r in records):
        savez_kwargs["word_tokens_has_fixation"] = np.asarray(
            [r["word_tokens_has_fixation"] for r in records], dtype=object
        )

    if all("selected_words" in r for r in records):
        savez_kwargs["selected_words"] = np.asarray(
            [r["selected_words"] for r in records], dtype=object
        )

    np.savez_compressed(out_dir / "embeddings.npz", **savez_kwargs)

    with open(out_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "row_idx",
            "subject_id",
            "sentence_index",
            "task_name",
            "split_name",
            "pickle_file",
            "target_string",
            "seq_len",
        ])
        for i, r in enumerate(records):
            writer.writerow([
                i,
                r["subject_id"],
                r["sentence_index"],
                r["task_name"],
                r["split_name"],
                r["pickle_file"],
                r["target_string"],
                r["seq_len"],
            ])

    summary = {
        "dataset_name": "TSR",
        "task_name": TASK_NAME,
        "n_samples": len(records),
        "input_shape": list(input_embeddings.shape),
        "note": "Built directly from TSR pickle files; preserves split metadata and TSR fixation fields when available.",
        "source_root": str(TSR_PICKLE_ROOT),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def build_tsr_export():
    if not TSR_PICKLE_ROOT.exists():
        raise FileNotFoundError(f"TSR pickle root not found: {TSR_PICKLE_ROOT}")

    split_dirs = ["train", "valid", "test"]
    pickle_paths = []
    for split in split_dirs:
        pickle_paths.extend(sorted((TSR_PICKLE_ROOT / split).glob("*.pickle")))

    if not pickle_paths:
        raise FileNotFoundError(f"No TSR .pickle files found under {TSR_PICKLE_ROOT}")

    if TSR_OUT.exists():
        shutil.rmtree(TSR_OUT)
    TSR_OUT.mkdir(parents=True, exist_ok=True)

    records = []
    skipped_none = 0
    skipped_missing_keys = 0

    required_keys = [
        "input_embeddings",
        "normalized_input_embeddings",
        "input_attn_mask",
        "input_attn_mask_invert",
        "target_string",
    ]

    for pickle_path in pickle_paths:
        with open(pickle_path, "rb") as f:
            sample = pickle.load(f)

        if sample is None:
            skipped_none += 1
            continue

        if any(k not in sample for k in required_keys):
            skipped_missing_keys += 1
            continue

        meta = parse_pickle_metadata(pickle_path)

        attn_mask = tensor_to_numpy(sample["input_attn_mask"]).astype(np.int64)
        true_seq_len = int(attn_mask.sum())

        record = {
            "input_embeddings": tensor_to_numpy(sample["input_embeddings"]).astype(np.float32),
            "normalized_input_embeddings": tensor_to_numpy(sample["normalized_input_embeddings"]).astype(np.float32),
            "input_attn_mask": attn_mask,
            "input_attn_mask_invert": tensor_to_numpy(sample["input_attn_mask_invert"]).astype(np.int64),
            "seq_len": true_seq_len,
            "target_string": str(sample["target_string"]),
            "subject_id": meta["subject_id"],
            "sentence_index": meta["sentence_index"],
            "task_name": TASK_NAME,
            "split_name": meta["split_name"],
            "pickle_file": meta["pickle_file"],
            "source_prefix": meta["source_prefix"],
        }

        if "target_id_has_fixation" in sample:
            record["target_id_has_fixation"] = np.asarray(sample["target_id_has_fixation"], dtype=np.int64)

        if "seq_len_has_fixation" in sample:
            record["seq_len_has_fixation"] = int(sample["seq_len_has_fixation"])

        if "word_tokens_has_fixation" in sample:
            record["word_tokens_has_fixation"] = list(sample["word_tokens_has_fixation"])

        if "selected_words" in sample:
            record["selected_words"] = list(sample["selected_words"])

        records.append(record)

    print(f"[TSR] kept_samples={len(records)}")
    print(f"[TSR] skipped_none={skipped_none}")
    print(f"[TSR] skipped_missing_keys={skipped_missing_keys}")

    finalize_and_save(records, TSR_OUT)


if __name__ == "__main__":
    build_tsr_export()
    print(f"[OK] Wrote: {TSR_OUT / 'embeddings.npz'}")
