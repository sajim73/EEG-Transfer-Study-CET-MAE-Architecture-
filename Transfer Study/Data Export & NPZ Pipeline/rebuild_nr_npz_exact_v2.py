#!/usr/bin/env python3
import os
import sys
import csv
import json
import shutil
import h5py
import numpy as np
import importlib.util
from glob import glob
from pathlib import Path

ROOT = Path("/home/sajim/CET-MAE").resolve()
os.chdir(ROOT)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRANSFER_DIR = ROOT / "transfer_study"
EXPORTS_DIR = TRANSFER_DIR / "exports"
NR_OUT = EXPORTS_DIR / "embeddings_NR_pretrained_unlabeled"
NR_DIR = ROOT / "zuco_dataset" / "task2-NR-2.0" / "Matlab_files"

MAX_LEN = 58
DIM = 105
BANDS = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']
EEG_TYPE = "GD"

BART_DIR = ROOT / "models" / "huggingface" / "bart-large"
PEGASUS_DIR = ROOT / "models" / "huggingface" / "pegasus-large"


def load_module(name, path, register_as=None):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    if register_as is not None:
        sys.modules[register_as] = mod
    spec.loader.exec_module(mod)
    return mod


dh = load_module(
    "dh_mod",
    ROOT / "data_loading_helpers_modified.py",
    register_as="data_loading_helpers_modified",
)

v2 = load_module("v2_mod", ROOT / "data2pickle_v2.py")


def resolve_tokenizer():
    if PEGASUS_DIR.exists():
        tok = v2.PegasusTokenizer.from_pretrained(str(PEGASUS_DIR), local_files_only=True)
        return tok, "pegasus-large", str(PEGASUS_DIR)
    if BART_DIR.exists():
        tok = v2.BartTokenizer.from_pretrained(str(BART_DIR), local_files_only=True)
        return tok, "bart-large", str(BART_DIR)
    raise FileNotFoundError(
        f"Missing tokenizer dirs: tried {PEGASUS_DIR} and {BART_DIR}"
    )


def tensor_to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def finalize_and_save(records, out_dir, dataset_name, tokenizer_name, tokenizer_path):
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
        writer.writerow([
            "row_idx", "subject_id", "sentence_index", "task_name",
            "source_file", "target_string", "seq_len"
        ])
        for i, r in enumerate(records):
            writer.writerow([
                i,
                r["subject_id"],
                r["sentence_index"],
                r["task_name"],
                r["source_file"],
                r["target_string"],
                r["seq_len"],
            ])

    summary = {
        "dataset_name": dataset_name,
        "n_samples": len(records),
        "input_shape": list(input_embeddings.shape),
        "note": "Full export with subject metadata; no train/val/test split applied here.",
        "tokenizer_name": tokenizer_name,
        "tokenizer_path": tokenizer_path,
        "preprocessing_reference": "data2pickle_v2.get_input_sample",
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def rebuild_sent_obj_from_nr_file(f, sent_idx):
    sentence_data = f["sentenceData"]

    mean_t1_objs = sentence_data["mean_t1"]
    mean_t2_objs = sentence_data["mean_t2"]
    mean_a1_objs = sentence_data["mean_a1"]
    mean_a2_objs = sentence_data["mean_a2"]
    mean_b1_objs = sentence_data["mean_b1"]
    mean_b2_objs = sentence_data["mean_b2"]
    mean_g1_objs = sentence_data["mean_g1"]
    mean_g2_objs = sentence_data["mean_g2"]
    content_data = sentence_data["content"]
    word_data = sentence_data["word"]

    content_ref = content_data[sent_idx][0]
    sent_string = dh.load_matlab_string(f[content_ref])

    sent_obj = {"content": sent_string}
    sent_obj["sentence_level_EEG"] = {
        "mean_t1": np.squeeze(f[mean_t1_objs[sent_idx][0]][()]),
        "mean_t2": np.squeeze(f[mean_t2_objs[sent_idx][0]][()]),
        "mean_a1": np.squeeze(f[mean_a1_objs[sent_idx][0]][()]),
        "mean_a2": np.squeeze(f[mean_a2_objs[sent_idx][0]][()]),
        "mean_b1": np.squeeze(f[mean_b1_objs[sent_idx][0]][()]),
        "mean_b2": np.squeeze(f[mean_b2_objs[sent_idx][0]][()]),
        "mean_g1": np.squeeze(f[mean_g1_objs[sent_idx][0]][()]),
        "mean_g2": np.squeeze(f[mean_g2_objs[sent_idx][0]][()]),
    }

    sent_obj["word"] = []

    word_ref = f[word_data[sent_idx][0]]
    word_level_data, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask = dh.extract_word_level_data(
        f, word_ref
    )

    if word_level_data == {} or len(word_tokens_all) == 0:
        return None, "no_words"

    for widx in range(len(word_level_data)):
        data_dict = word_level_data[widx]
        word_obj = {"content": data_dict["content"], "nFixations": data_dict["nFix"]}

        if "GD_EEG" in data_dict:
            gd = data_dict["GD_EEG"]
            ffd = data_dict["FFD_EEG"]
            trt = data_dict["TRT_EEG"]

            if len(gd) == len(ffd) == len(trt) == 8:
                word_obj["word_level_EEG"] = {
                    "GD": {
                        "GD_t1": gd[0], "GD_t2": gd[1], "GD_a1": gd[2], "GD_a2": gd[3],
                        "GD_b1": gd[4], "GD_b2": gd[5], "GD_g1": gd[6], "GD_g2": gd[7],
                    },
                    "FFD": {
                        "FFD_t1": ffd[0], "FFD_t2": ffd[1], "FFD_a1": ffd[2], "FFD_a2": ffd[3],
                        "FFD_b1": ffd[4], "FFD_b2": ffd[5], "FFD_g1": ffd[6], "FFD_g2": ffd[7],
                    },
                    "TRT": {
                        "TRT_t1": trt[0], "TRT_t2": trt[1], "TRT_a1": trt[2], "TRT_a2": trt[3],
                        "TRT_b1": trt[4], "TRT_b2": trt[5], "TRT_g1": trt[6], "TRT_g2": trt[7],
                    },
                }
                sent_obj["word"].append(word_obj)

    sent_obj["word_tokens_has_fixation"] = word_tokens_has_fixation
    sent_obj["word_tokens_with_mask"] = word_tokens_with_mask
    sent_obj["word_tokens_all"] = word_tokens_all

    if len(sent_obj["word"]) == 0:
        return None, "no_valid_words"

    return sent_obj, None


def build_nr_export():
    if not NR_DIR.exists():
        raise FileNotFoundError(f"NR Matlab directory not found: {NR_DIR}")

    mat_files = sorted(glob(str(NR_DIR / "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No NR .mat files found in {NR_DIR}")

    if NR_OUT.exists():
        shutil.rmtree(NR_OUT)
    NR_OUT.mkdir(parents=True, exist_ok=True)

    tokenizer, tokenizer_name, tokenizer_path = resolve_tokenizer()

    records = []
    skipped_no_words = 0
    skipped_no_valid_words = 0
    skipped_bad_sample = 0

    for mat_file in mat_files:
        subject_id = os.path.basename(mat_file).split("_")[0].replace("results", "").strip()

        with h5py.File(mat_file, "r") as f:
            sentence_data = f["sentenceData"]
            n_sentences = len(sentence_data["content"])

            for sent_idx in range(n_sentences):
                sent_obj, err = rebuild_sent_obj_from_nr_file(f, sent_idx)

                if sent_obj is None:
                    if err == "no_words":
                        skipped_no_words += 1
                    else:
                        skipped_no_valid_words += 1
                    continue

                sample = v2.get_input_sample(
                    sent_obj,
                    tokenizer=tokenizer,
                    eeg_type=EEG_TYPE,
                    bands=BANDS,
                    max_len=MAX_LEN,
                    dim=DIM,
                    add_CLS_token=False,
                )

                if sample is None:
                    skipped_bad_sample += 1
                    continue

                attn_mask = tensor_to_numpy(sample["input_attn_mask"]).astype(np.int64)
                true_seq_len = int(attn_mask.sum())

                records.append({
                    "input_embeddings": tensor_to_numpy(sample["input_embeddings"]).astype(np.float32),
                    "normalized_input_embeddings": tensor_to_numpy(sample["normalized_input_embeddings"]).astype(np.float32),
                    "input_attn_mask": attn_mask,
                    "input_attn_mask_invert": tensor_to_numpy(sample["input_attn_mask_invert"]).astype(np.int64),
                    "seq_len": true_seq_len,
                    "target_string": str(sample["target_string"]),
                    "subject_id": subject_id,
                    "sentence_index": int(sent_idx),
                    "task_name": "task2-NR-2.0",
                    "source_file": os.path.basename(mat_file),
                })

    print(f"[NR] kept_sentences={len(records)}")
    print(f"[NR] skipped_no_words={skipped_no_words}")
    print(f"[NR] skipped_no_valid_words={skipped_no_valid_words}")
    print(f"[NR] skipped_bad_sample={skipped_bad_sample}")

    finalize_and_save(records, NR_OUT, "NR", tokenizer_name, tokenizer_path)


if __name__ == "__main__":
    build_nr_export()
    print(f"[OK] Wrote: {NR_OUT / 'embeddings.npz'}")
