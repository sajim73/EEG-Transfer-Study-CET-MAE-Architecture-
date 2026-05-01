#!/usr/bin/env python3
import os
import json
import pickle
import argparse
from collections import Counter, defaultdict

import numpy as np

def safe_len(x):
    try:
        return len(x)
    except Exception:
        return None

def safe_shape(x):
    try:
        arr = np.array(x)
        return list(arr.shape)
    except Exception:
        return None

def short_text(s, n=300):
    s = str(s)
    return s[:n] + ("..." if len(s) > n else "")

def summarize_value(v):
    info = {
        "python_type": str(type(v)),
        "len": safe_len(v),
        "shape": None,
        "dtype": None,
        "preview": None,
    }

    if isinstance(v, str):
        info["preview"] = short_text(v, 300)
        return info

    if isinstance(v, dict):
        info["preview"] = {"dict_keys": list(v.keys())[:30]}
        return info

    if isinstance(v, (list, tuple)):
        info["preview"] = f"{type(v).__name__} len={len(v)}"
        if len(v) > 0:
            first = v[0]
            try:
                arr0 = np.array(first)
                info["preview_first_item_type"] = str(type(first))
                info["preview_first_item_shape"] = list(arr0.shape)
                info["preview_first_item_dtype"] = str(arr0.dtype)
            except Exception:
                info["preview_first_item_type"] = str(type(first))
        try:
            arr = np.array(v)
            info["shape"] = list(arr.shape)
            info["dtype"] = str(arr.dtype)
        except Exception:
            pass
        return info

    try:
        arr = np.array(v)
        info["shape"] = list(arr.shape)
        info["dtype"] = str(arr.dtype)
        if arr.ndim == 0:
            info["preview"] = arr.item()
        elif arr.ndim == 1:
            info["preview"] = arr[:10].tolist()
        else:
            info["preview"] = f"array ndim={arr.ndim}"
        return info
    except Exception:
        info["preview"] = short_text(repr(v), 300)
        return info

def find_candidate_text_keys(sample):
    cands = []
    for k, v in sample.items():
        if isinstance(v, str):
            cands.append((k, "string"))
        elif isinstance(v, list) and len(v) > 0 and all(isinstance(x, str) for x in v[: min(5, len(v))]):
            cands.append((k, "list_of_strings"))
        elif isinstance(v, dict):
            keys = set(v.keys())
            if {"input_ids", "attention_mask"} & keys:
                cands.append((k, "tokenized_dict"))
    return cands

def find_candidate_eeg_keys(sample):
    cands = []
    for k, v in sample.items():
        try:
            arr = np.array(v)
            if arr.ndim >= 2:
                last = arr.shape[-1]
                if last in [840, 832, 768, 105]:
                    cands.append((k, list(arr.shape), str(arr.dtype)))
                elif arr.ndim == 2 and arr.shape[0] <= 100 and arr.shape[1] <= 2000:
                    cands.append((k, list(arr.shape), str(arr.dtype)))
            elif isinstance(v, list) and len(v) > 0:
                first = np.array(v[0])
                if first.ndim >= 1:
                    cands.append((k, f"list_first_shape={list(first.shape)}", str(first.dtype)))
        except Exception:
            pass
    return cands

def inspect_sample(path):
    with open(path, "rb") as f:
        x = pickle.load(f)

    report = {
        "file": path,
        "top_level_type": str(type(x)),
    }

    if not isinstance(x, dict):
        report["value_preview"] = short_text(repr(x), 1000)
        return report

    report["keys"] = sorted(list(x.keys()))
    report["fields"] = {}
    for k, v in x.items():
        report["fields"][k] = summarize_value(v)

    report["candidate_text_keys"] = find_candidate_text_keys(x)
    report["candidate_eeg_keys"] = find_candidate_eeg_keys(x)

    return report

def gather_files(folder):
    if not os.path.isdir(folder):
        return []
    files = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".pickle") or name.endswith(".pkl"):
            files.append(os.path.join(folder, name))
    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./datasets/data_word_sentence_5")
    parser.add_argument("--out", type=str, default="./pickle_schema_report.json")
    parser.add_argument("--max-per-split", type=int, default=3)
    args = parser.parse_args()

    splits = ["train", "valid", "test"]
    final_report = {
        "root": os.path.abspath(args.root),
        "splits": {},
        "global_key_counter": {},
        "notes": [
            "Goal: recover schema needed for EEG_dataset_add_sentence_mae.",
            "Need likely text field, likely EEG field, shapes, and whether tokenized text already exists.",
        ],
    }

    global_key_counter = Counter()

    for split in splits:
        folder = os.path.join(args.root, split)
        files = gather_files(folder)
        split_report = {
            "folder": os.path.abspath(folder),
            "num_files": len(files),
            "sampled_files": [],
            "key_counter": {},
        }

        for fp in files[: args.max_per_split]:
            try:
                rep = inspect_sample(fp)
                split_report["sampled_files"].append(rep)
                if "keys" in rep:
                    global_key_counter.update(rep["keys"])
            except Exception as e:
                split_report["sampled_files"].append({
                    "file": fp,
                    "error": str(e),
                })

        split_report["key_counter"] = dict(global_key_counter)
        final_report["splits"][split] = split_report

    final_report["global_key_counter"] = dict(global_key_counter)

    with open(args.out, "w") as f:
        json.dump(final_report, f, indent=2)

    print("=" * 80)
    print("PICKLE SCHEMA REPORT")
    print("=" * 80)
    print("Root:", final_report["root"])
    print("Saved JSON report to:", os.path.abspath(args.out))
    print()

    for split in splits:
        sr = final_report["splits"].get(split, {})
        print(f"[{split}] num_files={sr.get('num_files', 0)}")
        for sample in sr.get("sampled_files", []):
            print("-" * 80)
            print("FILE:", sample.get("file"))
            if "error" in sample:
                print("ERROR:", sample["error"])
                continue
            print("TYPE:", sample.get("top_level_type"))
            print("KEYS:", sample.get("keys", []))
            print("CANDIDATE TEXT KEYS:", sample.get("candidate_text_keys", []))
            print("CANDIDATE EEG KEYS:", sample.get("candidate_eeg_keys", []))

            fields = sample.get("fields", {})
            for k in sample.get("keys", []):
                meta = fields.get(k, {})
                print(f"  {k}:")
                print(f"    type   = {meta.get('python_type')}")
                print(f"    len    = {meta.get('len')}")
                print(f"    shape  = {meta.get('shape')}")
                print(f"    dtype  = {meta.get('dtype')}")
                print(f"    preview= {meta.get('preview')}")
        print()

if __name__ == "__main__":
    main()
