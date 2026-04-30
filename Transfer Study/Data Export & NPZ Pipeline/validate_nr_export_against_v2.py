#!/usr/bin/env python3
import os
import sys
import json
import h5py
import numpy as np
import importlib.util
from pathlib import Path

ROOT = Path("/home/sajim/CET-MAE").resolve()
os.chdir(ROOT)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NPZ_PATH = ROOT / "transfer_study" / "exports" / "embeddings_NR_pretrained_unlabeled" / "embeddings.npz"
MAT_DIR = ROOT / "zuco_dataset" / "task2-NR-2.0" / "Matlab_files"

BART_DIR = ROOT / "models" / "huggingface" / "bart-large"
PEGASUS_DIR = ROOT / "models" / "huggingface" / "pegasus-large"

N_CHECK = 8
SEED = 42
ATOL = 1e-6
RTOL = 1e-6


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
    if BART_DIR.exists():
        tokenizer = v2.BartTokenizer.from_pretrained(str(BART_DIR), local_files_only=True)
        return tokenizer, str(BART_DIR), "bart-large"
    if PEGASUS_DIR.exists():
        tokenizer = v2.PegasusTokenizer.from_pretrained(str(PEGASUS_DIR), local_files_only=True)
        return tokenizer, str(PEGASUS_DIR), "pegasus-large"
    raise FileNotFoundError(
        f"Missing tokenizer dirs: tried {BART_DIR} and {PEGASUS_DIR}"
    )


def rebuild_sent_obj_from_original_nr(mat_file, sent_idx):
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
            return None

        for widx in range(len(word_level_data)):
            data_dict = word_level_data[widx]
            word_obj = {"content": data_dict["content"], "nFixations": data_dict["nFix"]}

            if "GD_EEG" in data_dict:
                gd = data_dict["GD_EEG"]
                ffd = data_dict["FFD_EEG"]
                trt = data_dict["TRT_EEG"]

                assert len(gd) == len(ffd) == len(trt) == 8

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
            return None

        return sent_obj


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    rng = np.random.default_rng(SEED)

    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Missing NPZ: {NPZ_PATH}")

    tokenizer, tokenizer_path, tokenizer_name = resolve_tokenizer()

    data = np.load(NPZ_PATH, allow_pickle=True)

    subject_ids = data["subject_id"]
    sentence_indices = data["sentence_index"]
    source_files = data["source_file"]
    target_strings = data["target_string"]
    input_embeddings = data["input_embeddings"]
    normalized_input_embeddings = data["normalized_input_embeddings"]
    input_attn_mask = data["input_attn_mask"]
    input_attn_mask_invert = data["input_attn_mask_invert"]
    seq_len = data["seq_len"]

    n = len(subject_ids)
    sample_rows = rng.choice(n, size=min(N_CHECK, n), replace=False)

    report = []
    all_ok = True

    for row_idx in sample_rows:
        row_idx = int(row_idx)

        subject_id = str(subject_ids[row_idx])
        sent_idx = int(sentence_indices[row_idx])
        source_file = str(source_files[row_idx])
        mat_file = MAT_DIR / source_file

        sent_obj = rebuild_sent_obj_from_original_nr(mat_file, sent_idx)
        if sent_obj is None:
            report.append({
                "row_idx": row_idx,
                "subject_id": subject_id,
                "sentence_index": sent_idx,
                "source_file": source_file,
                "status": "SKIP",
                "reason": "Original reconstruction returned None for this sentence"
            })
            continue

        ref = v2.get_input_sample(
            sent_obj,
            tokenizer=tokenizer,
            eeg_type="GD",
            bands=['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2'],
            max_len=58,
            dim=105,
            add_CLS_token=False
        )

        if ref is None:
            report.append({
                "row_idx": row_idx,
                "subject_id": subject_id,
                "sentence_index": sent_idx,
                "source_file": source_file,
                "status": "SKIP",
                "reason": "Original v2 get_input_sample returned None"
            })
            continue

        ref_input = to_numpy(ref["input_embeddings"]).astype(np.float32)
        ref_norm = to_numpy(ref["normalized_input_embeddings"]).astype(np.float32)
        ref_mask = to_numpy(ref["input_attn_mask"]).astype(np.int64)
        ref_mask_inv = to_numpy(ref["input_attn_mask_invert"]).astype(np.int64)
        ref_target = str(ref["target_string"])
        ref_true_seq_len = int(ref_mask.sum())

        npz_input = input_embeddings[row_idx].astype(np.float32)
        npz_norm = normalized_input_embeddings[row_idx].astype(np.float32)
        npz_mask = input_attn_mask[row_idx].astype(np.int64)
        npz_mask_inv = input_attn_mask_invert[row_idx].astype(np.int64)
        npz_target = str(target_strings[row_idx])
        npz_seq = int(seq_len[row_idx])

        target_ok = (ref_target == npz_target)
        input_ok = np.allclose(ref_input, npz_input, atol=ATOL, rtol=RTOL)
        norm_ok = np.allclose(ref_norm, npz_norm, atol=ATOL, rtol=RTOL)
        mask_ok = np.array_equal(ref_mask, npz_mask)
        mask_inv_ok = np.array_equal(ref_mask_inv, npz_mask_inv)
        seq_ok = (npz_seq == ref_true_seq_len)

        row_ok = target_ok and input_ok and norm_ok and mask_ok and mask_inv_ok and seq_ok
        all_ok = all_ok and row_ok

        report.append({
            "row_idx": row_idx,
            "subject_id": subject_id,
            "sentence_index": sent_idx,
            "source_file": source_file,
            "status": "PASS" if row_ok else "FAIL",
            "target_ok": target_ok,
            "input_ok": input_ok,
            "norm_ok": norm_ok,
            "mask_ok": mask_ok,
            "mask_inv_ok": mask_inv_ok,
            "seq_ok": seq_ok,
            "npz_seq_len": npz_seq,
            "ref_true_seq_len": ref_true_seq_len,
            "max_abs_input_diff": float(np.max(np.abs(ref_input - npz_input))),
            "max_abs_norm_diff": float(np.max(np.abs(ref_norm - npz_norm))),
        })

    out = {
        "all_ok": all_ok,
        "n_checked": len(report),
        "tokenizer_name": tokenizer_name,
        "tokenizer_path": tokenizer_path,
        "npz_path": str(NPZ_PATH),
        "report": report,
    }

    out_path = ROOT / "transfer_study" / "nr_v2_validation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({
        "all_ok": all_ok,
        "n_checked": len(report),
        "tokenizer_name": tokenizer_name,
        "tokenizer_path": tokenizer_path,
        "report_path": str(out_path)
    }, indent=2))

    for item in report:
        print(item)


if __name__ == "__main__":
    main()
