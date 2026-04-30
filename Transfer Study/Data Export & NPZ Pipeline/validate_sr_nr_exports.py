#!/usr/bin/env python3
import os
import sys
import json
import h5py
import numpy as np
import scipy.io as sio
import importlib.util
from pathlib import Path

ROOT = Path("/home/sajim/CET-MAE").resolve()
os.chdir(ROOT)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SR_NPZ = ROOT / "transfer_study" / "exports" / "embeddings_SR_pretrained_unlabeled" / "embeddings.npz"
NR_NPZ = ROOT / "transfer_study" / "exports" / "embeddings_NR_pretrained_unlabeled" / "embeddings.npz"

SR_DIR = ROOT / "zuco_dataset" / "task1-SR" / "Matlab_files"
NR_DIR = ROOT / "zuco_dataset" / "task2-NR-2.0" / "Matlab_files"

BART_DIR = ROOT / "models" / "huggingface" / "bart-large"
PEGASUS_DIR = ROOT / "models" / "huggingface" / "pegasus-large"

MAX_LEN = 58
DIM = 105
BANDS = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']
EEG_TYPE = "GD"

N_CHECK_SR = 8
N_CHECK_NR = 8
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

v1 = load_module("v1_mod", ROOT / "data2pickle_v1.py")
v2 = load_module("v2_mod", ROOT / "data2pickle_v2.py")


def resolve_tokenizer():
    if BART_DIR.exists():
        tok = v1.BartTokenizer.from_pretrained(str(BART_DIR), local_files_only=True)
        return tok, "bart-large", str(BART_DIR)
    if PEGASUS_DIR.exists():
        tok = v2.PegasusTokenizer.from_pretrained(str(PEGASUS_DIR), local_files_only=True)
        return tok, "pegasus-large", str(PEGASUS_DIR)
    raise FileNotFoundError(
        f"Missing tokenizer dirs: tried {BART_DIR} and {PEGASUS_DIR}"
    )


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def infer_mat_filename(task_tag, subject_id, source_file_field=None):
    if source_file_field is not None and str(source_file_field).strip():
        return str(source_file_field)

    subject_id = str(subject_id).strip()
    if task_tag == "SR":
        return f"results{subject_id}_SR.mat"
    if task_tag == "NR":
        return f"results{subject_id}_NR.mat"
    raise ValueError(task_tag)


def get_attr_any(obj, *names, default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    if default is not None:
        return default
    raise AttributeError(f"{type(obj).__name__} missing all of: {names}")


def scalarize_first(x, default=0.0):
    if x is None:
        return default
    arr = np.asarray(x)
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])


def rebuild_sr_sent_obj(mat_file, sent_idx):
    mat = sio.loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)
    sentence_data = mat["sentenceData"]

    if isinstance(sentence_data, np.ndarray):
        sent = sentence_data[sent_idx]
    else:
        sent = sentence_data

    word_data = sent.word
    if isinstance(word_data, float):
        return None, "no_words"

    sent_obj = {
        "content": sent.content,
        "sentence_level_EEG": {
            "mean_t1": get_attr_any(sent, "mean_t1", "meant1"),
            "mean_t2": get_attr_any(sent, "mean_t2", "meant2"),
            "mean_a1": get_attr_any(sent, "mean_a1", "meana1"),
            "mean_a2": get_attr_any(sent, "mean_a2", "meana2"),
            "mean_b1": get_attr_any(sent, "mean_b1", "meanb1"),
            "mean_b2": get_attr_any(sent, "mean_b2", "meanb2"),
            "mean_g1": get_attr_any(sent, "mean_g1", "meang1"),
            "mean_g2": get_attr_any(sent, "mean_g2", "meang2"),
        },
        "word": [],
    }

    word_tokens_has_fixation = []
    word_tokens_with_mask = []
    word_tokens_all = []

    if isinstance(word_data, np.ndarray):
        word_iter = list(np.ravel(word_data))
    else:
        word_iter = [word_data]

    for word in word_iter:
        if word is None:
            continue

        content = getattr(word, "content", None)
        if content is None:
            continue

        nfix = scalarize_first(getattr(word, "nFixations", 0), default=0.0)

        word_obj = {
            "content": content,
            "nFixations": nfix,
        }
        word_tokens_all.append(content)

        if nfix > 0:
            word_obj["word_level_EEG"] = {
                "FFD": {
                    "FFD_t1": get_attr_any(word, "FFD_t1", "FFDt1"),
                    "FFD_t2": get_attr_any(word, "FFD_t2", "FFDt2"),
                    "FFD_a1": get_attr_any(word, "FFD_a1", "FFDa1"),
                    "FFD_a2": get_attr_any(word, "FFD_a2", "FFDa2"),
                    "FFD_b1": get_attr_any(word, "FFD_b1", "FFDb1"),
                    "FFD_b2": get_attr_any(word, "FFD_b2", "FFDb2"),
                    "FFD_g1": get_attr_any(word, "FFD_g1", "FFDg1"),
                    "FFD_g2": get_attr_any(word, "FFD_g2", "FFDg2"),
                },
                "TRT": {
                    "TRT_t1": get_attr_any(word, "TRT_t1", "TRTt1"),
                    "TRT_t2": get_attr_any(word, "TRT_t2", "TRTt2"),
                    "TRT_a1": get_attr_any(word, "TRT_a1", "TRTa1"),
                    "TRT_a2": get_attr_any(word, "TRT_a2", "TRTa2"),
                    "TRT_b1": get_attr_any(word, "TRT_b1", "TRTb1"),
                    "TRT_b2": get_attr_any(word, "TRT_b2", "TRTb2"),
                    "TRT_g1": get_attr_any(word, "TRT_g1", "TRTg1"),
                    "TRT_g2": get_attr_any(word, "TRT_g2", "TRTg2"),
                },
                "GD": {
                    "GD_t1": get_attr_any(word, "GD_t1", "GDt1"),
                    "GD_t2": get_attr_any(word, "GD_t2", "GDt2"),
                    "GD_a1": get_attr_any(word, "GD_a1", "GDa1"),
                    "GD_a2": get_attr_any(word, "GD_a2", "GDa2"),
                    "GD_b1": get_attr_any(word, "GD_b1", "GDb1"),
                    "GD_b2": get_attr_any(word, "GD_b2", "GDb2"),
                    "GD_g1": get_attr_any(word, "GD_g1", "GDg1"),
                    "GD_g2": get_attr_any(word, "GD_g2", "GDg2"),
                },
            }
            sent_obj["word"].append(word_obj)
            word_tokens_has_fixation.append(content)
            word_tokens_with_mask.append(content)
        else:
            word_tokens_with_mask.append("MASK")

    sent_obj["word_tokens_has_fixation"] = word_tokens_has_fixation
    sent_obj["word_tokens_with_mask"] = word_tokens_with_mask
    sent_obj["word_tokens_all"] = word_tokens_all

    if len(sent_obj["word"]) == 0:
        return None, "no_valid_words"

    return sent_obj, None


def rebuild_nr_sent_obj(mat_file, sent_idx):
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


def validate_dataset(name, npz_path, mat_dir, rebuild_fn, preprocess_mod, tokenizer, n_check, rng):
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing NPZ: {npz_path}")
    if not mat_dir.exists():
        raise FileNotFoundError(f"Missing MAT dir: {mat_dir}")

    data = np.load(npz_path, allow_pickle=True)

    subject_ids = data["subject_id"]
    sentence_indices = data["sentence_index"]
    target_strings = data["target_string"]
    input_embeddings = data["input_embeddings"]
    normalized_input_embeddings = data["normalized_input_embeddings"]
    input_attn_mask = data["input_attn_mask"]
    input_attn_mask_invert = data["input_attn_mask_invert"]
    seq_len = data["seq_len"]
    source_files = data["source_file"] if "source_file" in data.files else np.array([None] * len(subject_ids), dtype=object)

    n = len(subject_ids)
    sample_rows = rng.choice(n, size=min(n_check, n), replace=False)

    report = []
    all_ok = True

    for row_idx in sample_rows:
        row_idx = int(row_idx)
        subject_id = str(subject_ids[row_idx])
        sent_idx = int(sentence_indices[row_idx])
        source_file = infer_mat_filename(name, subject_id, source_files[row_idx])
        mat_file = mat_dir / source_file

        if not mat_file.exists():
            report.append({
                "row_idx": row_idx,
                "subject_id": subject_id,
                "sentence_index": sent_idx,
                "source_file": source_file,
                "status": "FAIL",
                "reason": f"MAT file not found: {mat_file}",
            })
            all_ok = False
            continue

        sent_obj, err = rebuild_fn(mat_file, sent_idx)
        if sent_obj is None:
            report.append({
                "row_idx": row_idx,
                "subject_id": subject_id,
                "sentence_index": sent_idx,
                "source_file": source_file,
                "status": "SKIP",
                "reason": err,
            })
            continue

        ref = preprocess_mod.get_input_sample(
            sent_obj,
            tokenizer=tokenizer,
            eeg_type=EEG_TYPE,
            bands=BANDS,
            max_len=MAX_LEN,
            dim=DIM,
            add_CLS_token=False,
        )

        if ref is None:
            report.append({
                "row_idx": row_idx,
                "subject_id": subject_id,
                "sentence_index": sent_idx,
                "source_file": source_file,
                "status": "SKIP",
                "reason": "get_input_sample returned None",
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

    return {
        "all_ok": all_ok,
        "n_checked": len(report),
        "report": report,
    }


def main():
    rng = np.random.default_rng(SEED)
    tokenizer, tokenizer_name, tokenizer_path = resolve_tokenizer()

    sr_result = validate_dataset(
        name="SR",
        npz_path=SR_NPZ,
        mat_dir=SR_DIR,
        rebuild_fn=rebuild_sr_sent_obj,
        preprocess_mod=v1,
        tokenizer=tokenizer,
        n_check=N_CHECK_SR,
        rng=rng,
    )

    nr_result = validate_dataset(
        name="NR",
        npz_path=NR_NPZ,
        mat_dir=NR_DIR,
        rebuild_fn=rebuild_nr_sent_obj,
        preprocess_mod=v2,
        tokenizer=tokenizer,
        n_check=N_CHECK_NR,
        rng=rng,
    )

    out = {
        "tokenizer_name": tokenizer_name,
        "tokenizer_path": tokenizer_path,
        "sr": sr_result,
        "nr": nr_result,
    }

    out_path = ROOT / "transfer_study" / "sr_nr_validation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({
        "tokenizer_name": tokenizer_name,
        "tokenizer_path": tokenizer_path,
        "sr_all_ok": sr_result["all_ok"],
        "sr_checked": sr_result["n_checked"],
        "nr_all_ok": nr_result["all_ok"],
        "nr_checked": nr_result["n_checked"],
        "report_path": str(out_path),
    }, indent=2))

    print("\n=== SR ===")
    for item in sr_result["report"]:
        print(item)

    print("\n=== NR ===")
    for item in nr_result["report"]:
        print(item)


if __name__ == "__main__":
    main()
