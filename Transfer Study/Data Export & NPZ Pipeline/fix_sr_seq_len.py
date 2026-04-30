#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

ROOT = Path("/home/sajim/CET-MAE").resolve()
SR_DIR = ROOT / "transfer_study" / "exports" / "embeddings_SR_pretrained_unlabeled"
SR_NPZ = SR_DIR / "embeddings.npz"

data = np.load(SR_NPZ, allow_pickle=True)

fixed_seq_len = data["input_attn_mask"].sum(axis=1).astype(np.int64)

save_dict = {}
for k in data.files:
    if k == "seq_len":
        save_dict[k] = fixed_seq_len
    else:
        save_dict[k] = data[k]

np.savez_compressed(SR_NPZ, **save_dict)

summary = {
    "patched_file": str(SR_NPZ),
    "n_samples": int(fixed_seq_len.shape[0]),
    "min_seq_len": int(fixed_seq_len.min()),
    "max_seq_len": int(fixed_seq_len.max()),
    "note": "SR seq_len replaced with input_attn_mask.sum(axis=1)"
}

print(json.dumps(summary, indent=2))
