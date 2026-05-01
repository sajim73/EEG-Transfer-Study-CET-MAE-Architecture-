import os
import pickle
from typing import List

import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import BatchEncoding


class EEG_dataset_add_sentence_mae(Dataset):
    """
    Minimal CET-MAE dataset loader reconstructed from:
    - current pickle schema
    - CET-MAE pretraining script
    - EEG-To-Text ancestor repo structure

    Expected pickle keys:
      normalized_input_embeddings : [58, 840]
      input_embeddings            : [58, 840]
      input_attn_mask             : [58]
      input_attn_mask_invert      : [58]
      target_ids                  : [58]
      target_mask                 : [58]
      target_string               : str
      target_tokenized            : BatchEncoding with input_ids/attention_mask
    """

    def __init__(self, path: str):
        self.path = path
        if not os.path.isdir(self.path):
            raise FileNotFoundError(f"Dataset path not found: {self.path}")

        self.files: List[str] = sorted(
            [
                os.path.join(self.path, f)
                for f in os.listdir(self.path)
                if f.endswith(".pickle") or f.endswith(".pkl")
            ]
        )

        if len(self.files) == 0:
            raise RuntimeError(f"No pickle files found in: {self.path}")

    def __len__(self):
        return len(self.files)

    def _to_float_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().clone().float()
        return torch.tensor(x, dtype=torch.float32)

    def _to_long_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().clone().long()
        return torch.tensor(x, dtype=torch.long)

    def _fix_batch_encoding(self, target_tokenized):
        if isinstance(target_tokenized, BatchEncoding):
            out = {}
            for k, v in target_tokenized.items():
                if isinstance(v, torch.Tensor):
                    out[k] = v.detach().clone().long()
                else:
                    out[k] = torch.tensor(v, dtype=torch.long)
            return BatchEncoding(out)

        if isinstance(target_tokenized, dict):
            out = {}
            for k, v in target_tokenized.items():
                if isinstance(v, torch.Tensor):
                    out[k] = v.detach().clone().long()
                else:
                    out[k] = torch.tensor(v, dtype=torch.long)
            return BatchEncoding(out)

        raise TypeError(f"Unsupported target_tokenized type: {type(target_tokenized)}")

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with open(file_path, "rb") as f:
            sample = pickle.load(f)

        if not isinstance(sample, dict):
            raise TypeError(f"Sample is not dict: {file_path}")

        # EEG: use normalized embeddings for model input, matching pickle naming.
        if "normalized_input_embeddings" in sample:
            input_embeddings = self._to_float_tensor(sample["normalized_input_embeddings"])
        elif "input_embeddings" in sample:
            input_embeddings = self._to_float_tensor(sample["input_embeddings"])
        else:
            raise KeyError(f"Missing normalized/input embeddings in {file_path}")

        # Non-normalized counterpart for logging / compatibility with training script.
        if "input_embeddings" in sample:
            non_normalized_input_embeddings = self._to_float_tensor(sample["input_embeddings"])
        elif "non_normalized_embeddings_for_vis" in sample:
            arr = sample["non_normalized_embeddings_for_vis"]
            arr = torch.tensor(arr, dtype=torch.float32)
            pad_len = input_embeddings.size(0) - arr.size(0)
            if pad_len > 0:
                arr = torch.cat([arr, torch.zeros(pad_len, arr.size(1), dtype=torch.float32)], dim=0)
            non_normalized_input_embeddings = arr
        else:
            non_normalized_input_embeddings = input_embeddings.detach().clone()

        input_attn_mask = self._to_float_tensor(sample["input_attn_mask"])
        input_attn_mask_invert = self._to_float_tensor(sample["input_attn_mask_invert"])

        target_ids = self._to_long_tensor(sample["target_ids"])
        target_mask = self._to_long_tensor(sample["target_mask"])
        target_tokenized = self._fix_batch_encoding(sample["target_tokenized"])
        text = sample["target_string"]

        # Shape checks based on CET-MAE model expectations.
        if input_embeddings.ndim != 2 or input_embeddings.shape[1] != 840:
            raise ValueError(
                f"Unexpected EEG shape in {file_path}: {tuple(input_embeddings.shape)}; expected [L, 840]"
            )

        if input_attn_mask.ndim != 1 or input_attn_mask_invert.ndim != 1:
            raise ValueError(
                f"Unexpected mask shape in {file_path}: "
                f"{tuple(input_attn_mask.shape)}, {tuple(input_attn_mask_invert.shape)}"
            )

        return (
            input_embeddings,
            non_normalized_input_embeddings,
            input_attn_mask,
            input_attn_mask_invert,
            target_ids,
            target_mask,
            target_tokenized,
            text,
        )
