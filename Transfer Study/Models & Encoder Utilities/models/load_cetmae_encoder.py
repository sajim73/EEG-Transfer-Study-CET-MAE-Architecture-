#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            out[k[len('module.'):]] = v
        else:
            out[k] = v
    return out


def load_checkpoint_state(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(ckpt, dict):
        for key in ['state_dict', 'model_state_dict', 'model', 'net']:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise ValueError('Checkpoint does not contain a state_dict-like object')
    return strip_module_prefix(ckpt)


def infer_repo_root(start_file: str = __file__) -> Path:
    return Path(start_file).resolve().parent.parent


def add_repo_root_to_syspath():
    import sys
    root = str(infer_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


class CETMAEEncoder(nn.Module):
    def __init__(self, checkpoint_path: str, device: torch.device, pretrain_path: str = './models/huggingface/bart-large', freeze: bool = True):
        super().__init__()
        add_repo_root_to_syspath()
        from model_mae_bart import CETMAE_project_late_bart

        self.backbone = CETMAE_project_late_bart(pretrain_path=pretrain_path, device=device)
        state = load_checkpoint_state(checkpoint_path)
        self.missing_keys, self.unexpected_keys = self.backbone.load_state_dict(state, strict=False)
        self.input_dim = int(self.backbone.fc_eeg.in_features)
        self.output_dim = int(self.backbone.fc_eeg.out_features)
        self.device_ref = device
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def encode_sequence(self, eeg: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if eeg.size(-1) == self.input_dim:
            x = self.backbone.pos_embed_e(eeg)
            x = self.backbone.e_branch(x, src_key_padding_mask=(mask == 0))
            x = self.backbone.act(self.backbone.fc_eeg(x))
            return x
        if eeg.size(-1) == self.output_dim:
            return eeg
        raise ValueError(f'Unexpected EEG feature dim {eeg.size(-1)}; expected {self.input_dim} or {self.output_dim}')

    def pool_mean(self, sequence_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (sequence_embeddings * mask.unsqueeze(-1)).sum(dim=1) / denom

    def pool_last_valid(self, sequence_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        last_idx = (mask.sum(dim=1).long() - 1).clamp(min=0, max=sequence_embeddings.size(1) - 1)
        return sequence_embeddings[torch.arange(sequence_embeddings.size(0), device=sequence_embeddings.device), last_idx]

    def forward(self, eeg: torch.Tensor, mask: torch.Tensor, pooling: str = 'mean', return_sequence: bool = False):
        seq = self.encode_sequence(eeg, mask)
        if pooling == 'mean':
            pooled = self.pool_mean(seq, mask)
        elif pooling == 'last':
            pooled = self.pool_last_valid(seq, mask)
        else:
            raise ValueError("pooling must be one of: 'mean', 'last'")
        if return_sequence:
            return pooled, seq
        return pooled


def build_cetmae_encoder(checkpoint_path: str, device: torch.device = None, pretrain_path: str = './models/huggingface/bart-large', freeze: bool = True) -> CETMAEEncoder:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CETMAEEncoder(checkpoint_path=checkpoint_path, device=device, pretrain_path=pretrain_path, freeze=freeze)
    model.to(device)
    return model


def describe_checkpoint(model: CETMAEEncoder) -> Dict[str, object]:
    return {
        'input_dim': model.input_dim,
        'output_dim': model.output_dim,
        'n_missing_keys': len(model.missing_keys),
        'n_unexpected_keys': len(model.unexpected_keys),
        'missing_keys_preview': list(model.missing_keys[:20]) if hasattr(model.missing_keys, '__iter__') else [],
        'unexpected_keys_preview': list(model.unexpected_keys[:20]) if hasattr(model.unexpected_keys, '__iter__') else [],
    }


if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--pretrain-path', default='./models/huggingface/bart-large')
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_cetmae_encoder(args.checkpoint, device=device, pretrain_path=args.pretrain_path, freeze=args.freeze)
    print(json.dumps(describe_checkpoint(model), indent=2))
