# CET-MAE: Cross-modal EEG-Text Masked Autoencoder

CET-MAE is a cross-modal masked autoencoder that jointly pre-trains an EEG encoder and a text decoder on parallel EEG-sentence pairs. Given masked EEG input, the model learns to reconstruct the corresponding natural language sentence. The pre-trained EEG encoder is then used as a general-purpose brain-signal encoder for downstream cognitive EEG tasks.

This repository contains:
- The full CET-MAE pretraining pipeline (data loading, model, training loop, evaluation)
- The transfer study (A1–C2) that evaluates encoder generalisation on sentiment and reading-task classification

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Prerequisites and Environment Setup](#2-prerequisites-and-environment-setup)
3. [Dataset: ZuCo](#3-dataset-zuco)
4. [Data Preprocessing — MATLAB to Pickle](#4-data-preprocessing--matlab-to-pickle)
5. [BART-large Download](#5-bart-large-download)
6. [Pre-training CET-MAE](#6-pre-training-cet-mae)
7. [Evaluating the Pretrained Model](#7-evaluating-the-pretrained-model)
8. [Checkpoints](#8-checkpoints)
9. [Transfer Study](#9-transfer-study)
10. [Configuration Files](#10-configuration-files)
11. [Key Hyperparameters and Design Decisions](#11-key-hyperparameters-and-design-decisions)
12. [Hardware Requirements](#12-hardware-requirements)
13. [Troubleshooting](#13-troubleshooting)
14. [Citation](#14-citation)

---

## 1. Repository Structure

```
CET-MAE/
│
├── README.md                                          ← This file
├── transfer_study/README_TRANSFER_STUDY.md            ← Transfer study guide
├── requirements.txt
├── environment.yml
│
│── config/
│   ├── config_cet_mae_transfer3.yaml                  ← Main pretraining config
│   ├── config_cet_mae_v2.yaml
│   ├── config_cet_mae_v3.yaml
│   ├── config_cet_mae_twostream_v1.yaml
│   ├── config_finetune.yaml
│   ├── config_pretrain_v2.yaml
│   └── config_pretrain_v3.yaml
│
├── dataset/
│   ├── ZuCo-1/
│   │   ├── raw/                                       ← Raw MATLAB (.mat) files go here
│   │   └── task1_SR_pickle/                           ← Pre-processed pickle files
│   └── ZuCo-2/
│       ├── raw/                                       ← Raw MATLAB (.mat) files go here
│       ├── NR_pickle/                                 ← Pre-processed pickle files
│       └── TSR_pickle/
│
├── data2pickle/
│   ├── construct_dataset_mat2pickle_v1_ZuCo.py        ← ZuCo 1.0 MATLAB→pickle converter
│   ├── construct_dataset_mat2pickle_v1_ZuCo2.py       ← ZuCo 2.0 MATLAB→pickle converter
│   └── utils.py                                       ← Signal normalisation utilities
│
├── contrastive_eeg_pretraining/
│   ├── dataset.py                                     ← PyTorch Dataset for CET-MAE pretraining
│   └── utils.py
│
├── dataset.py                                         ← Main dataset class used by training loop
├── utils.py                                           ← Shared utilities
├── optim_new.py                                       ← Custom optimiser (AdamW + cosine warmup)
│
├── Multi_Stream_TransformerEncoder.py                 ← EEG Transformer encoder (e_branch, fc_eeg, unify_branch)
├── model_mae_bart.py                                  ← Full CETMAEprojectlatebart model definition
│
├── pre_train_eval_cet_mae_later_project_7575.py       ← Main pretraining + evaluation script
│
├── checkpoints/
│   └── cet_mae/
│       ├── cet_mae_transfer3_a100_80gb_best.pt        ← Best pretraining checkpoint ← GITIGNORED
│       └── cet_mae_transfer3_a100_80gb_last.pt
│
├── logs/
│   ├── pretrain_transfer3/                            ← TensorBoard + CSV logs from pretraining
│   └── eval/
│
├── models/
│   └── huggingface/
│       └── bart-large/                                ← Local BART-large cache ← GITIGNORED
│
└── transfer_study/                                    ← All transfer experiments (see README_TRANSFER_STUDY.md)
```

---

## 2. Prerequisites and Environment Setup

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 16 GB VRAM (single GPU) | 80 GB A100 |
| RAM | 32 GB | 64 GB |
| Storage | 50 GB | 200 GB |
| CUDA | 11.7+ | 12.1 |

Training was originally run on a single NVIDIA A100 80 GB with `bf16` mixed precision. On a 16 GB GPU, use `fp16` and halve the batch size.

### Python environment

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate cetmae
```

Or install via pip into an existing environment:

```bash
pip install -r requirements.txt
```

The key packages are:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0.0 | Model, training |
| `transformers` | ≥ 4.35.0 | BART-large decoder |
| `numpy` | ≥ 1.24.0 | Data handling |
| `scipy` | ≥ 1.10.0 | Signal processing |
| `scikit-learn` | ≥ 1.3.0 | Metrics, LOSO splits |
| `matplotlib` | ≥ 3.7.0 | Training plots |
| `tensorboard` | ≥ 2.13.0 | Loss logging |
| `pandas` | ≥ 2.0.0 | Label CSV loading |
| `einops` | ≥ 0.7.0 | Attention tensor operations |
| `tqdm` | ≥ 4.65.0 | Progress bars |
| `h5py` | ≥ 3.9.0 | Reading ZuCo-2.0 .mat files (v7.3 format) |

### Verify GPU is available

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 3. Dataset: ZuCo

CET-MAE is trained and evaluated on the **ZuCo** (Zurich Cognitive Language processing) benchmark — a collection of simultaneous EEG and eye-tracking recordings of natural language reading.

### Versions used in this project

| Dataset | Version | Subjects | Sentences | Tasks | EEG System |
|---------|---------|----------|-----------|-------|-----------|
| ZuCo 1.0 | Task 1 — Sentiment Reading (SR) | 21 | ~400 SST sentences | Normal reading | EGI 512-channel, 256 Hz |
| ZuCo 2.0 | Normal Reading (NR) | 18 | ~349 sentences | Normal reading | EGI 512-channel, 256 Hz |
| ZuCo 2.0 | Task-Specific Reading (TSR) | 18 | ~349 sentences | Maze reading task | EGI 512-channel, 256 Hz |

### How to download ZuCo

#### ZuCo 1.0

ZuCo 1.0 is hosted on the Open Science Framework (OSF):

1. Go to: **https://osf.io/q3zws/**
2. Log in or create a free OSF account
3. Under "Files", navigate to `ZuCo_Task1_SR/EEG/`
4. Download all `.mat` files for all 21 subjects (files are named `ZAB_SR.mat`, `ZDM_SR.mat`, etc.)
5. Place them in: `dataset/ZuCo-1/raw/`

You can also use the OSF CLI:

```bash
pip install osfclient
osf -p q3zws clone dataset/ZuCo-1/raw/
```

#### ZuCo 2.0

ZuCo 2.0 is also on OSF:

1. Go to: **https://osf.io/2urht/**
2. Navigate to `EEG_recordings/`
3. Download the NR and TSR folders

Or via CLI:

```bash
osf -p 2urht clone dataset/ZuCo-2/raw/
```

The ZuCo 2.0 `.mat` files use HDF5 v7.3 format, which requires `h5py` to read (unlike ZuCo 1.0 which uses older MATLAB format readable with `scipy.io.loadmat`).

> **Storage note**: The raw `.mat` files are large (~2–4 GB per subject). Total raw data is approximately 80 GB for ZuCo 1.0 + 2.0 combined.

### Expected raw directory layout

```
dataset/
├── ZuCo-1/
│   └── raw/
│       ├── ZAB_SR.mat
│       ├── ZDM_SR.mat
│       ├── ZDN_SR.mat
│       ├── ZGW_SR.mat
│       ├── ZJM_SR.mat
│       ├── ZJN_SR.mat
│       ├── ZJS_SR.mat
│       ├── ZKB_SR.mat
│       ├── ZKH_SR.mat
│       ├── ZKW_SR.mat
│       ├── ZMG_SR.mat
│       ├── ZPH_SR.mat
│       ├── ZRV_SR.mat
│       ├── ZSM_SR.mat
│       └── ... (21 subjects total)
│
└── ZuCo-2/
    └── raw/
        ├── NR/
        │   ├── YAC_NR.mat
        │   ├── YAG_NR.mat
        │   └── ... (18 subjects)
        └── TSR/
            ├── YAC_TSR.mat
            ├── YAG_TSR.mat
            └── ... (18 subjects)
```

---

## 4. Data Preprocessing — MATLAB to Pickle

The raw `.mat` files must be converted to Python pickle files before any model training or NPZ export can happen. This step:
- Loads raw EEG per subject and normalises it (z-score per electrode, per subject)
- Aligns EEG segments with the corresponding sentence text and word-level timestamps
- Saves each subject as a pickle file containing a list of trial dictionaries

### Preprocess ZuCo 1.0 SR

```bash
python data2pickle/construct_dataset_mat2pickle_v1_ZuCo.py \
  --input-dir  dataset/ZuCo-1/raw \
  --output-dir dataset/ZuCo-1/task1_SR_pickle \
  --task SR
```

Expected output (one file per subject):

```
dataset/ZuCo-1/task1_SR_pickle/
  ZAB_SR.pickle
  ZDM_SR.pickle
  ...
  ZPH_SR.pickle
```

### Preprocess ZuCo 2.0 NR

```bash
python data2pickle/construct_dataset_mat2pickle_v1_ZuCo2.py \
  --input-dir  dataset/ZuCo-2/raw/NR \
  --output-dir dataset/ZuCo-2/NR_pickle \
  --task NR
```

### Preprocess ZuCo 2.0 TSR

```bash
python data2pickle/construct_dataset_mat2pickle_v1_ZuCo2.py \
  --input-dir  dataset/ZuCo-2/raw/TSR \
  --output-dir dataset/ZuCo-2/TSR_pickle \
  --task TSR
```

### What each pickle file contains

Each pickle is a Python `list` of `dict`, one dict per trial (sentence). Each trial dict has:

| Key | Type | Description |
|-----|------|-------------|
| `content` | `str` | The full sentence text |
| `word_tokens_embeddings` | `List[np.ndarray]` | Per-word EEG segments, each shaped `(105, n_samples)` after the 105-channel frequency-band decomposition |
| `word_tokens_posList` | `List[int]` | Word token positions in the sentence |
| `sentiment_label` | `str` | `"Positive"`, `"Negative"`, or `"Neutral"` (ZuCo 1.0 only) |
| `relation_label` | `str` | Semantic relation type (ZuCo 2.0 only) |
| `subject` | `str` | Subject identifier string |

The EEG features are 105-dimensional per word token, computed as:
- 5 frequency bands (theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz, low-gamma 30–70 Hz, high-gamma 70–120 Hz)
- × 21 Electrodes (selected from the full 512-channel cap based on coverage and signal quality)
= 105 features per word token.

> **Note**: Each sentence is represented as a sequence of word-level EEG segments. The model pads all sequences to a maximum token length and builds an attention mask. The maximum sequence length used in this project is 56 tokens (words per sentence).

---

## 5. BART-large Download

The CET-MAE decoder is BART-large (`facebook/bart-large`). Download it once and store it locally so the training loop does not need internet access:

```bash
python -c "
from transformers import BartForConditionalGeneration, BartTokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model.save_pretrained('models/huggingface/bart-large')
tokenizer.save_pretrained('models/huggingface/bart-large')
print('Done.')
"
```

Or with the `huggingface_hub` CLI:

```bash
huggingface-cli download facebook/bart-large --local-dir models/huggingface/bart-large
```

BART-large is approximately 1.6 GB.

> **Why local?** The pretraining script loads BART from `models/huggingface/bart-large/` using `from_pretrained('models/huggingface/bart-large')`. This ensures reproducible model initialisation even after upstream model updates, and works on GPU clusters without internet access.

---

## 6. Pre-training CET-MAE

### What CET-MAE learns

CET-MAE is trained on pairs `(EEG_sentence, text_sentence)`. The training objective is:

- **Masked EEG reconstruction → text**: Randomly mask a proportion of EEG word tokens; the model learns to decode the original sentence from the masked EEG input through a cross-attention decoder (BART-large).
- The EEG encoder learns to produce representations that are semantically aligned with text, without any explicit alignment loss — alignment emerges because the decoder needs text-like representations to reconstruct the sentence.

### Architecture overview

```
EEG input (N, L, 105)
    │
    ▼
pos_embed_e (positional embedding for EEG tokens)
    │
    ▼
e_branch (Transformer encoder — 6 layers, 512 hidden, 8 heads)
    │
    ▼
fc_eeg (projection: 512 → 1024)
    │
    ▼
unify_branch (Transformer encoder — 6 layers, 1024 hidden, 8 heads)
    │
    ▼
BART-large decoder (cross-attention on EEG features → generate text tokens)
    │
    ▼
Predicted sentence tokens
```

The components `pos_embed_e + e_branch + fc_eeg + unify_branch` together form the **EEG encoder** used in all transfer study experiments.

### Run pretraining

The main pretraining script is `pre_train_eval_cet_mae_later_project_7575.py`. Edit `config/config_cet_mae_transfer3.yaml` to set your data paths and hyperparameters, then run:

```bash
python pre_train_eval_cet_mae_later_project_7575.py \
  --config config/config_cet_mae_transfer3.yaml \
  --output-dir checkpoints/cet_mae \
  --log-dir logs/pretrain_transfer3
```

Key flags (all also settable in the config YAML):

| Flag | Default | Description |
|------|---------|-------------|
| `--batch-size` | 32 | Sentences per batch |
| `--epochs` | 200 | Total pretraining epochs |
| `--lr` | 5e-5 | Peak learning rate (after warmup) |
| `--warmup-steps` | 2000 | Linear warmup steps |
| `--mask-ratio` | 0.75 | Proportion of EEG tokens masked |
| `--eeg-dropout` | 0.1 | Dropout on EEG encoder |
| `--checkpoint-every` | 5 | Save checkpoint every N epochs |
| `--fp16` | False | Use fp16 mixed precision |
| `--bf16` | True | Use bf16 mixed precision (preferred on A100) |

### Resume training from a checkpoint

```bash
python pre_train_eval_cet_mae_later_project_7575.py \
  --config config/config_cet_mae_transfer3.yaml \
  --output-dir checkpoints/cet_mae \
  --log-dir logs/pretrain_transfer3 \
  --resume checkpoints/cet_mae/cet_mae_transfer3_a100_80gb_last.pt
```

### Monitor training

```bash
tensorboard --logdir logs/pretrain_transfer3
```

Open `http://localhost:6006` to view loss curves, BLEU score on validation set, and learning rate schedule.

---

## 7. Evaluating the Pretrained Model

Run the held-out evaluation of the pretraining reconstruction task:

```bash
python pre_train_eval_cet_mae_later_project_7575.py \
  --config config/config_cet_mae_transfer3.yaml \
  --checkpoint checkpoints/cet_mae/cet_mae_transfer3_a100_80gb_best.pt \
  --eval-only \
  --output-dir logs/eval
```

This generates:
- `logs/eval/eval_bleu.json` — BLEU-1/2/3/4 on the reconstruction task
- `logs/eval/sample_outputs.txt` — 50 randomly sampled `(true sentence, predicted sentence)` pairs
- `logs/eval/loss_curve.png`

---

## 8. Checkpoints

The best pretrained checkpoint is too large for Git (≈ 2 GB). It is stored externally and must be downloaded separately.

### Download the pretrained checkpoint

```bash
# Option A: from the project's shared storage link (ask the project maintainer)
wget -O checkpoints/cet_mae/cet_mae_transfer3_a100_80gb_best.pt \
  <SHARED_CHECKPOINT_URL>

# Option B: if you have a Google Drive link, use gdown
pip install gdown
gdown <GDRIVE_FILE_ID> -O checkpoints/cet_mae/cet_mae_transfer3_a100_80gb_best.pt
```

### Verify the checkpoint

```bash
python -c "
import torch
ckpt = torch.load('checkpoints/cet_mae/cet_mae_transfer3_a100_80gb_best.pt', map_location='cpu')
# Depending on how it was saved, the top-level is either a state_dict or {'model': ..., 'epoch': ...}
if isinstance(ckpt, dict) and 'model' in ckpt:
    keys = list(ckpt['model'].keys())
    print(f'Epoch: {ckpt.get(\"epoch\", \"unknown\")}')
    print(f'Val loss: {ckpt.get(\"best_val_loss\", \"unknown\")}')
else:
    keys = list(ckpt.keys())
print(f'Number of parameter tensors: {len(keys)}')
print(f'First 5 keys: {keys[:5]}')
"
```

---

## 9. Transfer Study

Once the pretrained checkpoint is available and the pickle files are built, follow `transfer_study/README_TRANSFER_STUDY.md` for the complete step-by-step guide.

**Quick summary of the execution order**:

```
Step 1: Build NPZ exports from pickle files
         extract_encoder_embeddings_cet_mae.py  (SR, NR)
         build_tsr_npz_from_pickles.py           (TSR)

Step 2: Validate NPZ exports
         validate_sr_nr_exports.py
         validate_nr_export_against_v2.py

Step 3: Run experiments in order:
         A1 → A2 → B1 (pretrained + random) → B2 (pretrained + random) → C1 → C2

Step 4: Build result tables and significance tests
         analysis/build_tables.py
         analysis/test_stats.py
```

---

## 10. Configuration Files

All hyperparameters are controlled by YAML config files in `config/`. The pretraining scripts accept a `--config` argument and merge the YAML with any command-line overrides (command-line takes priority).

### Key config sections

```yaml
# config/config_cet_mae_transfer3.yaml (abbreviated)

data:
  zuco1_sr_pickle:  dataset/ZuCo-1/task1_SR_pickle
  zuco2_nr_pickle:  dataset/ZuCo-2/NR_pickle
  zuco2_tsr_pickle: dataset/ZuCo-2/TSR_pickle
  max_seq_len:      56        # max EEG tokens per sentence
  eeg_feature_dim:  840       # 105 channel-features × 8 time steps (per word token)

model:
  bart_path:         models/huggingface/bart-large
  encoder_hidden:    512       # e_branch hidden dim
  encoder_layers:    6         # e_branch transformer layers
  encoder_heads:     8
  unify_hidden:      1024      # unify_branch hidden dim (= BART hidden dim)
  unify_layers:      6
  unify_heads:       8
  eeg_dropout:       0.1
  mask_ratio:        0.75

training:
  batch_size:        32
  epochs:            200
  lr:                5.0e-5
  warmup_steps:      2000
  weight_decay:      0.01
  grad_clip:         1.0
  bf16:              true
  checkpoint_every:  5
  save_best_by:      val_loss
```

---

## 11. Key Hyperparameters and Design Decisions

### Why 105-dimensional EEG features per word?

EEG preprocessing extracts power in 5 frequency bands (theta, alpha, beta, low-gamma, high-gamma) across 21 electrodes = 105 features per word token. This is the ZuCo standard feature representation used in Hollenstein et al. (2019, 2021, 2023).

### Why is `eeg_feature_dim = 840` in the config?

Each 105-dimensional word token is further time-windowed over 8 time steps, producing an 840-dimensional input vector to the encoder. The Transformer encoder then processes a sequence of these 840-dim vectors (one per word).

### Why mask ratio 0.75?

The 75% masking ratio follows the original MAE paper (He et al., 2022). In pilot runs, lower masking ratios (50%) led to faster initial convergence but worse final representations; higher ratios (90%) required more epochs to converge. 75% yielded the best trade-off for this dataset size.

### Why BART-large (not BART-base)?

BART-large provides a stronger language prior, which forces the EEG encoder to produce higher-quality representations. In ablation runs, BART-base achieved lower reconstruction BLEU but also produced weaker transfer representations in the frozen probe experiments.

### Why bf16?

bf16 (brain float 16) has the same exponent range as fp32, avoiding the dynamic range issues that cause fp16 to overflow or underflow on large language model decoders. On A100 GPUs, bf16 tensor cores give approximately 2× throughput over fp32 with no measurable loss quality degradation on this task.

---

## 12. Hardware Requirements

### Pretraining

| GPU | Batch size | Epochs | Approx. time |
|-----|-----------|--------|-------------|
| A100 80 GB | 32 | 200 | ~18–24 hours |
| A100 40 GB | 16 | 200 | ~30–40 hours |
| V100 32 GB | 8 | 200 | ~60–80 hours |
| RTX 3090 24 GB | 8 | 200 | ~80–100 hours |

### Transfer study

| Experiment | GPU | Batch size | Approx. time |
|-----------|-----|-----------|-------------|
| A1, A2 (frozen probe) | Any 8 GB+ | 64 | 5–20 min |
| B1, B2 (fine-tuning) | 16 GB+ | 32 | 30–60 min each |
| C1, C2 | Any 8 GB+ | 64 | 5–20 min |

---

## 13. Troubleshooting

### `ModuleNotFoundError: No module named 'einops'`

```bash
pip install einops
```

### `RuntimeError: Expected all tensors to be on the same device`

Ensure that the EEG input tensor, attention mask, and BART decoder are all moved to the same device. Check that `model.to(device)` is called before any forward pass. The training script handles this, but if you are writing custom evaluation code, explicitly call `.to(device)` on all input tensors.

### ZuCo 2.0 `.mat` files fail to load with `scipy.io.loadmat`

ZuCo 2.0 saves `.mat` files in HDF5 v7.3 format, which `scipy.io.loadmat` cannot read. Use `h5py`:

```python
import h5py
with h5py.File('YAC_NR.mat', 'r') as f:
    print(list(f.keys()))
```

Make sure `h5py` is installed: `pip install h5py`.

### OOM during pretraining on smaller GPU

- Reduce `--batch-size` to 8 or 4
- Enable gradient checkpointing by adding `--gradient-checkpointing` (supported in the training script)
- Ensure no other processes are using the GPU: `nvidia-smi`

### BLEU score stuck near zero during pretraining

This is normal for the first 10–20 epochs. The decoder needs time to learn to use the EEG cross-attention. Check:
- That the mask ratio is not 100% (which would give the decoder no signal)
- That the learning rate warmup is running (log `lr` in TensorBoard)
- That BART parameters are not frozen (they should be updated)

### Pickle files load but produce empty EEG arrays

Some ZuCo subjects have missing trials due to recording issues. The preprocessing script logs a warning for each missing trial and skips it. This is expected. Verify that the majority of trials have valid EEG by checking the pickle summary:

```python
import pickle
with open('dataset/ZuCo-1/task1_SR_pickle/ZAB_SR.pickle', 'rb') as f:
    trials = pickle.load(f)
print(f'Trials: {len(trials)}')
print(f'Non-empty: {sum(1 for t in trials if t["word_tokens_embeddings"] is not None)}')
```

---

## 14. Citation

If you use CET-MAE or the ZuCo datasets in your research, please cite:

**CET-MAE** (this work):
```bibtex
@article{cetmae2024,
  title   = {CET-MAE: Cross-modal EEG-Text Masked Autoencoder for Brain-Language Alignment},
  author  = {[Author names]},
  year    = {2024},
  journal = {[Journal/Conference]}
}
```

**ZuCo 1.0**:
```bibtex
@article{hollenstein2018zuco,
  title     = {ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading},
  author    = {Hollenstein, Nora and Rothe, Antonio and Troendle, Marius and Zhang, Ce and Langer, Nicolas},
  journal   = {Scientific Data},
  volume    = {5},
  pages     = {180001},
  year      = {2018},
  publisher = {Nature Publishing Group}
}
```

**ZuCo 2.0**:
```bibtex
@inproceedings{hollenstein2020zuco,
  title     = {ZuCo 2.0: A dataset of physiological recordings during natural reading and annotation},
  author    = {Hollenstein, Nora and Troendle, Marius and Zhang, Ce and Langer, Nicolas},
  booktitle = {Proceedings of LREC 2020},
  year      = {2020}
}
```

**Masked Autoencoders (MAE)**:
```bibtex
@inproceedings{he2022masked,
  title     = {Masked Autoencoders Are Scalable Vision Learners},
  author    = {He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Dollár, Piotr and Girshick, Ross},
  booktitle = {CVPR},
  year      = {2022}
}
```

**BART**:
```bibtex
@inproceedings{lewis2020bart,
  title     = {BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension},
  author    = {Lewis, Mike and Liu, Yinhan and Goyal, Naman and Ghahraman, Marjan and Mohamed, Abdelrahman and Levy, Omer and Stoyanov, Veselin and Zettlemoyer, Luke},
  booktitle = {ACL},
  year      = {2020}
}
```
