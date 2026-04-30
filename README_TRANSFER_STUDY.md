# Transfer Study: CET-MAE EEG Encoder Generalisation

This document is the complete guide for the **CET-MAE Transfer Study** — a systematic evaluation of whether the EEG encoder pre-trained for EEG-to-text decoding produces representations that generalise to downstream EEG cognitive tasks it was never trained on.

---

## Table of Contents

1. [Research Questions](#1-research-questions)
2. [Overview of Experiments](#2-overview-of-experiments)
3. [Prerequisites](#3-prerequisites)
4. [Data Requirements for the Transfer Study](#4-data-requirements-for-the-transfer-study)
5. [Step 1 — Build the NPZ Embedding Exports](#5-step-1--build-the-npz-embedding-exports)
6. [Step 2 — Verify the NPZ Exports](#6-step-2--verify-the-npz-exports)
7. [Step 3 — Prepare Labels](#7-step-3--prepare-labels)
8. [Step 4 — Run Experiment A1 (Frozen Sentiment Probe)](#8-step-4--run-experiment-a1-frozen-sentiment-probe)
9. [Step 5 — Run Experiment A2 (Frozen Reading-Task Probe)](#9-step-5--run-experiment-a2-frozen-reading-task-probe)
10. [Step 6 — Run Experiment B1 (Fine-Tuned Sentiment)](#10-step-6--run-experiment-b1-fine-tuned-sentiment)
11. [Step 7 — Run Experiment B2 (Fine-Tuned Reading Task)](#11-step-7--run-experiment-b2-fine-tuned-reading-task)
12. [Step 8 — Run Experiment C1 (Relations Probe)](#12-step-8--run-experiment-c1-relations-probe)
13. [Step 9 — Run Experiment C2 (Noise Control)](#13-step-9--run-experiment-c2-noise-control)
14. [Evaluation & Analysis Scripts](#14-evaluation--analysis-scripts)
15. [Understanding the Output Files](#15-understanding-the-output-files)
16. [Interpreting Results](#16-interpreting-results)
17. [Experiment Design Notes](#17-experiment-design-notes)
18. [Troubleshooting](#18-troubleshooting)
19. [Directory Reference](#19-directory-reference)

---

## 1. Research Questions

| ID | Question |
|----|----------|
| RQ1 | Does CET-MAE's pre-trained encoder produce representations useful for **sentiment classification** from reading EEG? |
| RQ2 | Does it help with **reading task classification** (Normal Reading vs. Task-Specific Reading)? |
| RQ3 | How does CET-MAE compare against **(a)** training from scratch, **(b)** a simple feature baseline, and **(c)** a randomly-initialised encoder with the same architecture? |
| RQ4 | Does **fine-tuning** the encoder improve over frozen-encoder probing, and by how much? |

The transfer study is structured so each experiment directly addresses one or more of these questions, with controlled baselines designed to isolate the contribution of pre-training.

---

## 2. Overview of Experiments

| Exp | Name | Task | Encoder Mode | Control / Baseline | Answers |
|-----|------|------|--------------|-------------------|---------|
| A1 | Frozen Sentiment Probe | Sentiment (3-class or binary) on ZuCo 1.0 SR | Frozen — only linear head trains | None (baseline is chance) | RQ1 |
| A2 | Frozen Reading-Task Probe | Normal Reading vs. Task-Specific Reading on ZuCo 2.0 | Frozen — only linear head trains | None | RQ2 |
| B1 | Fine-Tuned Sentiment | Same as A1 | Full encoder + head fine-tuned | Random-init run (no pretraining) | RQ3, RQ4 |
| B2 | Fine-Tuned Reading Task | Same as A2 | Full encoder + head fine-tuned | Random-init run (no pretraining) | RQ3, RQ4 |
| C1 | Relations Probe | Semantic relation type on ZuCo 2.0 NR | Frozen | None | RQ1 (extension) |
| C2 | Noise Control | Sentiment on phase-randomised EEG | Frozen | Phase-randomised NPZ | RQ3 |

### How the experiments connect

```
ZuCo 1.0 SR NPZ ──► A1 (frozen, sentiment)
                 └─► B1 (finetune, sentiment) ──► B1_pretrained vs B1_random   }
                                                                                 }── RQ3: value of pretraining
ZuCo 2.0 NR + TSR NPZ ──► A2 (frozen, NR vs TSR)                               }
                       └─► B2 (finetune, NR vs TSR) ──► B2_pretrained vs B2_random }

A1 (frozen) vs B1 (finetune, pretrained) ──► RQ4: value of fine-tuning
A2 (frozen) vs B2 (finetune, pretrained) ──► RQ4: value of fine-tuning
```

---

## 3. Prerequisites

Before running any transfer study experiment, the following must be completed:

1. **CET-MAE pretraining is done** and the best checkpoint exists at:
   `checkpoints/cet_mae/cet_mae_transfer3_a100_80gb_best.pt`

2. **ZuCo pickle files exist** for all three conditions:
   - `dataset/ZuCo-1/task1_SR_pickle/` — ZuCo 1.0 SR (21 subjects)
   - `dataset/ZuCo-2/NR_pickle/` — ZuCo 2.0 NR (18 subjects)
   - `dataset/ZuCo-2/TSR_pickle/` — ZuCo 2.0 TSR (18 subjects)

   If these are missing, go back to the main `README.md` sections 4–6 and run the preprocessing scripts first.

3. **`facebook/bart-large` is cached** at `models/huggingface/bart-large/` (needed because the encoder loading code initialises the full CETMAEprojectlatebart model before extracting the EEG-only submodules).

4. **Python environment is active** (`conda activate cetmae`).

---

## 4. Data Requirements for the Transfer Study

The transfer study does **not** directly load raw MATLAB or pickle files. Instead it works from pre-exported **NPZ files** — compact NumPy archives containing the encoder's token-level EEG representations for every sentence in each task condition.

Three NPZ files are needed:

| NPZ | Source Data | Used by Experiments |
|-----|------------|---------------------|
| `exports/embeddings_SR_pretrained_unlabeled/embeddings.npz` | ZuCo 1.0 SR (all 21 subjects) | A1, B1 |
| `exports/embeddings_NR_pretrained_unlabeled/embeddings.npz` | ZuCo 2.0 NR (all 18 subjects) | A2, B2, C1 |
| `exports/embeddings_TSR_pretrained_unlabeled/embeddings.npz` | ZuCo 2.0 TSR (all 18 subjects) | A2, B2 |

Each NPZ contains the following arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `eeg` or `normalized_input_embeddings` | `(N, L, D_in)` | Raw token-level EEG inputs — shape is (n_samples, max_seq_len, eeg_feature_dim). This is the raw EEG fed into the encoder, **not** the encoder's output. |
| `input_attn_mask` | `(N, L)` | Attention mask: 1 for real EEG tokens, 0 for padding |
| `subject_id` | `(N,)` | Subject identifier string per sample |
| `target_string` or `text` | `(N,)` | The original sentence string per sample |

> **Why token-level inputs and not pre-pooled features?** The fine-tuning experiments (B1, B2) need to run the encoder forward pass end-to-end, so they need raw token-level EEG. The probe experiments (A1, A2) auto-detect the dimensionality and will either run the encoder or just mean-pool, depending on whether the input matches `encoder.input_dim` or `encoder.output_dim`.

---

## 5. Step 1 — Build the NPZ Embedding Exports

All three NPZ files are built using `extract_encoder_embeddings_cet_mae.py`. Run the following three commands from the `transfer_study/` directory.

```bash
cd transfer_study
```

### Export ZuCo 1.0 SR

```bash
python extract_encoder_embeddings_cet_mae.py \
  --pickle-dir ../dataset/ZuCo-1/task1_SR_pickle \
  --checkpoint ../checkpoints/cet_mae/cet_mae_transfer3_a100_80gb_best.pt \
  --bart-path ../models/huggingface/bart-large \
  --output-dir exports/embeddings_SR_pretrained_unlabeled \
  --task SR \
  --seed 42
```

### Export ZuCo 2.0 NR

```bash
python extract_encoder_embeddings_cet_mae.py \
  --pickle-dir ../dataset/ZuCo-2/NR_pickle \
  --checkpoint ../checkpoints/cet_mae/cet_mae_transfer3_a100_80gb_best.pt \
  --bart-path ../models/huggingface/bart-large \
  --output-dir exports/embeddings_NR_pretrained_unlabeled \
  --task NR \
  --seed 42
```

### Export ZuCo 2.0 TSR

```bash
python build_tsr_npz_from_pickles.py \
  --pickle-dir ../dataset/ZuCo-2/TSR_pickle \
  --checkpoint ../checkpoints/cet_mae/cet_mae_transfer3_a100_80gb_best.pt \
  --bart-path ../models/huggingface/bart-large \
  --output-dir exports/embeddings_TSR_pretrained_unlabeled \
  --seed 42
```

> **Note**: TSR uses `build_tsr_npz_from_pickles.py` rather than the generic extractor because the TSR pickle files have a slightly different internal structure from the NR/SR files.

Each export writes:
```
exports/embeddings_<TASK>_pretrained_unlabeled/
  embeddings.npz       ← the NPZ used by all experiment scripts
  metadata.csv         ← per-sample: subject, sentence text, task, row index
  summary.json         ← export config: n_samples, shape, checkpoint path, date
```

If you need to wipe and rebuild all exports from scratch (e.g., after updating the checkpoint):

```bash
bash ../reset_and_rebuild_exports.sh
```

---

## 6. Step 2 — Verify the NPZ Exports

Run the validation scripts to confirm the exports are consistent and correctly shaped before running experiments.

### Validate SR and NR exports

```bash
python validate_sr_nr_exports.py \
  --npz-sr exports/embeddings_SR_pretrained_unlabeled/embeddings.npz \
  --npz-nr exports/embeddings_NR_pretrained_unlabeled/embeddings.npz
```

This checks:
- Both NPZs contain the required keys (`eeg`/`normalized_input_embeddings`, `input_attn_mask`, `subject_id`, `text`)
- EEG shapes are consistent (same `D_in` across both)
- No NaN or Inf values
- Subject IDs are parseable strings

Results are saved to `sr_nr_validation_report.json`.

### Validate NR export against ZuCo 2.0 v2 schema

```bash
python validate_nr_export_against_v2.py \
  --npz exports/embeddings_NR_pretrained_unlabeled/embeddings.npz \
  --metadata exports/embeddings_NR_pretrained_unlabeled/metadata.csv
```

Results saved to `nr_v2_validation_report.json`.

### Quick shape check (manual)

```bash
python -c "
import numpy as np

for tag, path in [
    ('SR',  'exports/embeddings_SR_pretrained_unlabeled/embeddings.npz'),
    ('NR',  'exports/embeddings_NR_pretrained_unlabeled/embeddings.npz'),
    ('TSR', 'exports/embeddings_TSR_pretrained_unlabeled/embeddings.npz'),
]:
    npz = np.load(path, allow_pickle=True)
    eeg_key = [k for k in npz.files if 'eeg' in k.lower() or 'embed' in k.lower()][0]
    eeg = npz[eeg_key]
    subj = npz[[k for k in npz.files if 'subject' in k.lower()][0]]
    print(f'{tag}: eeg={eeg.shape}, subjects={sorted(set(subj.astype(str).tolist()))}')
"
```

Expected output for a complete export:
```
SR:  eeg=(8400, 56, 840), subjects=['ZAB', 'ZDM', ..., 'ZPH']   # 21 subjects × ~400 sentences
NR:  eeg=(6282, 56, 840), subjects=['YAC', 'YAG', ..., 'YYS']   # 18 subjects × ~349 sentences
TSR: eeg=(6282, 56, 840), subjects=['YAC', 'YAG', ..., 'YYS']   # 18 subjects × ~349 sentences
```

> The exact number of sentences per subject may vary slightly because some subjects have missing trials in the original MATLAB files.

---

## 7. Step 3 — Prepare Labels

### Sentiment labels (A1, B1)

The sentiment labels for ZuCo 1.0 SR are pre-built and already in the repository:

```
transfer_study/labels/sentiment_normal_reading.csv
```

This CSV has columns: `sentence`, `sentiment_label`, `subject` (optional). The `sentiment_label` column contains string values `"Positive"`, `"Negative"`, or `"Neutral"` from the Stanford Sentiment Treebank (SST) annotations used in ZuCo 1.0.

To use binary labels instead (positive vs. negative, dropping neutral), pass `--binary` to the A1/B1 scripts.

If you need to regenerate or re-align this file from the raw pickle sentiment fields:

```bash
python -c "
import pickle, glob, csv

rows = []
for pkl in sorted(glob.glob('../dataset/ZuCo-1/task1_SR_pickle/*.pickle')):
    subj = pkl.split('/')[-1].replace('_SR.pickle','')
    with open(pkl, 'rb') as f:
        trials = pickle.load(f)
    for t in trials:
        rows.append({'sentence': t['content'], 'sentiment_label': t['sentiment_label'], 'subject': subj})

with open('labels/sentiment_normal_reading.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['sentence','sentiment_label','subject'], delimiter=';')
    writer.writeheader()
    writer.writerows(rows)
print(f'Wrote {len(rows)} rows.')
"
```

### Relations labels (C1)

The relations probe labels are in:

```
transfer_study/labels/relations_normal_reading.csv
transfer_study/labels/relations_normal_reading_utf8.csv
```

These contain sentence–relation-type pairs from ZuCo 2.0 NR.

---

## 8. Step 4 — Run Experiment A1 (Frozen Sentiment Probe)

**Purpose**: Test whether the frozen CET-MAE encoder produces linearly separable representations for sentiment classification. The encoder weights are completely locked; only a single linear layer trains on top of the pooled EEG features. This is the most direct test of RQ1.

**Evaluation**: Leave-one-subject-out (LOSO) cross-validation over all 21 ZuCo 1.0 SR subjects. In each fold, one subject is the test set and the remaining 20 are split 85/15 into train/val.

**Chance baseline**: 33.3% (3-class) or 50% (binary).

```bash
cd transfer_study

python 1_train_probe_sentiment_cet_mae.py \
  --embeddings-npz exports/embeddings_SR_pretrained_unlabeled/embeddings.npz \
  --labels-csv labels/sentiment_normal_reading.csv \
  --checkpoint ../checkpoints/cet_mae/cet_mae_transfer3_a100_80gb_best.pt \
  --bart-path ../models/huggingface/bart-large \
  --output-dir results/A1_frozen_probe_sentiment \
  --head-type linear \
  --freeze-encoder \
  --epochs 50 \
  --lr 1e-3 \
  --batch-size 64 \
  --val-ratio 0.15 \
  --seed 42
```

For binary sentiment (drop neutral, classify pos vs. neg):

```bash
python 1_train_probe_sentiment_cet_mae.py \
  --embeddings-npz exports/embeddings_SR_pretrained_unlabeled/embeddings.npz \
  --labels-csv labels/sentiment_normal_reading.csv \
  --checkpoint ../checkpoints/cet_mae/cet_mae_transfer3_a100_80gb_best.pt \
  --bart-path ../models/huggingface/bart-large \
  --output-dir results/A1_frozen_probe_sentiment_binary \
  --head-type linear \
  --freeze-encoder \
  --epochs 50 \
  --lr 1e-3 \
  --binary \
  --seed 42
```

Or use the shell launcher:

```bash
bash run_scripts/run_probe_sentiment_cet_mae.sh
```

**Expected runtime**: ~5–15 minutes on GPU (21 LOSO folds × 50 epochs each, fast because encoder is frozen and features are cached once per fold).

---

## 9. Step 5 — Run Experiment A2 (Frozen Reading-Task Probe)

**Purpose**: Test whether the frozen encoder distinguishes cognitive state between Normal Reading and Task-Specific Reading. This uses a within-ZuCo-2.0 design: both classes come from the same subjects recorded in the same session, eliminating dataset-level confounds.

**Task definition**: NR = class 0 (label 0), TSR = class 1 (label 1). Binary classification.

**Evaluation**: LOSO over all subjects common to both NR and TSR NPZs (subjects not present in both are dropped automatically by `combine_nr_tsr()`).

**Chance baseline**: 50% (binary).

```bash
cd transfer_study

python 2_train_probe_reading_task_cet_mae.py \
  --npz-nr  exports/embeddings_NR_pretrained_unlabeled/embeddings.npz \
  --npz-tsr exports/embeddings_TSR_pretrained_unlabeled/embeddings.npz \
  --checkpoint ../checkpoints/cet_mae/cet_mae_transfer3_a100_80gb_best.pt \
  --bart-path ../models/huggingface/bart-large \
  --output-dir results/A2_frozen_probe_reading_task \
  --head-type linear \
  --freeze-encoder \
  --epochs 50 \
  --lr 1e-3 \
  --batch-size 64 \
  --val-ratio 0.15 \
  --seed 42
```

Or:

```bash
bash run_scripts/run_probe_reading_task_cet_mae.sh
```

**Expected runtime**: ~10–25 minutes on GPU.

---

## 10. Step 6 — Run Experiment B1 (Fine-Tuned Sentiment)

**Purpose**: Measure the initialisation advantage of CET-MAE pretraining. The encoder is fully unfrozen and trained jointly with the classification head. Running twice — once from the pretrained checkpoint and once with `--random-init` — isolates the exact value of pretraining by holding everything else constant.

**Optimisation**: Differential learning rates — encoder at 1e-5 (slow, to preserve learned representations) and classification head at 1e-3 (fast, to adapt quickly to the task).

**The pretraining advantage is**: `B1_pretrained accuracy − B1_random accuracy`.

### B1 — Pretrained initialisation

```bash
cd transfer_study

python 1_train_finetune_sentiment_cet_mae.py \
  --npz exports/embeddings_SR_pretrained_unlabeled/embeddings.npz \
  --labels-csv labels/sentiment_normal_reading.csv \
  --checkpoint ../checkpoints/cet_mae/cet_mae_transfer3_a100_80gb_best.pt \
  --bart-path ../models/huggingface/bart-large \
  --output-dir results/B1_finetune_sentiment_pretrained \
  --encoder-lr 1e-5 \
  --head-lr 1e-3 \
  --epochs 25 \
  --batch-size 32 \
  --val-ratio 0.15 \
  --seed 42
```

### B1 — Random initialisation control (⚠️ required for RQ3)

```bash
python 1_train_finetune_sentiment_cet_mae.py \
  --npz exports/embeddings_SR_pretrained_unlabeled/embeddings.npz \
  --labels-csv labels/sentiment_normal_reading.csv \
  --random-init \
  --output-dir results/B1_finetune_sentiment_random \
  --encoder-lr 1e-5 \
  --head-lr 1e-3 \
  --epochs 25 \
  --batch-size 32 \
  --val-ratio 0.15 \
  --seed 42
```

Or:

```bash
bash run_scripts/run_finetune_sentiment_cet_mae.sh
```

> **Important**: When `--random-init` is passed, no `--checkpoint` argument is needed or used. The encoder is initialised with random weights drawn from the default PyTorch weight initialisation scheme, using `--seed` for reproducibility.

**Expected runtime**: ~30–60 minutes per run on GPU (21 folds × 25 epochs, full encoder forward+backward each step).

---

## 11. Step 7 — Run Experiment B2 (Fine-Tuned Reading Task)

**Purpose**: Same as B1 but for the NR vs. TSR cognitive-state binary task. Measures the pretraining advantage on a task with a published EEG benchmark.

### B2 — Pretrained initialisation

```bash
cd transfer_study

python 2_train_finetune_reading_task_cet_mae.py \
  --npz-nr  exports/embeddings_NR_pretrained_unlabeled/embeddings.npz \
  --npz-tsr exports/embeddings_TSR_pretrained_unlabeled/embeddings.npz \
  --checkpoint ../checkpoints/cet_mae/cet_mae_transfer3_a100_80gb_best.pt \
  --bart-path ../models/huggingface/bart-large \
  --output-dir results/B2_finetune_reading_pretrained \
  --encoder-lr 1e-5 \
  --head-lr 1e-3 \
  --epochs 25 \
  --batch-size 32 \
  --val-ratio 0.15 \
  --seed 42
```

### B2 — Random initialisation control (⚠️ required for RQ3)

```bash
python 2_train_finetune_reading_task_cet_mae.py \
  --npz-nr  exports/embeddings_NR_pretrained_unlabeled/embeddings.npz \
  --npz-tsr exports/embeddings_TSR_pretrained_unlabeled/embeddings.npz \
  --random-init \
  --output-dir results/B2_finetune_reading_random \
  --encoder-lr 1e-5 \
  --head-lr 1e-3 \
  --epochs 25 \
  --batch-size 32 \
  --val-ratio 0.15 \
  --seed 42
```

Or:

```bash
bash run_scripts/run_finetune_reading_task_cet_mae.sh
```

**Expected runtime**: ~30–60 minutes per run on GPU.

---

## 12. Step 8 — Run Experiment C1 (Relations Probe)

**Purpose**: Extension of the frozen probing study to semantic relation classification. Tests whether the encoder represents higher-level semantic structure beyond sentiment.

```bash
cd transfer_study

python train_probe_relations_cet_mae.py \
  --npz-nr exports/embeddings_NR_pretrained_unlabeled/embeddings.npz \
  --labels-csv labels/relations_normal_reading.csv \
  --checkpoint ../checkpoints/cet_mae/cet_mae_transfer3_a100_80gb_best.pt \
  --bart-path ../models/huggingface/bart-large \
  --output-dir results/C1_frozen_probe_relations \
  --head-type linear \
  --freeze-encoder \
  --epochs 50 \
  --lr 1e-3 \
  --seed 42
```

---

## 13. Step 9 — Run Experiment C2 (Noise Control)

**Purpose**: Sanity check / negative control. Phase-randomised EEG destroys all temporal structure while preserving the spectral power distribution. If the model still performs above chance on phase-randomised inputs, the results are driven by spectral features rather than temporal EEG structure. If it falls to chance, the above-chance results in A1/A2 are genuine.

```bash
cd transfer_study

python eval_noise_control_cet_mae.py \
  --npz exports/embeddings_SR_pretrained_unlabeled/embeddings.npz \
  --labels-csv labels/sentiment_normal_reading.csv \
  --checkpoint ../checkpoints/cet_mae/cet_mae_transfer3_a100_80gb_best.pt \
  --bart-path ../models/huggingface/bart-large \
  --output-dir results/C2_noise_control_sentiment \
  --n-noise-permutations 10 \
  --seed 42
```

Or:

```bash
bash run_scripts/run_noise_control_cet_mae.sh
```

---

## 14. Evaluation & Analysis Scripts

After running experiments, use the analysis scripts to build result tables and visualisations.

### Build summary result tables

```bash
cd transfer_study/analysis

python build_tables.py \
  --results-dir ../results \
  --output-dir ../results/tables \
  --experiments A1 A2 B1_pretrained B1_random B2_pretrained B2_random
```

This reads each experiment's `summary.json`, computes mean ± std and 95% CI across LOSO folds, and outputs:
- `results/tables/main_results.csv` — all metrics in tidy format
- `results/tables/main_results.tex` — LaTeX table ready for paper

### Run significance tests

```bash
python analysis/test_stats.py \
  --results-dir results \
  --pair A1 B1_pretrained \
  --pair B1_pretrained B1_random \
  --pair A2 B2_pretrained \
  --pair B2_pretrained B2_random
```

Uses paired bootstrap tests (n=10000) on per-subject accuracy values across LOSO folds.

### Layer-wise probe analysis

```bash
python analysis/layerwise_probe.py \
  --checkpoint ../checkpoints/cet_mae/cet_mae_transfer3_a100_80gb_best.pt \
  --npz exports/embeddings_SR_pretrained_unlabeled/embeddings.npz \
  --labels-csv labels/sentiment_normal_reading.csv \
  --output-dir results/C1_layerwise_probe
```

### t-SNE / UMAP embedding visualisation

```bash
python analysis/tsne_umap.py \
  --npz exports/embeddings_SR_pretrained_unlabeled/embeddings.npz \
  --labels-csv labels/sentiment_normal_reading.csv \
  --output-dir results/visualisations/SR_sentiment \
  --method umap
```

---

## 15. Understanding the Output Files

Each experiment writes a structured output directory. Here is what every file means.

```
results/A1_frozen_probe_sentiment/
│
├── run.log                          ← Full text log of the run (timestamps, metrics per epoch)
├── label_map.json                   ← Mapping from integer label to class name string
├── per_subject_results.csv          ← One row per LOSO fold: subject_id, accuracy, balanced_acc, macro_f1
├── all_predictions.csv              ← Every test prediction across all folds: subject, sentence, true, pred, prob_*
├── subject_accuracy.png             ← Bar chart of per-subject test accuracy
├── overall_confusion_matrix.png     ← Aggregated confusion matrix across all folds
├── overall_probability_histogram.png ← Distribution of predicted probabilities by true class
├── overall_<name>_roccurve.png      ← Aggregated ROC curve
├── overall_<name>_prcurve.png       ← Aggregated precision-recall curve
├── overall_<name>_reliability.png   ← Reliability / calibration diagram
├── summary.json                     ← KEY FILE: mean±std per metric + 95% CI across LOSO folds
│
└── fold_<N>_<subject>/              ← One directory per LOSO fold
    ├── best_linear_probe.pt         ← Best checkpoint for this fold (by val macro-F1)
    ├── training_curves.png          ← Loss, accuracy, balanced acc, macro-F1 vs epoch
    ├── confusion_matrix.png         ← Confusion matrix for this subject's test set
    ├── confusion_matrix.csv
    ├── history.csv                  ← Epoch-by-epoch metrics (train + val)
    ├── predictions.csv              ← Per-sentence predictions for train, val, test splits
    ├── metrics.json                 ← Final test metrics for this fold
    ├── split_manifest.csv           ← Which subjects were in train/val/test for this fold
    ├── embeddings.npz               ← Pooled EEG features for the test subject
    ├── pca_by_subject.png           ← PCA scatter coloured by subject identity
    └── pca_by_taskname.png          ← PCA scatter coloured by task/label
```

### The summary.json structure

```json
{
  "num_subjects": 21,
  "class_names": ["Negative", "Neutral", "Positive"],
  "summary_mean_std": {
    "accuracy":          {"mean": 0.421, "std": 0.087},
    "balanced_accuracy": {"mean": 0.389, "std": 0.092},
    "macro_f1":          {"mean": 0.361, "std": 0.101}
  },
  "summary_bootstrap_ci": {
    "accuracy":          {"mean": 0.421, "ci_lower": 0.381, "ci_upper": 0.462, "n_boot": 1000},
    "balanced_accuracy": {"mean": 0.389, "ci_lower": 0.348, "ci_upper": 0.431, "n_boot": 1000},
    "macro_f1":          {"mean": 0.361, "ci_lower": 0.318, "ci_upper": 0.402, "n_boot": 1000}
  },
  "aggregate_confusion_matrix": [[...], [...], [...]]
}
```

---

## 16. Interpreting Results

### Significance thresholds

| Task | Chance | Meaningful above-chance |
|------|--------|------------------------|
| A1 (3-class sentiment) | 33.3% accuracy | > 40% with CI not overlapping 33.3% |
| A1 (binary sentiment) | 50.0% accuracy | > 57% with CI not overlapping 50% |
| A2 (NR vs TSR) | 50.0% accuracy | > 57% with CI not overlapping 50% |
| B1/B2 pretrained vs random | — | Pretrained CI above random CI (non-overlapping) |

### Reading the pretraining advantage (RQ3)

Look at `summary.json` in:
- `results/B1_finetune_sentiment_pretrained/summary.json`
- `results/B1_finetune_sentiment_random/summary.json`

The pretraining advantage = pretrained_mean − random_mean for each metric. Use `analysis/test_stats.py` to test whether this gap is statistically significant.

### Reading the fine-tuning benefit (RQ4)

Compare:
- A1 `summary.json` (frozen probe accuracy) vs. B1_pretrained `summary.json` (fine-tuned accuracy)
- A2 `summary.json` vs. B2_pretrained `summary.json`

If fine-tuned accuracy is substantially and significantly higher than frozen probe accuracy, fine-tuning the encoder adds value beyond what the frozen representations already encode.

---

## 17. Experiment Design Notes

### Why LOSO cross-validation?

EEG data is highly subject-specific. A model trained and evaluated on the same subjects, even with a standard random split, can inflate performance by learning subject-specific artefacts. LOSO guarantees that the test subject is completely unseen during training — it is the most conservative and scientifically valid evaluation for EEG.

### Why ZuCo 2.0 NR vs. TSR (not ZuCo 2.0 NR vs. ZuCo 1.0 SR)?

An earlier version of this study planned to use ZuCo 2.0 NR vs. ZuCo 1.0 SR for the reading-task classification. This was revised for the following reasons:

- **Different subjects**: ZuCo 1.0 has 21 subjects; ZuCo 2.0 has 18. A LOSO design across two datasets requires the same subject to appear in both conditions. With disjoint pools, LOSO is impossible.
- **Confounded classification signal**: Any above-chance result would be ambiguous — the classifier could be learning ZuCo-1 vs. ZuCo-2 recording differences (different EEG cap layout, different preprocessing pipeline) rather than Normal Reading vs. Sentiment Reading cognitive differences.
- **Cleaner within-dataset control**: Using ZuCo 2.0 NR vs. ZuCo 2.0 TSR keeps subjects, recording setup, and preprocessing identical across both classes. The only meaningful difference is the reading task itself.

### Why differential learning rates in B1/B2?

The encoder was pre-trained on a large, multi-objective task. Fine-tuning the entire encoder at the same learning rate as the classification head would risk catastrophic forgetting — the encoder would rapidly lose its pre-trained representations before the head has had time to learn which features matter for the downstream task. Setting `encoder_lr = 1e-5` and `head_lr = 1e-3` (a 100× ratio) allows the head to adapt quickly while the encoder evolves slowly and stably.

### What does `--random-init` actually do?

When `--random-init` is passed, the script initialises the `CETMAEprojectlatebart` model with default PyTorch weight initialisation (Xavier uniform for linear layers, normal for embeddings) and does **not** load any checkpoint. The model architecture is identical to the pretrained version. This is the cleanest possible baseline for isolating the value of pretraining.

---

## 18. Troubleshooting

### "Could not find EEG/features array in NPZ"

Your NPZ was exported with a different key name. The scripts search for keys in order: `normalized_input_embeddings`, `input_embeddings`, `eeg`, `embeddings`, `features`. Run:

```python
import numpy as np
npz = np.load('exports/.../embeddings.npz', allow_pickle=True)
print(npz.files)
```

and verify the key name. If the key is non-standard, the extractor script may need a `--eeg-key` argument.

### "NR and TSR token shapes differ"

The NR and TSR NPZs were exported with different max sequence lengths (`L`). This happens if the exports were built at different times with different padding settings. Re-export both using `reset_and_rebuild_exports.sh` to ensure consistent padding.

### "B1/B2 fine-tuning requires raw token-level EEG inputs (got 2D shape)"

Your NPZ contains pre-pooled sentence embeddings of shape `(N, 1024)` rather than raw token-level EEG of shape `(N, L, D)`. Fine-tuning scripts need the raw token array. Re-export without the `--pool` flag in the extractor. For frozen probe scripts (A1, A2), pass `--input-is-pooled` to use pooled arrays.

### "CUDA out of memory during fine-tuning"

Reduce `--batch-size` to 16 or even 8. The fine-tuning experiments run the full encoder forward pass every batch, which is significantly more memory-intensive than the frozen probe runs.

### "Missing keys: pos_embed_e, e_branch..." when loading checkpoint

The checkpoint was saved with a `module.` prefix (from DistributedDataParallel). The `load_checkpoint_state()` function in all experiment scripts strips `module.`, `model.`, and `backbone.` prefixes automatically. If you still see this error, verify that the checkpoint file is not corrupted by checking:

```python
import torch
ckpt = torch.load('checkpoints/cet_mae/cet_mae_transfer3_a100_80gb_best.pt', map_location='cpu')
print(type(ckpt), list(ckpt.keys())[:10])
```

---

## 19. Directory Reference

```
transfer_study/
├── 1_train_probe_sentiment_cet_mae.py         # Exp A1 (final version)
├── 2_train_probe_reading_task_cet_mae.py      # Exp A2 (final version)
├── 1_train_finetune_sentiment_cet_mae.py      # Exp B1 (final version)
├── 2_train_finetune_reading_task_cet_mae.py   # Exp B2 (final version)
├── train_probe_relations_cet_mae.py           # Exp C1
├── eval_noise_control_cet_mae.py              # Exp C2
│
├── extract_encoder_embeddings_cet_mae.py      # Build SR and NR NPZs
├── build_tsr_npz_from_pickles.py              # Build TSR NPZ
├── rebuild_nr_npz_exact_v2.py                 # Rebuild NR NPZ with exact v2 schema
├── fix_sr_seq_len.py                          # Fix sequence-length mismatch in SR NPZ
├── validate_sr_nr_exports.py                  # Cross-validate SR and NR exports
├── validate_nr_export_against_v2.py           # Validate NR export against ZuCo 2.0 v2
├── merge_pickles_for_extractor.py             # Merge pickle files for extraction
├── reset_and_rebuild_exports.sh               # Full rebuild of all NPZ exports
│
├── exports/
│   ├── embeddings_SR_pretrained_unlabeled/    # ZuCo 1.0 SR NPZ (A1, B1)
│   ├── embeddings_NR_pretrained_unlabeled/    # ZuCo 2.0 NR NPZ (A2, B2, C1)
│   └── embeddings_TSR_pretrained_unlabeled/   # ZuCo 2.0 TSR NPZ (A2, B2)
│
├── labels/
│   ├── sentiment_normal_reading.csv           # SST sentiment labels for SR
│   ├── relations_normal_reading.csv           # Semantic relation labels for NR
│   └── relations_normal_reading_utf8.csv
│
├── splits/
│   ├── loso_subject_splits.json               # Pre-built LOSO fold assignments
│   ├── dev_test_subject_split.json            # Dev/test subject split
│   ├── noise_control_split.json               # Split for C2
│   └── split_build_manifest.json
│
├── models/
│   ├── load_cetmae_encoder.py                 # Checkpoint loading + encoder extraction
│   ├── freeze_utils.py                        # Freeze/unfreeze utilities
│   ├── pooling.py                             # Masked mean-pooling
│   ├── probe_heads.py                         # Linear and MLP probe heads
│   └── huggingface/bart-large/               # Local BART-large cache
│
├── analysis/
│   ├── build_tables.py                        # Build LaTeX/CSV result tables
│   ├── layerwise_probe.py                     # Layer-wise representational analysis
│   ├── test_stats.py                          # Bootstrap significance tests
│   └── tsne_umap.py                           # t-SNE / UMAP visualisations
│
├── run_scripts/
│   ├── run_probe_sentiment_cet_mae.sh         # A1 launcher
│   ├── run_probe_reading_task_cet_mae.sh      # A2 launcher
│   ├── run_finetune_sentiment_cet_mae.sh      # B1 launcher (pretrained + random)
│   ├── run_finetune_reading_task_cet_mae.sh   # B2 launcher (pretrained + random)
│   └── run_noise_control_cet_mae.sh           # C2 launcher
│
├── utils/
│   ├── logger.py                              # Logging setup
│   ├── metrics_classification.py             # Metrics + bootstrap CI
│   └── seed.py                               # Global seed setter
│
├── results/                                   # Auto-generated by experiment runs (gitignored)
│   ├── A1_frozen_probe_sentiment/
│   ├── A2_frozen_probe_reading_task/
│   ├── B1_finetune_sentiment_pretrained/
│   ├── B1_finetune_sentiment_random/
│   ├── B2_finetune_reading_pretrained/
│   ├── B2_finetune_reading_random/
│   ├── C1_frozen_probe_relations/
│   ├── C2_noise_control_sentiment/
│   └── tables/                                # Output of build_tables.py
│
├── metadata/
│   ├── checkpoint_registry.csv                # All trained checkpoint metadata
│   └── sample_index_map.csv                   # NPZ row → ZuCo subject/sentence map
│
└── Exp_A1_*, Exp_A2_*, Exp_B1_*, Exp_B2_*    # Legacy intermediate versions (superseded)
```
