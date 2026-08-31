# Asteria — Robust AI Image Detection

**TikTok TechJam 2026 · Problem 5 · Mixed100K V2**

Asteria distinguishes real photographs from AI-generated images, including images affected by JPEG compression, blur, resizing, noise, color changes, and cropping. It combines a DINOv2 backbone with mixed-source training, paired clean/degraded learning, and model-specific threshold calibration.

- [Open the English Colab notebook](https://colab.research.google.com/github/sunzk111/ai-image-detector/blob/main/tiktok_techjam.ipynb)
- [Download the trained model and its configuration](https://github.com/sunzk111/ai-image-detector/releases/tag/model)
- [Full robustness results](outputs/dinov2_mixed100k_v2/robustness.csv)
- [Detailed dataset preparation guide](MIXED100K_README.md)

**For evaluation, you do not need to download the training set or train a model.** Follow sections 1–4 below. Training from scratch is a separate workflow in section 5.

## 1. Installation

The commands below target Linux or Google Colab. Colab does not require a `.venv`. Use a GPU for training and full evaluation; downloading and preparing data does not require a GPU.

```bash
git clone https://github.com/sunzk111/ai-image-detector.git
cd ai-image-detector
python -m pip install -r requirements.txt
python -m pip install kagglehub modelscope-hub
```

Run all subsequent commands from the project root. The notebook uses `/content/ai_image_detector` as its checkout directory; the command-line clone above uses `ai-image-detector`. Either name works when paths are configured correctly.

The reported experiment used an NVIDIA A100-SXM4-40GB, PyTorch 2.11.0+cu128, and BF16. Dependencies have minimum-version constraints, not a fully pinned environment. Different hardware/software or precision can change scores. `runtime.device: auto` selects an available accelerator or CPU; CPU evaluation is slower. On CUDA, `runtime.precision: auto` selects supported BF16/FP16. Model initialization also downloads the upstream `facebook/dinov2-base` model from Hugging Face.

## 2. Download the trained checkpoint from Releases

Open the [model release](https://github.com/sunzk111/ai-image-detector/releases/tag/model). Under **Assets**, download:

- [`best.pt`](https://github.com/sunzk111/ai-image-detector/releases/download/model/best.pt): the trained Mixed100K V2 checkpoint, approximately 1 GB.
- [`config.mixed100k.yaml`](https://github.com/sunzk111/ai-image-detector/releases/download/model/config.mixed100k.yaml): the matching release configuration.

Keep the code from the current `main` branch to get the updated English documentation and notebook. The release's automatically generated **Source code** archives refer to its older tag snapshot; they are not the model weights.

For a **fresh checkout**, download the assets with:

```bash
mkdir -p outputs/dinov2_mixed100k_v2
curl -fL --retry 3 \
  https://github.com/sunzk111/ai-image-detector/releases/download/model/best.pt \
  -o outputs/dinov2_mixed100k_v2/best.pt
curl -fL --retry 3 \
  https://github.com/sunzk111/ai-image-detector/releases/download/model/config.mixed100k.yaml \
  -o config.release.yaml
```

We intentionally save the release configuration as **`config.release.yaml`** to keep it separate from the repository's training configuration, `config.mixed100k.yaml`. Do not overwrite an existing checkpoint from your own experiment. The notebook checks file hashes and refuses to replace a different existing checkpoint.

Verify the downloaded files before loading them:

```bash
sha256sum outputs/dinov2_mixed100k_v2/best.pt config.release.yaml
```

Expected SHA-256 values, as published by GitHub for this release:

```text
best.pt              f1e2d5470db116f59a64f65d8e3b7bccf2fd5fdec7c163023dbbd7b6d35b9508
config.release.yaml  feb1765f6d18db1f0853f22669c7fdb8fd79204380e43774eb308138557849a4
```

Load only checkpoints from sources you trust. Git cloning does not download `best.pt`; it is distributed as a release asset, not as a regular source file.

## 3. Prepare the evaluation images

The demonstration benchmark contains **4,998 COCO val2017 real images** and **8,843 DALL·E Advanced/DALLE3 AI images**, for a total of **13,841 images**. These images must not be used for training.

If you already have them, skip downloading and set `data.validation_demo.path` in `config.release.yaml` to their parent directory. The immediate class folders must match `class_to_label`, for example:

```text
data/wildfake_eval/
├── coco/                       # label 0: real; nested directories are supported
│   └── val2017/...
└── DALLE/                      # label 1: AI; class names are case-insensitive
    └── Advanced/DALLE3/...
```

The equivalent `coco/` and `dalle/` folders with images directly inside also work. If your folders are `data/coco` and `data/dalle`, set the parent path to `data`; do not point it at only one class. Avoid including unrelated class folders or extra images in the benchmark directory.

Otherwise, use the notebook's download cells or these commands:

```bash
mkdir -p data/wildfake_raw data/wildfake_eval
MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS=4 ms-hub download hy2628982280/WildFake \
  "Images/Diffusion_based/DALLE.zip" "label_csv_files/dalle3.csv" \
  --repo-type dataset --local-dir data/wildfake_raw --max-workers 2
MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS=4 ms-hub download hy2628982280/WildFake \
  "Images/Real/coco.zip" "label_csv_files/real_coco.csv" \
  --repo-type dataset --local-dir data/wildfake_raw --max-workers 2
unzip -n -q data/wildfake_raw/Images/Diffusion_based/DALLE.zip \
  'DALLE/Advanced/DALLE3/*' -d data/wildfake_eval
unzip -n -q data/wildfake_raw/Images/Real/coco.zip \
  '*val2017/*' -d data/wildfake_eval
```

This downloads the full upstream DALLE and COCO archives before extracting the selected subsets: roughly 28 GB of archives plus space for extracted files. Allow sufficient disk space and time. Downloading the evaluation data is separate from downloading the model. Follow the upstream datasets' access and licensing requirements.

## 4. Run inference or evaluate the released model

### Submission interface: image directory to JSON

Place the companion **`predict.py`** in the project root alongside `model.py`, `augmentations.py`, and `utils.py`. This separate script leaves `evaluate.py` unchanged. It accepts arbitrary unlabeled images; it does not need COCO/DALL·E, a training set, class folders, or manifests. For this workflow, skip section 3 and point it at your own image directory:

```bash
python predict.py --config config.release.yaml \
  --checkpoint outputs/dinov2_mixed100k_v2/best.pt \
  --input-dir /path/to/images --output-json predictions.json
```

It recursively scans JPG/JPEG, PNG, WEBP, BMP, TIF/TIFF files and writes a JSON array in stable path order. Example format only (the numbers below are illustrative, not measured results):

```json
[
  {"image_path": "example.jpg", "pred": 0.123},
  {"image_path": "nested/another.png", "pred": 0.987}
]
```

`image_path` is relative to `--input-dir`, using forward slashes. **`pred` is a continuous AI score in [0, 1], not a binary label.** It uses the checkpoint's model configuration and the same clean-image preprocessing as evaluation, with FP32 sigmoid after the model forward. No decision threshold is applied to these JSON scores. For an optional binary decision with the released model, compare `pred >= 0.000005` separately; do not replace the exported score with that decision.

Use `--device cpu --num-workers 0` for CPU inference, or lower `--batch-size` if memory is limited. The default worker count is 0; `--num-workers 4` enables parallel image loading. Empty input directories, unreadable images, invalid scores, and existing output files raise errors rather than silently skipping images or overwriting results. Symlinked image files are rejected and directory symlinks are not followed. JSON is written after all images are scored successfully.

### Labeled robustness benchmark

First, an optional small smoke check verifies loading and the evaluation pipeline:

```bash
python evaluate.py --config config.release.yaml \
  --checkpoint outputs/dinov2_mixed100k_v2/best.pt \
  --source validation_demo \
  --conditions clean jpeg_q30 blur_sigma2 --max-samples 64
```

Then run the **full benchmark**, without sample or condition limits:

```bash
python evaluate.py --config config.release.yaml \
  --checkpoint outputs/dinov2_mixed100k_v2/best.pt \
  --source validation_demo
```

The configuration uses `evaluation.threshold_source: checkpoint`, so the evaluator reads `best.pt`'s `decision_threshold`. The released model's decision threshold is **0.000005 (5e-6)**. Check the printed `Decision threshold` line. A numeric `evaluation.threshold` in YAML does not override a checkpoint threshold in this mode.

To explicitly use the reported operating point for this released model, append `--threshold 0.000005`. This changes only the current report, not the checkpoint. New models should use their own calibrated threshold, not automatically inherit 5e-6.

### Reports

Each evaluation creates `outputs/dinov2_mixed100k_v2/evaluation_runs/<run_id>/` with:

| File | Contents |
| --- | --- |
| `robustness.csv` / `robustness.json` | Per-condition accuracy, AUROC, F1, recalls, FP/FN counts, and summary metrics |
| `predictions.csv` | Per-image labels, logits, FP32 sigmoid scores, predictions, and error types |
| `threshold_sweep.csv` | Diagnostic comparisons at predefined thresholds; no automatic checkpoint change |
| `score_distributions.csv` | Real/AI score quantiles and saturation counts |
| `run_status.json` | Run settings, completion status, or failure details |

A full run also refreshes `robustness.csv/json` beside the checkpoint. The new run folder preserves its own reports. Use a unique `--output-dir` if you do not want those top-level reports refreshed. A smoke check is not a full benchmark result.

For CPU-only threshold comparisons from an existing prediction file:

```bash
python evaluate.py \
  --from-predictions outputs/dinov2_mixed100k_v2/evaluation_runs/<run_id>/predictions.csv \
  --threshold 0.000005 --thresholds 0.00001 0.000005 0.000003
```

Replace `<run_id>` with an actual run directory. Accuracy depends on a threshold; AUROC measures ranking. The evaluator reports both probability AUROC and raw-logit AUROC to expose possible sigmoid-saturation effects.

## 5. Train from scratch

Skip this section if you only want to test the released model. Use the repository's **`config.mixed100k.yaml`**, not the release-only `config.release.yaml`, as the training template.

### Prepare Mixed100K

Prepare the demonstration images first so that the dataset builder can reject exact decoded-image overlap. Then run:

```bash
python -u prepare_mixed_dataset.py --base-config config.mixed100k.yaml --workers 8
```

This creates 100,000 training images and 4,000 additional internal-validation images, their manifests, and updated training/smoke configurations. It preserves the template's model, augmentation, and calibration settings. Do not use the older `config.yaml` template for this workflow: it does not enable the current calibration policy.

- Re-run the same preparation command to continue from its cache.
- `--dry-run` prints quotas without downloads.
- `--existing-only` rebuilds from a complete cache; insufficient quotas fail explicitly.
- Keep the same seed and quotas when reusing an existing cache.

See [MIXED100K_README.md](MIXED100K_README.md) for source-level quotas and download behavior.

### Start training

```bash
python train.py --config config.mixed100k.yaml
```

The configured output directory is `outputs/dinov2_mixed100k_v2/`. **After dataset preparation and before training**, change `output_dir` to a new experiment directory if that path already contains downloaded weights or a previous run. Preparation resets this directory and `training.resume`, so apply experiment-specific edits after preparation.

| Setting | Value |
| --- | --- |
| Epochs / batch size / gradient accumulation | 10 / 16 / 1 |
| Backbone / head learning rate | 1e-5 / 1e-4 |
| Optimizer / weight decay | AdamW / 0.05 |
| Schedule | 10% warmup, then cosine decay |
| Backbone freezing | First epoch |
| Input size | 448 × 448 |
| Gradient checkpointing | Disabled |
| Evaluation batch size | 32 |

Each training sample produces a clean and a degraded view. Training concatenates the two views, so batch size 16 processes 32 image views per forward pass. Reduce training/evaluation batch sizes if GPU memory is insufficient. Set `data.num_workers: 0` when diagnosing DataLoader issues or using a constrained environment.

An optional one-epoch pipeline smoke test is available **after preparation**:

```bash
python train.py --config config.mixed100k.smoke.yaml
```

It uses 600 training and 300 internal-validation images and writes to a separate smoke output directory. It is not expected to reproduce full-model accuracy.

Training saves `best.pt`, `last.pt`, `history.json`, `best_threshold.json`, and `calibration_split.json`. To resume an interrupted run, set `training.resume` to its compatible `last.pt` and keep the data, split, and calibration policy unchanged.

Evaluate a newly trained model with its training configuration and actual checkpoint path:

```bash
python evaluate.py --config config.mixed100k.yaml \
  --checkpoint outputs/dinov2_mixed100k_v2/best.pt \
  --source validation_demo
```

If you changed `output_dir`, update `--checkpoint` accordingly. Do not add the released model's manual threshold to this command for a new experiment.

## 6. Model, training data, and threshold policy

### Architecture and robustness training

- Backbone: `facebook/dinov2-base`; approximately **87.4 million total parameters**.
- Pooling: concatenate the CLS token and the mean of patch tokens.
- Head: LayerNorm → Linear(512) → GELU → Dropout(0.2) → one logit.
- Labels: **0 = real**, **1 = AI**; predict AI when `sigmoid(logit) >= threshold`.
- Loss: clean-view BCE + degraded-view BCE + 0.2 × cosine feature-consistency loss.
- Augmentations: JPEG quality 90/70/50/30, Gaussian blur radius 0.5/1/2, resize scale 0.5/0.25, noise 0.02/0.05/0.1, color jitter 0.2, and center crop 0.8.

For a 10-epoch run, the curriculum uses mild 0–1-operation degradation in epochs 1–3, intermediate 1–2-operation degradation in epochs 4–6, and the full severity range with 1–3 operations in epochs 7–10. This explicitly exposes the model to strong blur and JPEG q30 while retaining clean supervision. Both classes receive the same augmentation policy. There is no dedicated deblurring module or extra blur-specific oversampling in this version.

### Data selection

| Source | Training real | Training AI | Additional internal validation |
| --- | ---: | ---: | ---: |
| CIFAKE | 5,000 | 5,000 | 500 |
| SID_Set | 5,000 | 5,000 | 500 |
| WildFake | 40,000 | 40,000 | 3,000 |
| **Total** | **50,000** | **50,000** | **4,000** |

WildFake real sources are ImageNet, Church, AFHQ, FFHQ, and CelebA-HQ. AI sources are DDIM, DDPM, ADM, Imagen, and VQDM under `Diffusion_based`. All WildFake COCO/DALL·E archives are excluded from training. SID_Set uses labels 0/1 only, excluding the tampered class 2.

The preparation script rejects exact decoded-RGB duplicates across the selected splits and available demonstration images. This is not perceptual or near-duplicate detection, and it does not establish that the pretrained backbone has never seen related content.

### Calibration versus the reported release threshold

The 4,000 internal-validation images are class-balanced and disjoint from training. A fixed sample-ID/seed-42 split reserves **1,000 for threshold calibration** and **3,000 for model selection**. At each epoch, calibration maximizes balanced accuracy on the 1,000 images; checkpoint selection uses accuracy on the other 3,000. The selected threshold is stored with its model.

**The release's final 5e-6 threshold was subsequently selected manually using demonstration-benchmark results.** The demonstration images were not used for gradient training, but they did influence this operating point. Therefore the reported results are a development benchmark, not an untouched final-test estimate. The sigmoid score is not claimed to be a calibrated real-world probability, and 5e-6 is not a universal optimum.

## 7. Results and limitations

The complete evaluation on August 31, 2026 used 13,841 images per condition and the released operating point of 5e-6.

| Condition | Accuracy | AUROC | AI recall |
| --- | ---: | ---: | ---: |
| Clean | 96.55% | 99.50% | 95.62% |
| JPEG q30 | 95.63% | 99.61% | 93.46% |
| Gaussian blur, radius 1 | 93.90% | 98.50% | 92.32% |
| Gaussian blur, radius 2 | 91.03% | 97.13% | 89.38% |
| Resize 0.25× | 93.87% | 98.33% | 92.91% |

Mean accuracy across the **15 transformed conditions** is **95.27%**. Strong blur is the weakest condition. Clean images produce 91 false positives and 387 false negatives; blur radius 2 increases these to 302 and 939. The conditions reuse the same source images and are not independent datasets. Full metrics are in [robustness.csv](outputs/dinov2_mixed100k_v2/robustness.csv); [threshold_sweep.csv](outputs/dinov2_mixed100k_v2/threshold_sweep.csv) contains diagnostic threshold comparisons.

Limitations and next steps:

- Evaluate on a fresh, held-out benchmark after fixing the threshold; avoid further tuning on the final test set.
- Test unseen generator families, near-duplicates, and compound real-world degradation.
- Study targeted strong-blur/JPEG sampling with controlled ablations; these proposed changes are not part of the reported model.
- Image-level detection only: no manipulation localization, video/audio detection, or production moderation guarantee.
- `evaluate.py` expects labeled datasets and produces robustness reports plus prediction CSVs. The separate `predict.py` provides the unlabeled image-directory → `image_path`/`pred` JSON interface; add the companion file to the repository before packaging the submission.
- The implementation is single-device; distributed training and measured inference-latency guarantees are not provided.

## 8. Files, storage, and attribution

Core files: `train.py`, `model.py`, `dataset.py`, `augmentations.py`, `losses.py`, `metrics.py`, and `utils.py`. Data preparation: `prepare_mixed_dataset.py`. Calibration: `threshold_calibration.py`. Robustness evaluation: `evaluate.py`. Unlabeled JSON inference: companion `predict.py`. Training configuration: `config.mixed100k.yaml`. Interactive workflow: [tiktok_techjam.ipynb](tiktok_techjam.ipynb).

Datasets are not included. Weights are distributed through Releases. The repository includes selected summary CSVs; per-image outputs, private backups, and full experiment directories are not generally tracked. The notebook's optional Drive backup excludes `data/`, but includes model/output files. Back up datasets separately if needed: Colab `/content` files can disappear when the runtime is recycled.

This version builds on the initial [ai_image_detector codebase](https://github.com/Uncle416/ai_image_detector), with Mixed100K preparation, internal threshold calibration, and detailed robustness diagnostics. Do not interpret the project as training DINOv2 from scratch.

Upstream resources:

- [DINOv2 Base](https://huggingface.co/facebook/dinov2-base)
- [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)
- [WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake/summary)

Use of upstream code, datasets, and pretrained models remains subject to their respective licenses, permissions, and access conditions. This repository does not grant additional rights to those resources.
