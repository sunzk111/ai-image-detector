# Mixed100K Dataset Preparation

See [README.md](README.md) for installation, release-weight downloads, training, evaluation, and results. **Skip training-data preparation when evaluating the released model.**

The training set contains 100,000 images, balanced between real and AI. A separate, balanced internal-validation set contains 4,000 images.

## Source quotas

| Source | Training real | Training AI | Internal validation |
| --- | ---: | ---: | ---: |
| CIFAKE | 5,000 | 5,000 | 500 |
| SID_Set | 5,000 | 5,000 | 500 |
| WildFake ImageNet | 20,000 | 0 | 750 |
| WildFake Church | 6,000 | 0 | 225 |
| WildFake AFHQ | 6,000 | 0 | 225 |
| WildFake FFHQ | 4,000 | 0 | 150 |
| WildFake CelebA-HQ | 4,000 | 0 | 150 |
| WildFake DDIM | 0 | 8,000 | 300 |
| WildFake DDPM | 0 | 8,000 | 300 |
| WildFake ADM | 0 | 8,000 | 300 |
| WildFake Imagen | 0 | 8,000 | 300 |
| WildFake VQDM | 0 | 8,000 | 300 |
| **Total** | **50,000** | **50,000** | **4,000** |

CIFAKE and SID_Set internal-validation quotas each contain 250 real and 250 AI images. SID_Set's tampered label 2 is excluded. WildFake AI images come from `Diffusion_based`, excluding all DALL·E archives; real sources exclude all COCO archives.

Training further splits the internal-validation set into 1,000 calibration and 3,000 model-selection images using fixed sample IDs and seed 42. The COCO/DALL·E demonstration benchmark is not part of this internal split.

## Run and resume preparation

Prepare the demonstration images under `data/wildfake_eval` first, following the main README or notebook. From the project root:

```bash
python -m pip install kagglehub datasets modelscope-hub
python -u prepare_mixed_dataset.py --base-config config.mixed100k.yaml --workers 8
```

Use the repository's training configuration as the template, not the older `config.yaml` and not the release companion saved as `config.release.yaml`. This preserves the original model, augmentation, optimizer, and automatic-calibration settings.

After all quota/split checks pass, the script writes manifests and regenerates `config.mixed100k.yaml` and `config.mixed100k.smoke.yaml`. The optional smoke run uses 600 training images, 300 internal-validation images, and one epoch.

- `--dry-run`: print quotas without downloading or modifying datasets.
- `--existing-only`: rebuild manifests/configurations from a complete cache; incomplete quotas raise an error.
- Repeat the same command to continue interrupted preparation. Do not change the seed or source quotas within the same cache.
- Keep the whole `data/mixed100k_v2` directory, including images, SQLite state, test-hash cache, and manifests. CSV files alone cannot restore the images.
- Preparation resets `training.resume` to null and `output_dir` to `outputs/dinov2_mixed100k_v2`. Set a new experiment directory or resume checkpoint **after** preparation, before starting training.

## Download behavior and overlap checks

CIFAKE may download its full upstream archive, but only selected images are copied into the mixed dataset. SID_Set uses streaming with a small shuffle buffer. The five WildFake real-image archives are downloaded in full; large AI ZIPs use parallel HTTP Range extraction by default.

If the server does not support Range requests, the script stops instead of silently downloading every large archive. Use `--download-mode full` only if you accept the additional download time and disk usage. No GPU is required for preparation.

The builder excludes all WildFake COCO/DALL·E archives and checks decoded-RGB exact duplicates across training, internal validation, and demonstration images already present locally. The reported preparation indexed 13,841 demonstration images. This does not detect resized/re-encoded near-duplicates or semantic overlap. If the demonstration images are absent, the cross-benchmark exact-overlap check is unavailable, although source exclusions still apply.

## Storage and access

Colab `/content` storage is temporary. The notebook's optional Drive backup excludes `data/`; back up datasets separately to avoid downloading them again. GitHub contains code and selected reports; the checkpoint is hosted under [Releases](https://github.com/sunzk111/ai-image-detector/releases/tag/model), not in the training-data directory.

Official sources: [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images), [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set), and [WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake/summary). Respect upstream licenses and access requirements; do not put credentials in a shared notebook.
