# Robust AI Image Detector — Mixed100K

面向 TikTok TechJam 2026 的 AI 生成图像检测项目：以 DINOv2 为骨干，通过多来源训练数据、退化增强与 clean/degraded 配对学习，识别真实图像和 AI 生成图像。

当前版本是 **Mixed100K V2**。仓库包含训练、数据准备、阈值校准、鲁棒性评估代码，以及从实际 Colab 工作流整理的 [tiktok_techjam.ipynb](tiktok_techjam.ipynb)。**数据集、模型权重、运行输出和私人备份不随仓库分发。**

## 方法

- `facebook/dinov2-base`，输入 448×448；拼接 CLS token 与 patch token 均值。
- 分类头：LayerNorm → Linear(512) → GELU → Dropout(0.2) → 单个 logit；约 87.4M 参数。
- 标签：`0 = Real`，`1 = AI`；当 `sigmoid(logit) >= threshold` 时判定为 AI。
- clean 与 degraded 配对训练：两路 BCE + 权重 0.2 的特征一致性损失。
- 课程式 JPEG、Gaussian blur、缩放、噪声、颜色和中心裁剪增强；两类图像使用相同增强流程。
- 单 GPU 训练；CUDA 自动选择可用的 BF16/FP16，也支持 CPU。未实现多卡 DDP。

## 数据与划分

| 来源 | 训练 Real | 训练 AI | 额外内部验证 |
| --- | ---: | ---: | ---: |
| CIFAKE | 5,000 | 5,000 | 500 |
| SID_Set | 5,000 | 5,000 | 500 |
| WildFake | 40,000 | 40,000 | 3,000 |
| 合计 | **50,000** | **50,000** | **4,000** |

WildFake Real 来自 ImageNet、Church、AFHQ、FFHQ、CelebA-HQ；AI 来自 Diffusion_based 中的 DDIM、DDPM、ADM、Imagen、VQDM。SID_Set 只使用标签 0/1，排除 tampered 类别 2。具体配额、续传与存储说明见 [MIXED100K_README.md](MIXED100K_README.md)。

内部 4,000 张数据与训练集分离，且 Real/AI 均衡；再按固定 sample ID、seed=42 划为 **1,000 张阈值校准 + 3,000 张模型评估**。每个 epoch 只在校准子集上选择 balanced accuracy 最优阈值，并用另外 3,000 张的 accuracy 选择 `best.pt`。阈值与对应模型一起保存。

演示评估集单独位于 `data/wildfake_eval/`：COCO val2017 4,998 张 Real + DALL·E Advanced/DALLE3 8,843 张 AI，共 13,841 张。准备脚本排除全部 WildFake COCO/DALL·E 训练归档；在演示图片已存在时，还执行解码 RGB 精确重复检查。此检查不等于近重复或语义重复检测。

**演示集不得用于训练。当前展示阈值曾根据演示集结果手动调整，因此下面的结果是开发阶段演示 benchmark，而不是未接触过的最终测试成绩，也不能据此断言对所有生成器都具有相同泛化能力。**

## 快速开始

推荐在 Colab GPU/Linux 环境运行；Colab 不需要创建 `.venv`。打开本仓库的 `tiktok_techjam.ipynb`，按章节选择需要的单元格，不要直接把包含可选训练/备份的整本 notebook 全部运行。

在项目目录安装依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install kagglehub modelscope-hub
```

当前成功运行环境为 NVIDIA A100-SXM4-40GB、PyTorch 2.11.0+cu128、BF16。依赖文件使用最低版本约束，不是锁定环境；其他设备/版本未保证数值完全一致。首次构建模型会下载 Hugging Face DINOv2 权重。

### 1. 准备数据

先按 notebook 下载/解压演示集，再准备训练集，以便进行精确重叠检查。如果数据已完整存在，无须重新下载。

```bash
python -u prepare_mixed_dataset.py --base-config config.mixed100k.yaml --workers 8
```

这里特意使用当前 mixed100k 配置作为模板，保留自动阈值校准设置。脚本会重新生成 `config.mixed100k.yaml`、训练/验证 manifest，以及可选的 smoke 配置。`config.yaml` 是保留的基础模板，不是当前正式训练入口；其原始设置没有启用校准。

准备过程只使用 CPU/网络，无须占用 A100。重新运行同一命令可从缓存继续；`--dry-run` 只显示配额。完整缓存已就绪时可加 `--existing-only` 重建 manifest/config，它不会默默接受不足 100,000 张的训练集。

### 2. 训练

```bash
python train.py --config config.mixed100k.yaml
```

当前配置：10 epochs、batch size 16、梯度累积 1、backbone/head 学习率分别为 `1e-5`/`1e-4`，AdamW、weight decay 0.05、10% warmup + cosine schedule；前 1 个 epoch 冻结 backbone。梯度检查点关闭，显存不足时优先调小 batch size。

结果写入 `outputs/dinov2_mixed100k_v2/`：`best.pt`、`last.pt`、`history.json`、`best_threshold.json`、`calibration_split.json`。开始新实验前请更换 `output_dir` 或自行保存旧权重，避免覆盖已有实验。

### 3. 评估

```bash
python evaluate.py --config config.mixed100k.yaml \
  --checkpoint outputs/dinov2_mixed100k_v2/best.pt \
  --source validation_demo
```

当前配置默认从 checkpoint 的 `decision_threshold` 读取阈值；配置中的 `threshold: 0.5` 并不覆盖它。新训练模型会保存自己的校准阈值，不会自动变成下述手动阈值。

当前已训练权重的演示阈值为 **0.000005（5e-6）**。如果要在该权重上显式复现此设置：

```bash
python evaluate.py --config config.mixed100k.yaml \
  --checkpoint outputs/dinov2_mixed100k_v2/best.pt \
  --source validation_demo --threshold 0.000005
```

`--threshold` 只影响此次报告，不修改权重。不要把 5e-6 当作所有新模型通用的最佳阈值。仓库不含 `best.pt`；请自行训练，或将自己保存的可信 checkpoint 放到上述位置。仅克隆代码不能恢复本次训练权重。

快速检查可增加 `--conditions clean jpeg_q30 blur_sigma2 --max-samples 64`；这是小样本检查，不是完整 benchmark。非 Colab 环境还需把 `data.validation_demo.path` 改成实际演示集目录。

### 4. 查看报告与离线比较阈值

每次评估生成独立的 `outputs/dinov2_mixed100k_v2/evaluation_runs/<run_id>/`，包含：

- `robustness.csv/json`：Accuracy、AUROC、F1、Real/AI recall、FP/FN、balanced accuracy 等。
- `predictions.csv`：逐图原始 logit、FP32 sigmoid 分数、标签与预测。
- `threshold_sweep.csv`：固定候选阈值的比较；不会自动修改模型阈值。
- `score_distributions.csv`：Real/AI 分数分位数及饱和值计数。
- `run_status.json`：运行状态与参数。

完整评估也更新输出根目录下的 `robustness.csv/json`。报告同时给出 probability AUROC 与 raw-logit AUROC；后者有助于识别 sigmoid 饱和造成的排序信息损失。Accuracy 依赖阈值，AUROC 衡量排序，两者不能互换。

已有逐图预测时，无须再次使用 GPU：

```bash
python evaluate.py \
  --from-predictions outputs/dinov2_mixed100k_v2/evaluation_runs/<run_id>/predictions.csv \
  --threshold 0.000005 --thresholds 0.00001 0.000005 0.000003
```

将 `<run_id>` 换成实际目录名。正式报告应事先固定阈值，并用独立、未参与调参的数据评估。

## 当前演示结果

2026-08-31 的完整评估：每种条件 13,841 张，阈值 5e-6，使用当前保存的 Mixed100K V2 权重。

| 条件 | Accuracy | AUROC | AI recall |
| --- | ---: | ---: | ---: |
| Clean | 96.55% | 99.50% | 95.62% |
| JPEG q30 | 95.63% | 99.61% | 93.46% |
| Gaussian blur σ=1 | 93.90% | 98.50% | 92.32% |
| Gaussian blur σ=2 | 91.03% | 97.13% | 89.38% |
| Resize 0.25× | 93.87% | 98.33% | 92.91% |

15 种变换的平均 Accuracy 为 **95.27%**；最弱条件为 blur σ=2。Clean 中 FP=91（真实图误判为 AI），FN=387（AI 图漏检）。这些是同一批图片上的多种退化结果，不是相互独立的 16 个数据集；仍需独立生成器与真实分发场景验证。

## 文件与存储

核心文件：`train.py` / `model.py` / `dataset.py` / `augmentations.py` / `losses.py` / `metrics.py` / `utils.py`；数据准备入口 `prepare_mixed_dataset.py`；校准模块 `threshold_calibration.py`；评估入口 `evaluate.py`；正式配置 `config.mixed100k.yaml`。

`.gitignore` 排除 `data/`、`outputs/`、模型权重、缓存、密钥与压缩备份。Notebook 已清除运行输出与私有运行元数据，保留数据下载、训练、评估及可选 Drive 备份命令。**Notebook 的代码备份排除 data，不会备份训练图片；Colab 断开/更换运行时前请单独保存需要的数据和权重。**

## 数据来源与使用边界

- [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)
- [WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake/summary)
- [DINOv2 Base](https://huggingface.co/facebook/dinov2-base)

数据与预训练模型的许可、访问条件及使用限制以各自上游为准。本仓库不重新分发其内容，不额外授予这些资源的使用权。初始代码来源于 [ai_image_detector](https://github.com/Uncle416/ai_image_detector)，当前版本在此基础上加入 Mixed100K 数据准备、内部阈值校准和详细评估诊断。
