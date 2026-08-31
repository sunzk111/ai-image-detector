# Mixed100K 数据准备

正式入口与结果说明见 [README.md](README.md)。训练集 100,000 张，内部验证额外 4,000 张；二者均 Real/AI 各半。

| 训练来源 | Real | AI |
| --- | ---: | ---: |
| CIFAKE | 5,000 | 5,000 |
| SID_Set | 5,000 | 5,000 |
| WildFake ImageNet | 20,000 | — |
| WildFake Church / AFHQ | 各 6,000 | — |
| WildFake FFHQ / CelebA-HQ | 各 4,000 | — |
| WildFake DDIM / DDPM / ADM / Imagen / VQDM | — | 各 8,000 |

内部验证为 CIFAKE 500 + SID_Set 500 + WildFake 3,000 张。训练程序再固定划分 1,000 张校准、3,000 张评估；演示集 COCO/DALL·E 不属于此内部划分。

## 运行与恢复

先准备 `data/wildfake_eval` 中的演示图片，再从项目目录运行：

```bash
python -m pip install kagglehub datasets modelscope-hub
python -u prepare_mixed_dataset.py --base-config config.mixed100k.yaml --workers 8
```

沿用 `config.mixed100k.yaml` 可保留当前模型、增强、学习率与校准策略。脚本只在配额/划分检查通过后重新生成正式配置及 `config.mixed100k.smoke.yaml`；后者可用于 600 张训练、300 张内部验证、1 epoch 的可选 smoke run。

- `--dry-run`：打印配额，不下载、不改数据。
- `--existing-only`：缓存完整时重建 manifest/config；配额不完整时明确失败。
- 重新运行同一命令可继续准备；不要在同一个缓存目录更换 seed 或配额。
- 保留整个 `data/mixed100k_v2`，包括图片、SQLite 状态、test hash 缓存和 manifest。仅保存 CSV 不能恢复图片。
- 训练恢复应另行设置 `training.resume`；数据准备脚本会将此字段重置为 null，因此不要在恢复训练前不必要地重新生成配置。

## 下载与去重

CIFAKE 上游归档可能完整下载，但只复制所选图片。SID_Set 采用流式读取与小 shuffle buffer，标签 2 不参与。WildFake 五个 Real 归档完整下载，大型 AI ZIP 默认使用并行 HTTP Range 按需提取图片。

若服务器不支持 Range，脚本停止而非自动下载全部大型归档。只有接受完整下载的时间/磁盘成本后，才使用 `--download-mode full`。下载不需要 GPU。

脚本排除全部 WildFake COCO/DALL·E 归档，并对训练、内部验证及已存在的演示图片进行解码 RGB 精确去重。当前准备记录中索引了 13,841 张演示图片；这不检测缩放、重编码后的近重复。若演示集尚未存在，精确交集检查不可用，仅有来源排除，建议先下载演示集。

## Colab 存储提醒

`/content` 可能随运行时回收而消失。Notebook 的 Drive 代码/权重备份明确排除 `data/`；需要避免重下时，必须另外备份数据。GitHub 同样不保存数据、权重或运行输出。

官方来源：[CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)、[SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)、[WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake/summary)。请遵守上游许可与访问要求。
