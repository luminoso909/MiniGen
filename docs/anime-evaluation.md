## **一、动漫头像评测入口**

在已安装依赖的 AutoDL 项目目录运行评测。训练配方、采样配置和评分脚本随仓库维护。真实图像、权重、缓存、生成结果、匿名评分和运行回执由运行环境单独保存。

### **1. 合成回归**

合成测试使用临时图像和合成权重身份。它覆盖 B2 图像 token 和梯度边界，以及四组评测的匿名题卡、锁分和解盲流程。运行只需已安装的 Python 测试依赖。

```bash
PYTHONPATH=src:scripts uv run --locked --offline --no-sync python -m pytest -q \
  -p no:cacheprovider tests/test_hybrid_image_tokens.py tests/test_anime_four_benchmark.py
```

### **2. 真实评测准备**

- 准备本地 `data/` 下与训练划分一致的图片和清单，以及 `artifacts/` 下的冻结 SDXL VAE、CLIP 和经核验的头像表示缓存。[划分脚本](../scripts/prepare_anime_faces_split.py) · [缓存脚本](../scripts/cache_anime_faces.py)
- 将原始题库 `configs/benchmark/anime_faces_v1.json` 及来源排除清单 `configs/benchmark/anime_faces_v1_exclusions.json` 放回相应位置。教师探针和分时诊断分别读取本地 `configs/evaluation/anime_repa_teacher_probe.json` 与 `configs/evaluation/anime_b1_time_eyes.json` 及其指定样本。[评分量表](../reports/v2/anime/benchmark_v1_rubric.md)
- 重现历史评测时，按所用范围配置恢复原始编号 checkpoint、对应冻结源码和训练验收回执。预检用对应配置和内容锁登记的 SHA-256 核对题库、量表、冻结源码、回执和权重。[四组 30k 范围配置](../configs/benchmark/anime_faces_v1_four_50m_30k.json) · [三组 20k 评测预登记](../reports/v2/anime/b1_b2_50m_benchmark_plan_2026-11-17.md)

### **3. 生成和评分**

| 对照 | 生成入口 | 评分入口 |
| --- | --- | --- |
| 四组 50M 30k | [采样脚本](../scripts/generate_anime_four_benchmark.py) 的 `preflight`、`generate`、`verify` | [评分脚本](../scripts/evaluate_anime_four_benchmark.py) 的 `render`、`freeze`、`reveal`、`report` |
| 三组 20k | [采样脚本](../scripts/generate_anime_scale_benchmark.py) 的 `preflight`、`generate` | [匿名题卡](../scripts/render_anime_benchmark_blind.py) 和 [评分脚本](../scripts/score_anime_scale_benchmark.py) 的 `freeze`、`reveal` |

执行历史采样前，将对应冻结源码的 `src/` 放在 `PYTHONPATH` 首位。先运行 `preflight` 核对资产和环境，再在已授权的 GPU 环境用 `generate --arm` 生成指定模型的图像。三组 20k 对照还需 30M 基线图片和原生成回执，可用 `--baseline-receipt` 指定回执路径。对生成文件和回执验收后制作匿名题卡，填写评分并锁分，再读取身份映射。评测产物默认写入本地 `artifacts/exports/anime-faces-b1/`。
