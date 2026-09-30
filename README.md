# MemoryVLA · 跨 episode 经验预填（免训练版）

> 本分支 `training-free-prefill` 基于官方 [MemoryVLA](https://github.com/shihao1895/MemoryVLA)（OpenVLA codebase，`openvla-codebase` 分支）。它**不做任何训练**，直接把**其他 episode** 的记忆 token 预填进 MemoryVLA 的记忆库，检验模型能否利用跨 episode 的经验。原仓库的 README 保留在本文后半部分。

## 做了什么

MemoryVLA 在每个 episode 内维护两个记忆库（CogMemBank / PerMemBank）。当前帧的 token（working memory）作为 Q，检索本局的历史记忆，结果经 GateFusion 融合后送进 DiT 动作头。本分支在记忆库里加了"固定条目"（pinned），流程如下：

1. **收集**：正常跑评测，不预填。每次调用模型时，记录进入记忆模块之前的原始 cog token（每帧 1 个）和 per token（每帧 N 个）。一局结束后按成败存入经验库。
2. **选经验**：新一局开始时，按模式从经验库里选 1 条轨迹。
3. **抽关键帧**：在这条轨迹上均匀取 8 帧（`linspace(0, T-1, 8)`，包含首尾两帧）。
4. **预填**：这 8 帧的 token 作为固定条目放在本局历史之前，即 `记忆 = [预填 8 帧] + [本局历史]`。固定条目整局都在，不会被挤出或合并，一局结束后清掉。

记忆库平时存的也是原始（融合前）token，预填的 token 和它们是同一种，所以不用训练就能直接接入。

### 经验模式（`exp_mode`）

| 模式 | 经验来源 | 选法 |
|---|---|---|
| `none` | 不预填（基线） | — |
| `other_init_success`（B） | 同任务、其他起点、成功 | 第一帧 cog token 与当前起点余弦相似度最高的一条 |
| `same_init_success`（C） | 同任务、同起点、成功 | 随机 |
| `same_init_failure`（D） | 同任务、同起点、失败 | 随机 |
| `other_task_success` | 其他任务、成功（安慰剂） | 随机 |
| `noise` | 高斯噪声，按同任务存储 token 的逐维均值和方差生成（安慰剂） | — |
| `other_init_success_same_color` / `other_init_success_diff_color` | 同任务、其他起点、成功，目标颜色与当前局相同 / 不同（仅 RememberColor） | 随机 |

"起点"指环境的初始场景。MIKASA 中 `env_seed = init_id + 1`；LIBERO 中是第 `init_id` 个初始状态。同一起点每次 reset 出来的场景完全相同。

### 配对设计

每局开始时（`/start_episode`）设置扩散噪声种子 `episode_seed = seed_offset + task_id*100000 + init_id*100 + trial`，与模式无关。所以同一个（任务, 起点, 第几次）在不同模式下场景相同、扩散噪声相同，只差预填的记忆，可以逐条配对比较。

`seed_offset` 的约定：阶段 1 收集用 `0`，评测用 `1000`，种子安慰剂用 `2000`。

## 代码改动

| 文件 | 内容 |
|---|---|
| `vla/experience.py`（新增） | 经验库：存储、按模式检索、关键帧选择（`uniform` / `event`）、构造预填条目、生成噪声 |
| `vla/memory_vla.py` | 记忆库支持固定条目（`set_pinned` / `clear_pinned`）；`predict_action` 可返回原始 token（`return_features`），支持 `timestep_stride` 和 `preserve_unmasked_actions`；诊断项：预填条目的注意力占比、GateFusion 门控值、相对"不预填"影子前向的动作偏移 |
| `deploy.py` | 新增 `/start_episode`、`/end_episode` 接口和经验相关参数（见下表） |
| `evaluation/mikasa/eval_mikasa.py`（新增） | MIKASA-Robo 评测客户端，逐局写 `*_results.jsonl` |
| `evaluation/libero/eval_libero.py`、`evaluation/libero/vla_policy.py` | LIBERO 评测支持 episode 接口、每个起点重复多次、逐局写 jsonl |
| `script/eval/mikasa/probe_round2.sh` | 第二轮完整流程：收集 → 各模式评测 → 可重复性检查 → 诊断 → 分析 |
| `script/eval/mikasa/probe_experience.sh` | 第一轮流程（`none` / B / C / D 四个模式） |
| `script/eval/mikasa/analyze_round2.py` | 按（任务, 起点）分层的 CMH 检验 + Holm 校正、种子安慰剂 McNemar、颜色检验、翻转率 |
| `script/eval/mikasa/analyze_diag.py`、`script/eval/mikasa/check_repro_round2.py` | 诊断汇总；与第一轮基线逐条对比的可重复性检查 |
| `script/eval/libero/probe_experience.sh`、`script/eval/libero/analyze_probe.py` | LIBERO 版流程和配对 McNemar 分析（第一轮 MIKASA 也用它分析） |
| `prismatic/models/backbones/llm/llama2.py` | 可以用环境变量 `MEMVLA_LLAMA2_7B_PATH` 指定 Llama-2-7B 的路径或镜像 |
| `vla/__init__.py` | 训练数据模块改为按需导入，推理时不加载训练依赖 |

新功能默认全部关闭。不传 `--exp_store_dir`、客户端也不调用 `/start_episode` 时，推理行为和原仓库一致（LIBERO 评测会额外写一份逐局 jsonl）。

### `deploy.py` 新增参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--exp_store_dir` | 无 | 经验库目录；不传则不能收集，也不能预填 |
| `--exp_keyframes` | `8` | 每条经验取几帧 |
| `--exp_keyframe_strategy` | `uniform` | `uniform` 均匀取；`event` 取首尾两帧和每次夹爪开合前的一步，不足时补离已选帧最远的帧 |
| `--exp_pin_timestep` | `orig` | 预填帧的时间步：`orig` 沿用原轨迹编号；`neg` 整体移到本局第 0 步之前；`zero` 全部为 0 |
| `--exp_pin_target` | `both` | 预填哪个记忆库：`both` / `cog` / `per` |
| `--timestep_stride` | `1` | 每次调用时记忆时间步的增量 |
| `--preserve_unmasked_actions` | 关 | 只裁剪归一化过的动作维度，不对夹爪做二值化（MIKASA 官方协议需要） |
| `--exp_diag`、`--exp_diag_path` | 关 | 逐步把诊断写进 jsonl。每步会多做一次影子前向，速度变慢 |

## 使用方法

### MIKASA-Robo（第二轮流程）

需要先安装 [MIKASA-Robo](https://github.com/CognitiveAISystems/MIKASA-Robo)（`mikasa_robo_suite`）和 ManiSkill，并下载 `memvla-mikasa.pt`。

```bash
CKPT_PATH=/path/to/memvla-mikasa.pt bash script/eval/mikasa/probe_round2.sh
```

常用环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CKPT_PATH` | 必填 | checkpoint 路径 |
| `PYTHON_BIN` | `python` | Python 解释器 |
| `TASK_IDS` | `1 2 3` | 任务：1 = InterceptMedium（IM），2 = RememberColor3（RC3），3 = RememberColor5（RC5） |
| `NUM_INITS` / `COLLECT_TRIALS` / `EVAL_TRIALS` | `30` / `5` / `4` | 起点数 / 每个起点收集几次 / 每个起点每个模式评测几次 |
| `PROFILE` | `full` | `full` 跑全部模式；`min` 只跑基线、噪声和颜色两组 |
| `COLLECT_ONLY` | `0` | 设为 `1` 时只收集 |
| `RUN_DIAG` | `1` | 设为 `0` 时跳过诊断 |
| `SKIP_REPRO` | `0` | 设为 `1` 时跳过与第一轮基线的可重复性检查 |
| `PORT`、`EXP_STORE_DIR`、`LOG_DIR`、`DIAG_DIR` | 见脚本 | 端口、经验库、结果和诊断目录 |

- 每完成一个（任务, 模式, 种子）组合，脚本会在 `$LOG_DIR/.done/` 写完成标记，中断后重跑会跳过已完成的部分。
- 多个任务可以用不同的 `PORT` 和 `TASK_IDS` 并行跑。模型加载会自动排队，避免几个进程的内存峰值叠加。每个进程加载时峰值要占用几十 GB 内存。
- 脚本会检查 checkpoint 大小是否为官方 `memvla-mikasa.pt` 的 33,507,444,130 字节，以防下载不完整。
- 默认从 `NousResearch/Llama-2-7b-hf` 读取 Llama-2 配置，可以用 `MEMVLA_LLAMA2_7B_PATH` 改成本地路径；离线环境需要再设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。
- 如果访问 HuggingFace 需要代理，请自己设置 `HTTP(S)_PROXY`。脚本会把 `127.0.0.1,localhost` 加进 `NO_PROXY`，避免客户端到本地服务的请求也走代理（那样会报 502）。

也可以手动启动服务，再逐步调用：

```bash
python deploy.py --saved_model_path /path/to/memvla-mikasa.pt --unnorm_key mikasa_dataset --cfg_scale 1.5 \
  --use_bf16 --preserve_unmasked_actions --action_chunking --action_chunking_window 4 \
  --exp_store_dir ./exp_store/mikasa --port 2345
# 阶段 1：收集经验（不预填）
python evaluation/mikasa/eval_mikasa.py --task_id 1 --num_inits 30 --trials_per_init 5 \
  --seed_offset 0 --record_experience --log_dir ./log/mikasa --port 2345
# 阶段 2：预填评测
python evaluation/mikasa/eval_mikasa.py --task_id 1 --num_inits 30 --trials_per_init 4 \
  --seed_offset 1000 --exp_mode other_init_success --log_dir ./log/mikasa --port 2345
# 分析
python script/eval/mikasa/analyze_round2.py --log_dir ./log/mikasa
```

### LIBERO

把 `script/eval/libero/probe_experience.sh` 里的 `ckpt_path` 改成自己的 checkpoint，按需修改 `task_ids`，然后运行：

```bash
bash script/eval/libero/probe_experience.sh
```

## 实验结果（MIKASA-Robo，`memvla-mikasa.pt`，任务 IM / RC3 / RC5）

协议：官方 `success_once` 判定、BF16、4 步动作块、`cfg_scale 1.5`。

第二轮设置：每个任务 30 个起点。阶段 1 每个起点收集 5 次，共 450 条；评测时每个模式每个起点跑 4 次，即每个模式 360 条，共 3000 条。第二轮的基线与第一轮逐条对比，60/60 一致。

| 模式 | 成功率 | 与同批起点基线的差值 | 95% 置信区间（按起点重抽样） | Holm 校正 p |
|---|---:|---:|---|---:|
| 基线 `none`（种子 1000 + 2000 合并） | 27.4%（197/720） | — | — | — |
| B 其他起点成功 | 32.2%（116/360） | +4.9pp | [+0.3, +9.4] | 0.28 |
| C 同起点成功 \* | 39.4%（104/264） | +5.3pp | [−0.6, +11.4] | 0.41 |
| D 同起点失败 | 31.4%（113/360） | +4.0pp | [−0.6, +8.6] | 0.41 |
| 别任务成功（安慰剂） | 29.7%（107/360） | +2.4pp | [−3.1, +7.4] | 0.71 |
| 噪声（安慰剂） | 27.5%（99/360） | +0.1pp | [−3.9, +4.3] | 0.96 |

\* 有 24 个起点在阶段 1 中 5 次全部失败，没有同起点的成功经验，所以 C 只覆盖剩下的 66 个起点。这批起点本身更容易（基线 34.1%），因此 C 的成功率要和 34.1% 比，不能和 27.4% 比。

其他检验：

- **种子安慰剂**：只换扩散种子、不预填（none@2000 对 none@1000），p = 0.54。
- **颜色**（RC3 + RC5）：同色经验 71/240，异色经验 74/240，p = 0.73。
- **模式之间直接对比**：B − 噪声 +4.7pp [0.0, +9.7]；B − D +0.8pp；C − D −0.4pp [−6.1, +5.7]。
- **诊断**：预填条目得到的注意力占比和噪声相当，也不随颜色、成败变化；门控值（cog 约 0.87，per 约 0.53）和第一步动作偏移（约 0.12）都与预填内容无关。
- **第一轮**（10 个起点 × 2 次）里 D 的 p = 0.021，原因是那组基线种子偏低；第二轮加大样本、加入安慰剂后不再成立。

**结论**：免训练地预填真实经验，成功率有约 +4~5pp 的正向趋势，噪声没有，但这个趋势没有达到统计显著。经验是成功还是失败、是否来自同一起点、颜色是否一致，都没有可检出的影响；诊断也显示，模型没有区分真实经验和噪声。现有检索通道没有利用跨 episode 经验的内容，下一步需要训练独立的 task memory 分支。

---

> 以下为原仓库 README。

# MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation
[Hao Shi](https://shihao1895.github.io/), [Bin Xie](https://xb534.github.io/), [Yingfei Liu](https://scholar.google.com/citations?user=pF9KA1sAAAAJ), [Lin Sun](https://github.com/linsun449), [Fengrong Liu](https://shihao1895.github.io/MemoryVLA/) [Tiancai Wang](https://scholar.google.com/citations?user=YI0sRroAAAAJ), [Erjin Zhou](https://scholar.google.com/citations?user=k2ziPUsAAAAJ), [Haoqiang Fan](https://scholar.google.com/citations?user=bzzBut4AAAAJ), [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ), [Gao Huang](https://scholar.google.com/citations?user=-P9LwcgAAAAJ)

Tsinghua University, Dexmal, MEGVII, TJU, HiT, StepFun

ICLR 2026

> This is the code for the paper "MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation".

### 🏠[MemoryVLA Project](https://shihao1895.github.io/MemoryVLA) | 🏠[MemoryVLA++ Project](https://shihao1895.github.io/MemoryVLA-PP-Web) | 📑[Paper](https://arxiv.org/abs/2508.19236) | 🤗[Models & Logs](https://huggingface.co/collections/shihao1895/memoryvla)

## 🌟 News

- 🔥 [2026-6-9] Extended journal version [MemoryVLA++](https://shihao1895.github.io/MemoryVLA-PP-Web) is available!
- 🔥 [2026-1-27] Our paper [MemoryVLA](https://arxiv.org/abs/2508.19236) is accepted by ICLR 2026!
- 🔥 [2025-11-5] The code of [MemoryVLA](https://arxiv.org/abs/2508.19236) is released! (Both MemoryVLA and MemoryVLA+)
- 🔥 [2025-10-20] Our VLA codebase [Dexbotic](https://github.com/Dexmal/dexbotic) is released, it now fully integrates MemoryVLA !
- 🔥 [2025-8-26] Our paper [MemoryVLA](https://arxiv.org/abs/2508.19236) is now on arxiv!

## Overview

MemoryVLA is a Cognition-Memory-Action framework for robotic manipulation inspired by human memory systems. It builds a hippocampal-like perceptual-cognitive memory to capture the temporal dependencies essential for current decision-making, enabling long-horizon, temporally aware action generation.

![MemoryVLA Overview](images/intro.png)

We release three versions of the code in separate branches:

- **[MemoryVLA](https://github.com/shihao1895/MemoryVLA/tree/openvla-codebase)**:  built upon the OpenVLA codebase.
- **[MemoryVLA+](https://github.com/shihao1895/MemoryVLA/tree/dexbotic-codebase)**:  built upon our self-developed [Dexbotic](https://dexbotic.com) codebase, which offers higher simulation performance.
- **MemoryVLA++**:  extended journey version of MemoryVLA.

## TODO

All components of MemoryVLA are now available, and MemoryVLA++ will be released in the coming months.

- [x] MemoryVLA (OpenVLA codebase)
  - [x] Code Release
  - [x] Model Weights Release
  - [x] Dataset Upload to HuggingFace

- [x] MemoryVLA+ (Dexbotic codebase)
- [ ] MemoryVLA++ (Extended Journey Version)
  - [ ] Code Release
  - [ ] Model Weights Release
  - [ ] Dataset Upload to HuggingFace


## Contents

This is MemoryVLA based on OpenVLA codebase, **if you need use dexbotic codebase**, please use [MemoryVLA+](https://github.com/shihao1895/MemoryVLA/tree/dexbotic-codebase).

 * [**Model Zoo & Benchmark Results**](#Model-Zoo-&-Benchmark-Results)
 * [**Install**](#Install)
 * [**Evaluation in Libero**](#Evaluation-in-Libero)
 * [**Evaluation in SimplerEnv**](#Evaluation-in-SimplerEnv)
 * [**Training**](#Training)
 * [**Deployment in The Real World**](#deployment-in-the-real-world)
 * [**FAQ**](#FAQ)
 * [**Citation**](#Citation)

## Model Zoo & Benchmark Results

> MemoryVLA means openvla-codebase version, MemoryVLA+ means dexbotic-codebase version.

### Libero

| Model            | Spatial | Object | Goal | Long-10 | Long-90 | Avg. | CKPT & Logs                                                  |
| ---------------- | ------- | ------ | ---- | ------- | ------- | ---- | ------------------------------------------------------------ |
| MemoryVLA        | 98.4    | 98.4   | 96.4 | 93.4    | 95.6    | 96.5 | [🤗 Spa](https://huggingface.co/shihao1895/memvla-libero-spatial), [🤗 Obj](https://huggingface.co/shihao1895/memvla-libero-object), [🤗 Goal](https://huggingface.co/shihao1895/memvla-libero-goal), [🤗 100](https://huggingface.co/shihao1895/memvla-libero-100) |
| MemoryVLA+       | 98.2    | 97.8   | 96.4 | 93.6    | 96.2    | 96.5 | [🤗 Spa](https://huggingface.co/shihao1895/memvla-plus-libero-spatial), [🤗 Obj](https://huggingface.co/shihao1895/memvla-plus-libero-object), [🤗 Goal](https://huggingface.co/shihao1895/memvla-plus-libero-goal), [🤗 100](https://huggingface.co/shihao1895/memvla-plus-libero-100) |
| MemoryVLA+ (mix) | 97.2    | 99.2   | 98.4 | 93.2    | 97.2    | 97.1 | [🤗 HF](https://huggingface.co/shihao1895/memvla-plus-libero-mix) |
| MemoryVLA++      | 99.8    | 100.0  | 98.2 | 96.0    | 97.8    | 98.4 | TBD                                                          |

### SimplerEnv-Bridge

| Model       | Spoon | Carrot | Cube | Eggplant | Avg. | CKPT & Logs                                                  |
| ----------- | ----- | ------ | ---- | -------- | ---- | ------------------------------------------------------------ |
| MemoryVLA   | 75.0  | 75.0   | 37.5 | 100.0    | 71.9 | [🤗 HF](https://huggingface.co/shihao1895/memvla-bridge)      |
| MemoryVLA+  | 100.0 | 66.7   | 70.8 | 100.0    | 84.4 | [🤗 HF](https://huggingface.co/shihao1895/memvla-plus-bridge) |
| MemoryVLA++ | 83.3  | 66.7   | 45.8 | 100.0    | 73.9 | TBD                                                          |

### Mikasa-Robo

| Model       | SGT  | IM   | RC3  | RC5  | RC9  | Avg. | CKPT & Logs                                             |
| ----------- | ---- | ---- | ---- | ---- | ---- | ---- | ------------------------------------------------------- |
| MemoryVLA   | 88   | 24   | 44   | 30   | 20   | 41.2 | [🤗 HF](https://huggingface.co/shihao1895/memvla-mikasa) |
| MemoryVLA++ | 97   | 40   | 50   | 19   | 16   | 44.4 | TBD                                                     |

### Libero-Plus

| Model             | Cam  | Robo | Lang | Light | Backg | Noi  | Layout | Avg. | CKPT & Logs |
| ----------------- | ---- | ---- | ---- | ----- | ----- | ---- | ------ | ---- | ----------- |
| MemoryVLA         | 42.7 | 44.9 | 84.4 | 92.8  | 95.0  | 62.1 | 84.7   | 70.2 | TBD         |
| MemoryVLA++       | 36.4 | 68.9 | 88.7 | 93.8  | 90.6  | 63.5 | 83.8   | 73.1 | TBD         |
| MemoryVLA (SFT)   | 91.4 | 48.6 | 79.4 | 95.2  | 95.3  | 94.0 | 75.7   | 81.9 | TBD         |
| MemoryVLA++ (SFT) | 96.8 | 49.7 | 71.0 | 96.6  | 97.0  | 96.0 | 78.6   | 82.7 | TBD         |

### Calvin

| Model       | 1    | 2    | 3    | 4    | 5    | Avg. | CKPT & Logs |
| ----------- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- |
| MemoryVLA   | 94.8 | 87.4 | 81.4 | 75.9 | 69.4 | 4.09 | TBD         |
| MemoryVLA++ | 95.6 | 90.2 | 85.7 | 81.7 | 76.1 | 4.29 | TBD         |

### Fractal-VM

| Model      | Coke Can | Move Near | Open/Close Drawer | Put In Drawer | Avg. | CKPT & Logs                                                  |
| ---------- | -------- | --------- | ----------------- | ------------- | ---- | ------------------------------------------------------------ |
| MemoryVLA  | 90.7     | 88.0      | 84.7              | 47.2          | 77.7 | [🤗 HF](https://huggingface.co/shihao1895/memvla-fractal)     |
| MemoryVLA+ | 92.0     | 91.7      | 71.8              | -             | -    | [🤗 HF](https://huggingface.co/shihao1895/memvla-plus-fractal) |

### Fractal-VA

| Model      | Coke Can | Move Near | Open/Close Drawer | Put In Drawer | Avg. | CKPT & Logs                                                  |
| ---------- | -------- | --------- | ----------------- | ------------- | ---- | ------------------------------------------------------------ |
| MemoryVLA  | 80.5     | 78.8      | 53.2              | 58.3          | 67.7 | [🤗 HF](https://huggingface.co/shihao1895/memvla-fractal)     |
| MemoryVLA+ | 83.5     | 81.8      | 63.2              | -             | -    | [🤗 HF](https://huggingface.co/shihao1895/memvla-plus-fractal) |

### Maniskill2

| Model      | Pick Cube | Stack Cube | Pick Single YCB | Pick Single EGAD | Pick Clutter YCB | Avg. | CKPT & Logs                                                  |
| ---------- | --------- | ---------- | --------------- | ---------------- | ---------------- | ---- | ------------------------------------------------------------ |
| MemoryVLA+ | 85        | 75         | 60              | 85               | 45               | 70   | [🤗 HF](https://huggingface.co/shihao1895/memvla-plus-maniskill2) |

## Install

The code is built using Python 3.10, and we use PyTorch == 2.2.0 and CUDA == 12.1 (It may run with lower versions, but we have not tested it).

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and setting up an environment:
```bash
conda create --name memvla python=3.10
conda activate memvla

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
conda install -c nvidia cuda-nvcc=12.1 cuda-toolkit=12.1 -y
```
If you need to use the traning code, please also install the [Flash Attention](https://github.com/Dao-AILab/flash-attention), we use flash-attn==2.5.5:

```bash
# Install Flash Attention 2.5.5, this is an example for pytorch2.2-cuda12.1
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Next, clone our repo and install the required packages:

```bash
git clone https://github.com/shihao1895/MemoryVLA
cd MemoryVLA
pip install -e .
```
If you are using an NVIDIA Hopper GPU (e.g., H20) and encounter the error  
“Floating point exception (core dumped)”, try reinstalling the specific cuBLAS version below:

```bash
# Fix for NVIDIA H20: "Floating point exception (core dumped)"
pip install nvidia-cublas-cu12==12.4.5.8
```

## Evaluation in Libero

We also provide evaluation interfaces and scripts based on [LIBERO](https://libero-project.github.io/intro.html).

1. Please follow the installation guide in the [LIBERO Repo](https://github.com/Lifelong-Robot-Learning/LIBERO) to set up the simulation environment, and make sure to place the repo under: `./third_libs/LIBERO`

2. Evaluation Example.

   ```bash
   # Run evaluation
   bash script/eval/libero/eval_libero.sh
   # Summarize results
   python script/eval/libero/extract_libero_results.py
   ```

   > **NOTE:** The evaluation mechanism here is different from SimplerEnv. The process first loads the model using `develop.py`, then waits for a period before running `evaluation/libero/eval_libero.py` for testing. In addition, since performance may vary across iterations, please evaluate multiple checkpoints and report the best result.

## Evaluation in SimplerEnv

We provide evaluation interfaces and scripts based on [SimplerEnv](https://simpler-env.github.io/).

1. Please follow the installation guide in the [SimplerEnv Repo](https://github.com/simpler-env/SimplerEnv) to set up the simulation environment, and make sure to place the repo under: `./third_libs/SimplerEnv`

2. Evaluation Example.

   ```bash
   # Run evaluation
   bash script/eval/bridge/eval_bridge.sh
   # Summarize results
   python script/eval/bridge/extract_bridge_results.py
   ```

   > **NOTE**: Due to the instability of the SimplerEnv benchmark and diffusion process, the performance scores across different iterations can vary significantly. Please evaluate checkpoints **every 2.5k steps** and report the best result.

## Training

1. Prepare training dataset with [RLDS](https://github.com/google-research/rlds) format:

   - [LIBERO](https://libero-project.github.io/intro.html) (including Spatial, Object, Goal, Long-10, Long-90 suites)
   - Bridge from [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/)
   - Fractal from [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/)

   ```bash
   # Make sure you have git-lfs installed (https://git-lfs.com)
   git lfs install
   # Download the LIBERO dataset (processed, ~22 GB)
   git clone https://huggingface.co/datasets/shihao1895/libero-rlds
   # Download the Bridge dataset (processed, ~157 GB)
   git clone https://huggingface.co/datasets/shihao1895/bridge-rlds
   # Download the Fractal dataset (processed)
   git clone https://huggingface.co/datasets/shihao1895/fractal-rlds
   ```

2. Download pretrained model, we use [OpenVLA Pretrained Model](https://huggingface.co/openvla/openvla-7b-prismatic) for LIBERO training, and [CogACT Pretrained Model](https://huggingface.co/CogACT/CogACT-Large) for Bridge and Fractal training.

   ```bash
   # Download OpenVLA pretrained checkpoint (~30 GB)
   git clone https://huggingface.co/openvla/openvla-7b-prismatic
   
   # Download CogACT pretrained checkpoint (~31 GB)
   git clone https://huggingface.co/CogACT/CogACT-Large
   ```

3. Train the model on different datasets

   Before training, modify several parameters in the corresponding scripts, such as `hf_token`, `wandb_entity`, checkpoint paths, dataset paths, and log directories.

   We train on a single node with 8× NVIDIA A100 GPUs.

   ```bash
   # Train on the Bridge dataset
   bash script/train/bridge/train_bridge.sh
   # Train on the LIBERO-Spatial dataset
   bash script/train/libero/train_libero_spatial.sh
   # Train on the LIBERO-Object dataset
   bash script/train/libero/train_libero_object.sh
   # Train on the LIBERO-Goal dataset
   bash script/train/libero/train_libero_goal.sh
   # Train on the LIBERO-100 dataset
   bash script/train/libero/train_libero_100.sh
   # Train on the Fractal dataset
   bash script/train/fractal/train_fractal.sh
   # Train on real-world data
   bash script/train/real_world/train_real.sh
   ```

   To finetune on your own customized data, please follow the instruction [(rlds_dataset_builder)](https://github.com/kpertsch/rlds_dataset_builder) for converting your data to RLDS format. The actions should be the deltas of end effector ``EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)``. Once your customized data is ready, place the customized data directly under the ``<data_root_dir>/custom_finetuning/1.0.0`` directory. Then set ``vla.data_mix="custom_finetuning"``.

## Deployment in the Real World

To deploy the model on your own robot, first collect corresponding real-world manipulation data (e.g., via teleoperation), and use it to fine-tune the pretrained model.

Next, set up the server and client as shown in [`deploy.py`](deploy.py), and deploy the system on your real robot.

The following command launches the server:
```bash
bash script/eval/real_world/deploy.sh
```

The robot acts as the client, and for each request it must send the following three items to obtain the action chunking result. The field episode_first_frame is a string ('True' or 'False') indicating whether the current frame is the first frame of the episode.

```bash
image = request.files['image']
query = request.form['text']
episode_first_frame = request.form['episode_first_frame']
```

This deployment process follows a similar design to [OpenVLA](https://github.com/openvla/openvla) and [CogACT](https://github.com/microsoft/CogACT).

## FAQ

SimplerEnv and ManiSkill may involve several dependency issues during installation. Below are some common troubleshooting tips based on our experience.

**(1) Vulkan / SAPIEN issues**  
Example errors:
ImportError: libvulkan.so.1: cannot open shared object file: No such file or directory
Some required Vulkan extension is not present. You may not use the renderer to render, however, CPU resources will be still available.

Fix:

```bash
sudo apt install -y libegl1-mesa libgl1-mesa-dev libgles2-mesa-dev
```

and reference:
https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#troubleshooting

> **Note**: Check that the .json files correctly link to the .so file corresponding to your current NVIDIA driver version. Use `nvidia-smi` to check your driver version and locate the correct .so under /usr/lib/x86_64-linux-gnu/.

**(2) OpenGL issues**  
Example errors:
ImportError: libGL.so.1: cannot open shared object file: No such file or directory

Fix:

```bash
sudo apt install -y libgl1 libglib2.0-0 libglx-mesa0 libopengl0 libglu1-mesa mesa-utils
```

**(3) Video recording in SimplerEnv**

```bash
sudo apt install -y ffmpeg
```

(4) **Benchmark Score Fluctuations**

Benchmark scores may fluctuate across training iterations, with particularly large variations observed on SimplerEnv. We therefore recommend **evaluating checkpoints at regular iteration intervals and reporting the best result**. In addition, even minor differences in Conda package versions may lead to variations in the scores.

## Citation

If you find our work helpful in your research, please consider citing [our paper](https://arxiv.org/abs/2508.19236). 

```bibtex
@article{shi2025memoryvla,
  title={MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation},
  author={Shi, Hao and Xie, Bin and Liu, Yingfei and Sun, Lin and Liu, Fengrong and Wang, Tiancai and Zhou, Erjin and Fan, Haoqiang and Zhang, Xiangyu and Huang, Gao},
  journal={arXiv preprint arXiv:2508.19236},
  year={2025}
}

@article{shi2026memoryvla++,
  title={MemoryVLA++: Temporal Modeling via Memory and Imagination in Vision-Language-Action Models},
  author={Shi, Hao and Li, Weiye and Xie, Bin and Wang, Yulin and Zhou, Renping and Wang, Tiancai and Zhang, Xiangyu and Luo, Ping and Huang, Gao},
  journal={arXiv preprint arXiv:2606.09827},
  year={2026}
}

@article{dexbotic,
  title={Dexbotic: Open-Source Vision-Language-Action Toolbox},
  author={Dexbotic Contributors},
  journal={arXiv preprint arXiv:2510.23511},
  year={2025}
}
```

