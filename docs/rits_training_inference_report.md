# RitS 训练、推理条件与速度报告

## 1. 结论摘要

RitS 是一个基于连续 flow matching 的分子过渡态三维坐标生成模型。模型输入为反应物和产物的连接关系（可带显式立体化学信息），输出过渡态原子坐标；推理时通过 ODE 时间积分从高斯坐标先验逐步得到结果。

仓库给出的推荐使用条件是 Python 3.10 以上、CUDA GPU 和 16--24 GB 显存。README 中作者记录的测试机为 Ubuntu 24.04、NVIDIA RTX 3090（24 GB）和 AMD Ryzen 9 5950X。本文另外给出了一组在当前工作区实测的速度：NVIDIA RTX PRO 6000 Blackwell（约 96 GiB 显存）上，16 个反应、每个生成 1 个样本时：

- 16 个 ODE 步、batch size 16：模型采样核心约 **1.56 s/批**，约 **10.25 个过渡态/s**；包含采样循环开销约 **1.65 s/批**，约 **9.72 个过渡态/s**。
- 25 个 ODE 步、batch size 16：模型采样核心约 **2.06 s/批**，约 **7.76 个过渡态/s**；包含采样循环开销约 **2.15 s/批**，约 **7.46 个过渡态/s**。
- 同一测试中 batch size 1、16 步的采样循环约 **16.06 s/16 个过渡态**，约 **1.00 个过渡态/s**。因此批处理对吞吐影响显著。
- 历史高并发批量运行中，2 个反应各生成 100 个样本、使用 batch100 的日志速率约 **0.23 个反应任务/s**，折算约 **23 个过渡态/s**；这是目前仓库中保留的最接近大批量吞吐的实测记录。
- 现场复测 90000 原子预算时，实际 89976 个原子的一批在 16 步 ODE 下耗时 **63.906 s**（完整循环 **65.392 s**），峰值已分配显存 **47.623 GiB**；这组结果比此前约 50 s 的经验值慢，当前应以现场复测值为准。

以上速度是当前硬件和一组 18--32 原子反应上的实测值，不应直接视为所有分子规模、GPU 或批大小下的固定性能。模型加载、反应解析和最终文件写出未计入“模型采样核心”时间。

## 2. 模型与数据

### 2.1 模型结构和采样方法

`scripts/conf/rits.yaml` 中的 RitS 配置包含以下主要设置：

| 项目 | 设置 |
| --- | --- |
| 动力学模型 | `megav3ts` |
| 网络层数 | 10 |
| 注意力头数 | 4 |
| 不变节点特征 | 256 |
| 不变边特征 | 64 |
| 向量特征 | 64 |
| 距离特征尺寸 | 16 |
| 坐标变量 | 连续 flow matching，预测 velocity |
| 坐标先验 | Gaussian |
| 最优传输 | rigid，配合质心自由约束 |
| 配置默认积分步数 | 25 |
| 推理噪声 | 0 |
| EMA | 开启 |

模型在预处理时将反应物和产物的连接信息合并为图，并构造全连接的分子内边。`prune_edges: False` 表示默认不剪枝；分子原子数增加时，边数和显存占用会明显上升。

项目介绍称，完整 RitS 权重使用约 200 万个过渡态反应训练，覆盖 CHNOSFP 元素、更大的分子体系以及中性和带电反应。仓库当前提供 `data/rits.ckpt`；完整的大规模 GFN2-xTB 数据集按 README 说明暂未随仓库提供。另有 `data/ts1x_rits.ckpt` 和 Transition1x 配置用于基准训练/评估。

### 2.2 输入和数据约束

推理输入支持：

- 带原子映射、显式氢的反应 SMARTS/SMILES，格式为 `reactants>>products`；
- 反应物和产物 XYZ 文件。XYZ 输入需要 Open Babel 推断键连接。

反应物和产物必须满足：

- 原子数相同；
- 原子种类和原子映射能够建立一一对应；
- 若需要立体化学控制，应保留映射、手性和 E/Z 信息，并按需使用 `--kekulize --add_stereo`。

输出为 XYZ 坐标文件。生成多个样本时，程序会把样本写入同一个 XYZ 文件或按队列目录写出。

## 3. 训练条件

### 3.1 软件环境

仓库的最低要求和主要固定依赖如下：

| 项目 | 要求/版本 |
| --- | --- |
| 操作系统 | README 测试环境为 Ubuntu 24.04 LTS |
| Python | >= 3.10，推荐 Conda/Mamba |
| PyTorch | 2.7.0 |
| PyTorch Geometric | 2.6.1 |
| Lightning | 2.5.1.post0 |
| RDKit | 2025.3.2（环境实测显示为 2025.03.2） |
| CUDA | 需要与 PyTorch、PyG 二进制包匹配 |
| 日志 | W&B；配置默认 `online` |

`requirements.txt` 中的 PyG wheel 链接面向 PyTorch 2.7.0 + CUDA 12.6。当前工作区的 `rits` 环境实际报告为 PyTorch 2.7.0+cu128、CUDA 12.8；本次推理已正常运行，但部署到其他机器时应重新确认 PyTorch、CUDA 和 PyG wheel 的组合。

### 3.2 RitS 大规模数据配置

`scripts/conf/rits.yaml` 的默认训练设置为：

| 项目 | 默认值 |
| --- | --- |
| 数据目录 | `data/rits_dataset/processed` |
| 数据加载器 | `midi` |
| 训练 batch size | 150 |
| 验证/推理 batch size | 150 |
| GPU 数量 | 1 |
| 训练轮数 | 1000 epochs |
| 坐标增强旋转 | 关闭 |
| 坐标缩放 | 1.0 |
| TS 比例 | 1.0，即全部使用过渡态目标 |
| 优化器 | AdamW |
| 学习率 | `1e-4` |
| 权重衰减 | `1e-12` |
| AMSGrad | 开启 |
| 学习率调度 | 从 `1e-6` 线性 warmup 到 `1e-4`，10000 steps |
| 梯度裁剪 | 1.0 |
| 验证频率 | 每 5 个 epoch |
| 检查点 | 每 500 个训练 step 保存一次，并保存最优检查点 |

训练入口 `scripts/train.py` 使用 Lightning 的 GPU accelerator；多 GPU 时使用 DDP。训练支持三种模式：从头训练、从已有权重开始但重新建立优化器/调度器的 fine-tune，以及恢复已有 checkpoint 的 optimizer、scheduler、epoch 和 step 状态。

训练 Transition1x 时使用 `scripts/conf/ts1x_rits.yaml`，主要区别是：数据目录为 `data/ts1x`，标准 DataLoader，训练/推理 batch size 为 200，默认 500 epochs，学习率 `5e-4`，权重衰减 `0.01`，并使用 plateau 学习率调度器。Transition1x 预处理脚本默认按原始数据 80%/10%/10% 划分训练、验证和测试集，并保留原始样本及其反向/增强样本的配对划分。

### 3.3 训练资源建议

训练显存会随最大分子原子数、batch size、全连接边数和 GPU 数量变化。建议：

- 先使用 1 张 CUDA GPU 验证数据、配置和 checkpoint 加载；
- 混合分子规模时优先降低 batch size，必要时采用数据加载器的动态/自适应批处理；
- 多 GPU 训练使用 `train.gpus=N`，但应单独确认每张 GPU 的显存和有效 batch size；
- 记录 PyTorch、CUDA、PyG、RDKit 和 Lightning 版本，避免二进制包不匹配；
- 本仓库没有保存大规模训练的总墙钟时间或稳定的 samples/s 记录，因此本文不对训练总时长作定量承诺。

## 4. 推理条件与流程

单机推理的基本流程是：

1. 读取配置和 checkpoint；
2. 将 SMARTS/SMILES 或 XYZ 转换为 PyG 图；
3. 按原子映射对齐反应物和产物，并构造反应物/产物键矩阵；
4. 在 GPU 上建立高斯坐标先验；
5. 使用 `model.sample(..., timesteps=N)` 做 N 步 flow/ODE 积分；
6. 将生成坐标转换为 XYZ。

常用命令如下：

```bash
PYTHONPATH=src python scripts/sample_transition_state.py \
  --reaction_file data/ts_inference_reactions.smi \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --output output.xyz \
  --n_samples 1 \
  --batch_size 32 \
  --num_steps 25 \
  --device cuda:0
```

当前工作区的 `sample_transition_state.py` 命令行默认值为 16 步，而配置文件和仓库示例显式使用 25 步。为了复现实验，建议始终显式传入 `--num_steps`；不要仅根据配置文件推断脚本实际使用的步数。

对于多个反应或多个随机样本，建议使用 `scripts/generate_reaction_queue.py`，它支持按图数量 batch，或者通过 `--max_atoms_per_batch` 按总原子数动态打包。后者更适合分子大小差异大的队列。显存不足时，优先降低 batch size、总原子数预算或采样步数。

## 5. 推理速度实测

### 5.1 测试条件

本次实测使用：

| 项目 | 设置 |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| GPU 数量 | 2 张可见；单次测试只使用 `cuda:1` |
| 单卡显存 | 97887 MiB，约 96 GiB |
| 驱动 | 595.58.03 |
| Python 环境 | Conda 环境 `rits` |
| PyTorch | 2.7.0+cu128 |
| CUDA runtime | 12.8 |
| PyG | 2.6.1 |
| Lightning | 2.5.1.post0 |
| checkpoint | `data/rits.ckpt` |
| 输入 | `data/ts_inference_reactions.smi` 中 16 个反应 |
| 分子规模 | 每个图 18--32 个原子，合计 380 个原子 |
| 推理模式 | `torch.no_grad()`，CUDA 起止同步 |
| 输出写盘 | 使用 `--no_write`，不计文件写出 |

计时由 `scripts/generate_reaction_queue.py` 完成。`sampling_wall_seconds` 是各批次模型采样时间之和；`loop_wall_seconds` 从进入采样循环到结束，包含 batch 准备、数据搬运、先验生成和结果转换等循环开销，但不包括 checkpoint 加载和反应队列构建。

### 5.2 结果

| ODE 步数 | Batch size | 采样核心时间 | 核心吞吐 | 循环墙钟时间 | 循环吞吐 | 峰值已分配显存 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 15.945 s / 16 个 | 1.00 个/s | 16.059 s / 16 个 | 1.00 个/s | 约 0.15 GiB |
| 16 | 16 | 1.561 s / 16 个 | 10.25 个/s | 1.646 s / 16 个 | 9.72 个/s | 约 0.217 GiB |
| 25 | 16 | 2.063 s / 16 个 | 7.76 个/s | 2.146 s / 16 个 | 7.46 个/s | 约 0.217 GiB |

在这组输入上，从 16 步增加到 25 步，核心采样时间约增加 32.5%，核心吞吐从约 10.25 个/s 降到约 7.76 个/s。理论上步数比例为 25/16=1.5625，实际增幅小于该比例，原因包括固定开销、GPU kernel 调度和批量矩阵运算的效率变化。

### 5.2.1 90000 原子预算现场复测

为复核此前“90000 原子一批约 50 s”的高并发结论，使用同一套 16 个反应输入，在 `cuda:1` 上运行了一个真实的动态原子预算 batch：

| 项目 | 现场设置/结果 |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition，约 96 GiB |
| checkpoint | `data/rits.ckpt` |
| ODE 步数 | 16 |
| 反应数 | 16 |
| 请求样本数 | 300/反应，共 4800 个队列图；仅执行第一个 batch |
| 原子批预算 | 90000 |
| 实际 batch | 3789 个图，89976 个原子 |
| 边数 | 2110350 |
| 模型采样核心时间 | 63.906 s |
| 完整采样循环时间 | 65.392 s |
| 核心吞吐 | 约 59.3 个过渡态/s |
| 循环吞吐 | 约 58.0 个过渡态/s |
| 峰值已分配显存 | 47.623 GiB |

本次计时使用 `torch.cuda.synchronize()` 对采样前后进行同步，且使用 `torch.no_grad()`；模型加载、反应解析和输出写盘不计入核心采样时间。该结果说明在当前代码、当前 checkpoint、当前 GPU 和 16 步设置下，90000 原子一批更接近 **64 s**，而不是 50 s。此前约 50 s 的结论没有对应的原始命令和计时日志，可能来自不同的 GPU、代码版本、batch 组成或计时边界，暂作为历史经验值保留。

已有并行生成日志还记录了 2026-08-03 的历史运行。结果和折算如下：

| 结果目录 | 工作量 | 日志末尾速率 | 折算过渡态吞吐 | 结果文件证据 |
| --- | ---: | ---: | ---: | --- |
| `results/rits_requested_ts` | 16 个反应 × 1 个样本 = 16 个 | 约 1.03 reaction job/s | 约 1.03 个/s | 16/16 成功 |
| `results/rits_requested_ts_seeds_0_99` | 2 个反应 × 100 个样本 = 200 个 | 约 0.02 reaction job/s | 约 2 个/s | 每个反应 100 个 seed 结果 |
| `results/rits_requested_ts_seeds_0_99_batch100` | 2 个反应 × 100 个样本 = 200 个 | 约 0.23 reaction job/s | 约 23 个/s | 每个反应 100 个 seed 结果 |

其中 `reaction job` 表示“一个反应的一组样本”任务；例如 0.23 reaction job/s 乘以每个任务的 100 个样本，约得到 23 个过渡态/s。目录名为 batch100 的运行相对于前一组运行，按日志末尾速率约提升 11.5 倍；前一组的完整命令行没有保存，脚本默认 batch size 是 32，因此不能把 32 视为已核验的运行参数。两次运行均完成 2/2 个任务；日志没有记录失败任务。

需要特别区分：当前仓库实际保存的是 16 个反应 × 1 个样本和 2 个反应 × 100 个样本，没有保存“16 个反应各几百样本”的完整日志。因此，如果将 batch100 的约 23 个过渡态/s 作为同等条件下的线性估计，则 16 个反应批量生成的规模估算为：

| 每个反应样本数 | 总过渡态数（16 个反应） | 预计时间 @ 23 个/s |
| ---: | ---: | ---: |
| 100 | 1,600 | 约 1.2 min |
| 300 | 4,800 | 约 3.5 min |
| 500 | 8,000 | 约 5.8 min |

上述三项是基于已保存 batch100 结果的线性外推，不应替代原始 16×几百任务的完整墙钟记录。原始日志没有完整保存 GPU 数量、实际命令行、ODE 步数和 batch 参数，因此报告不把外推值写成新的实测值。

### 5.3 速度影响因素

实际吞吐主要受以下因素影响：

- **ODE 步数**：每增加一步，通常都要增加一次动力学网络调用；
- **batch size**：对多个反应或多个样本进行批处理能显著摊薄 kernel 和 Python 调度开销；
- **原子数和边数**：模型使用分子内全连接图，边数近似随原子数平方增长；
- **显存容量**：大分子或高 batch 可能 OOM，需要按总原子数设置批预算；
- **输入准备**：SMARTS 解析、Open Babel 键推断、RDKit 立体化学处理和 XYZ 写盘不属于模型核心吞吐；
- **GPU 并发**：多 GPU 可以按反应队列并行，但单卡吞吐不应与多卡总吞吐混为一谈。

## 6. 复现实验

在已经安装 `rits` Conda 环境的情况下，复现 16 步、batch 16 的核心测速：

```bash
conda run -n rits env PYTHONPATH=src:scripts \
  python scripts/generate_reaction_queue.py \
  --reactions data/ts_inference_reactions.smi \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --n_samples 1 \
  --batch_size 16 \
  --num_steps 16 \
  --device cuda:1 \
  --no_write
```

复现 90000 原子预算的现场测试：

```bash
conda run -n rits env PYTHONPATH=src:scripts \
  python scripts/generate_reaction_queue.py \
  --reactions data/ts_inference_reactions.smi \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --n_samples 300 \
  --max_atoms_per_batch 90000 \
  --num_steps 16 \
  --device cuda:1 \
  --max_batches 1 \
  --no_write
```

复现 25 步结果时，将 `--num_steps 16` 改为 `--num_steps 25`。建议至少重复 3 次，丢弃首次 checkpoint 加载/首次 CUDA kernel 的启动影响，并报告中位数；本文表格是单次基准运行，适合描述当前仓库状态，不替代完整性能基准测试。

## 7. 参考文件

- [README.md](../README.md)：安装、数据、训练和基本推理命令。
- [scripts/conf/rits.yaml](../scripts/conf/rits.yaml)：大规模 RitS 模型和训练默认配置。
- [scripts/conf/ts1x_rits.yaml](../scripts/conf/ts1x_rits.yaml)：Transition1x 配置。
- [scripts/train.py](../scripts/train.py)：训练入口、Lightning GPU 设置和 checkpoint 模式。
- [scripts/sample_transition_state.py](../scripts/sample_transition_state.py)：单机推理入口和输入约束。
- [scripts/generate_reaction_queue.py](../scripts/generate_reaction_queue.py)：批量队列推理和 CUDA 同步测速入口。
- [data_processing/prepare_ts1x_for_training.py](../data_processing/prepare_ts1x_for_training.py)：Transition1x 预处理和数据划分。
