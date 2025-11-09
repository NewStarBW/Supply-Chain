# 基于强化学习的车辆路径问题求解器

本项目使用PyTorch实现了一个基于强化学习的车辆路径问题（VRP）求解器，特别关注带时间窗的场景（VRPTW）。该模型采用了基于注意力机制的Encoder-Decoder架构，并结合了REINFORCE算法和自批判基线（Self-Critic Baseline）进行训练。

## 主要特性

- **Encoder-Decoder 架构**: 使用Transformer Encoder提取节点间的复杂关系，并由一个自回归的Decoder生成车辆路径。
- **多智能体方法**: 每辆车被视为一个独立的智能体，通过共享的全局信息和各自的局部信息进行决策。
- **路径记录器 (Route Recorders)**: 引入了基于GRU的局部和全局记录器，使智能体能够记忆历史路径信息并进行有效的多智能体通信。
- **动态惩罚权重**: 实现了`penalty_weight_schedule`，允许在训练过程中动态调整时间窗惩罚的权重。这使得模型在训练初期专注于满足约束（高惩罚权重），在后期则更侧重于优化路径长度（低惩罚权重）。
- **灵活的配置**: 所有关键参数（如问题规模、模型维度、训练参数）都通过 `config.yaml` 文件进行管理，易于调整和实验。
- **断点续训**: 训练过程可以随时中断，并从上次保存的检查点自动恢复。
- **评估与可视化**: `evaluate.py` 脚本提供了与传统求解器（如OR-Tools）进行性能对比的功能，并能将生成的路径可视化。

## 项目结构

```
Supply_Chain/
├── Algorithm/              # 其他算法（OR-Tools, LKH3等）的参考实现
├── benchmarks/             # VRP问题的基准数据集
├── Solomon-dataset-main/   # VRPTW 经典数据集
├── common/                 # 环境、问题定义和实用工具
│   ├── env.py              # VRP环境，处理状态转移和奖励计算
│   └── problem.py          # VRP问题定义
├── model/                  # PyTorch模型定义
│   ├── policy.py           # 策略网络（Encoder-Decoder）
│   ├── critic.py           # 价值网络（基线）
│   └── recorder.py         # 局部和全局路径记录器
├── training/               # 训练相关模块
│   ├── trainer.py          # 核心训练逻辑，包括rollout和参数更新
│   └── checkpoints/        # 模型检查点保存目录
├── config.yaml             # 项目配置文件
├── main.py                 # 训练主入口
├── evaluate.py             # 评估和可视化脚本
├── requirements.txt        # Python依赖
└── README.md               # 本文档
```

## 安装与环境

建议使用虚拟环境以隔离项目依赖。

```bash
# 1. 创建并激活虚拟环境 (以Linux/macOS为例)
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt
```

## 使用指南

### 1. 配置

所有实验设置均在 `config.yaml` 中定义。

- **问题规模**:
  ```yaml
  problem_presets:
    small:
      num_customers: 20
      num_vehicles: 2
    medium:
      name: "medium"
      num_customers: 50
      num_vehicles: 3 
    # ... 其他规模
  ```
- **训练参数**:
  ```yaml
  training_params:
    num_epochs: 100
    batch_size: 128
    learning_rate: 0.0001
    # ...
  ```
- **惩罚权重调度器**:（为解决后期模型不能减小路径长度的问题，实测无效）
  ```yaml
  training_params:
    use_penalty_weight: true
    penalty_weight_schedule:
      enabled: true
      start_epoch: 72
      end_epoch: 100
      start_weight: 1.0
      end_weight: 0.25
  ```
- **解码时的迟到偏置，为避免“优先满足时间窗”的偏向**:
  ```yaml
  decoder_lateness_bias: 5.0   # gamma，0 表示关闭（短期微调时关闭解码偏置）
  time_norm_scale: 1.0        # 车辆状态中当前时间的归一化尺度（或使用各实例最大窗结束时间）
  ```

### 2. 训练模型

运行 `main.py` 启动训练。脚本会根据 `config.yaml` 中的设置，自动加载最新的检查点（如果存在）并继续训练。

```bash
python main.py
```

### 3. 评估模型

使用 `evaluate.py` 来评估已训练模型的性能，并与OR-Tools等基线方法进行比较。

```bash
# 评估 "small" 规模的模型
python evaluate.py --size small
```
评估脚本会输出路径长度、时间惩罚等关键指标，并生成路径可视化图片，保存在 `evaluation_plots/` 目录下。

