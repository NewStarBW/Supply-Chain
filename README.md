# 多智能体强化学习求解车辆路径问题 (MVRPSTW)

本项目是论文 [《A Multi-Agent Reinforcement Learning Method With Route Recorders for Vehicle Routing in Supply Chain Management》](https://ieeexplore.ieee.org/abstract/document/9714823/) 的一个简化复现。

该项目使用多智能体强化学习（MARL）来解决带软时间窗的多车辆路径问题（MVRPSTW），旨在同时优化路径长度和时间窗惩罚。

## 项目结构

```
.
├── common/               # 环境和问题定义
│   ├── env.py
│   └── problem.py
├── model/                # 神经网络模型组件
│   ├── decoder.py
│   ├── encoder.py
│   ├── policy.py
│   └── recorder.py
├── training/             # 训练逻辑和检查点
│   ├── checkpoints/
│   └── trainer.py
├── config.yaml           # 配置文件
├── main.py               # 主程序入口
└── requirements.txt      # Python包依赖
```

## 环境与依赖

本项目基于 Python 3.9 开发，需要安装以下依赖包。

### 1. 创建虚拟环境 (推荐)

为了避免与您系统中的其他Python项目产生冲突，建议您创建一个独立的虚拟环境。

```bash
# 创建一个名为 .venv 的虚拟环境
python -m venv .venv
```

### 2. 激活虚拟环境

- **Windows**:
  ```bash
  .\.venv\Scripts\activate
  ```
- **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  ```

### 3. 安装依赖包

我们已经将所有需要的包记录在 `requirements.txt` 文件中。请运行以下命令进行安装：

```bash
pip install -r requirements.txt
```

## 如何使用

### 1. 配置训练参数

打开 `config.yaml` 文件，您可以根据需要调整模型和训练参数：

- `num_customers`: 客户数量 (例如: 20, 50, 100)
- `num_vehicles`: 车辆数量 (例如: 2, 3, 4, 5)
- `num_epochs`: 训练的总轮数
- `batch_size`: 每批次训练的问题实例数量

### 2. 开始训练

配置完成后，直接运行主程序即可开始训练：

```bash
python main.py
```

训练过程中，模型检查点（checkpoint）会自动保存在 `training/checkpoints/` 目录下。您可以随时使用 `Ctrl+C` 中断训练，下次重新运行时，程序会自动加载最新的检查点并从中断处继续。

## 核心技术

- **模型框架**: 基于编码器-解码器 (Encoder-Decoder) 架构。
- **编码器**: 采用多头自注意力机制 (Multi-Head Attention) 提取客户节点间的关系。
- **路径记录器**: 论文的核心创新，使用GRU单元作为“局部”和“全局”路径记录器，为智能体（车辆）提供历史信息，并实现车辆间的通信。
- **训练方法**: 采用带“自批判”基线 (Self-Critic Baseline) 的策略梯度强化学习算法 (REINFORCE)。

