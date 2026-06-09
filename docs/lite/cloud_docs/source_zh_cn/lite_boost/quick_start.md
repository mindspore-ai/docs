# 快速使用

## 概述

本文介绍LiteBoost的快速使用方法，包括多卡并行推理的配置与使用。

## 多卡并行推理

LiteBoost提供一行API，即可对支持的模型使能多卡并行推理。根据模型组件，自动应用以下两种并行策略：

- **Ulysses序列并行（USP）** 用于DiT模型 — 通过在注意力层前后插入`all_to_all`通信实现序列维度并行。
- **数据并行（DP）时间切片** 用于VAE模型 — 沿时间维度将视频切分为重叠的帧片段，分发到各卡独立处理，最后收集拼接为完整结果。

### 模型背景

本指南以**Wan2.2-TI2V-5B**为例。Wan2.2-TI2V-5B是一个文本-图片到视频生成模型，接收文本提示词和参考图片作为输入，生成视频。该模型由三个主要组件构成：

- **T5文本编码器**：将文本提示词编码为嵌入向量。
- **DiT（扩散Transformer）**：基于文本和图片嵌入，迭代去噪潜在表示。
- **VAE**：将参考图片编码到潜在空间，并将去噪后的潜在表示解码为视频帧。

推理所需的输入包括：

- **文本提示词**：描述期望视频内容的自然语言文本。
- **参考图片**：提供视觉上下文的图片（例如，主体的初始外观）。

### 前置条件

- 已安装LiteBoost，参见[编译安装](build_and_install.md)。
- 昇腾CANN和HCCL已正确配置。
- 具备多张NPU设备。

### 使用方法

多卡并行推理的典型工作流程为：

1. 调用`initialize_usp()`初始化HCCL分布式环境。
2. 加载模型（如`WanTI2V`）。
3. 使用`ParallelManager`包装模型，使能并行推理。

以下代码标注了每部分属于**LiteBoost新增**还是**原始流程**内容：

```python
import os
import torch
import torch_npu

# --- 原始流程：加载模型 ---
from wan.configs import WAN_CONFIGS
from wan.textimage2video import WanTI2V

local_rank = int(os.getenv("LOCAL_RANK", "0"))
rank = int(os.getenv("RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

cfg = WAN_CONFIGS["ti2v-5B"]

pipe = WanTI2V(
    config=cfg,
    checkpoint_dir="/path/to/Wan2.2-TI2V-5B",
    device_id=local_rank,
    rank=rank,
    t5_fsdp=False,
    dit_fsdp=False,
    use_sp=False,
    t5_cpu=True,
    init_on_cpu=True,
)

# --- LiteBoost新增：多卡并行推理 ---
if world_size > 1:
    from lite_boost.parallel import initialize_usp, ParallelManager
    initialize_usp()
    ParallelManager(pipe)

# --- 原始流程：执行推理 ---
device = torch.device(f"npu:{local_rank}")
pipe.model.to(device)

from PIL import Image
img = Image.open("input.jpg").convert("RGB")
video = pipe.generate(
    "你的提示词",
    img=img,
    size=(832, 480),
    max_area=832 * 480,
    frame_num=81,
    shift=3.0,
    sample_solver="unipc",
    sampling_steps=20,
    guide_scale=5.0,
    seed=42,
    offload_model=False,
)

# --- 原始流程：保存生成的视频（仅rank 0） ---
if rank == 0:
    from wan.utils.utils import save_video
    tag = f"{world_size}card"
    save_video(tensor=video[None], save_file=f"ti2v-5B_{tag}.mp4",
               fps=cfg.sample_fps, nrow=1, normalize=True,
               value_range=(-1, 1))
```

### 启动多卡推理

使用`torchrun`在多张NPU设备上启动脚本：

```bash
# 2卡推理
ASCEND_RT_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 your_script.py
```

### 预期输出

成功运行后，生成的视频将保存为当前目录下的`ti2v-5B_2card.mp4`文件。

### 环境变量

以下环境变量控制分布式环境，由`initialize_usp()`读取：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `RANK` | 当前进程的全局rank | `0` |
| `WORLD_SIZE` | 分布式进程总数 | `1` |
| `LOCAL_RANK` | 当前节点上的本地rank | `0` |
| `MASTER_ADDR` | 主节点IP地址 | `127.0.0.1` |
| `MASTER_PORT` | 主节点端口 | `29502` |
| `NUM_THREADS` | 每个进程的CPU线程数 | `24` |

> 使用`torchrun`时，`RANK`、`WORLD_SIZE`、`LOCAL_RANK`、`MASTER_ADDR`和`MASTER_PORT`会自动设置。

### 工作原理

当对流水线对象调用`ParallelManager(pipe)`时，自动执行以下操作：

1. 检测模型类型。
2. 对于DiT模型：将`flash_attention`替换为NPU兼容版本，将每个注意力块的`forward`补丁替换为`usp_attn_forward`（插入`all_to_all`通信对），并将模型的`forward`替换为`usp_dit_forward`（入口序列切分 + 出口`all_gather`合并）。
3. 对于VAE模型：将`vae.encode`和`vae.decode`替换为DP时间切片版本，沿时间维度将视频切分为重叠的帧片段，分发到各卡独立处理，最后收集拼接为完整结果。

模型在原地修改后原样返回，因此所有已有的属性和方法（`.to`、`.cpu`、`.eval`等）均可正常使用。

### 参考

Wan2.2仓库中的[generate.py](https://github.com/Wan-Video/Wan2.2/blob/main/generate.py)是未使用LiteBoost的原始推理脚本，用户可以将其与上方LiteBoost代码对照阅读差异，并据此修改以使能多卡并行推理。

## 融合算子

当前lite_boost 支持的融合算子列表：

| 算子名称                    | 硬件            | 算子接口 |
|-------------------------|---------------|------------|
| [RainFusionAttention](https://atomgit.com/mindspore/mindspore-lite/blob/master/mindspore-lite/lite_boost/docs/ops/RainFusionAttention.md) | Atlas 800I A2 | lite_boost.ops.rain_fusion_attention<br/>lite_boost.ops.sparse_attention |
