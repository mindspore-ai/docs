# Quick Start

## Overview

This article introduces the quick start guide for LiteBoost, including the configuration and usage of multi-card parallel inference.

## Multi-Card Parallel Inference

LiteBoost provides a one-line API to enable multi-card parallel inference for supported models. Two parallelism strategies are applied automatically:

- **Ulysses Sequence Parallel (USP)** for DiT models — sequence-dimension parallelism via `all_to_all` communication around attention layers.
- **Data Parallel (DP) temporal tiling** for VAE models — temporal-dimension slicing with overlap, distributed across devices.

### Model Background

This guide uses **Wan2.2-TI2V-5B** as an example. Wan2.2-TI2V-5B is a text-image-to-video generation model that takes a text prompt and a reference image as input and generates a video. The model consists of three main components:

- **T5 text encoder**: Encodes the text prompt into embeddings.
- **DiT (Diffusion Transformer)**: Iteratively denoises latent representations conditioned on text and image embeddings.
- **VAE**: Encodes the reference image into latent space and decodes the denoised latents back into video frames.

The inputs required for inference are:

- **Text prompt**: A natural language description of the desired video content.
- **Reference image**: An image that provides visual context (e.g., the starting appearance of a subject).

### Prerequisites

- LiteBoost has been installed. See [Build and Install](build_and_install.md).
- Ascend CANN and HCCL are properly configured.
- Multiple NPU devices are available.

### Usage

The typical workflow for multi-card parallel inference is:

1. Call `initialize_usp()` to initialize the HCCL distributed environment.
2. Load the model (e.g., `WanTI2V`).
3. Wrap the model with `ParallelManager` to enable parallel inference.

The code below marks each section as either a **LiteBoost addition** or part of the **original workflow**:

```python
import os
import torch
import torch_npu

# --- Original workflow: Load the model ---
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

# --- LiteBoost addition: Multi-card parallel inference ---
if world_size > 1:
    from lite_boost.parallel import initialize_usp, ParallelManager
    initialize_usp()
    ParallelManager(pipe)

# --- Original workflow: Run inference ---
device = torch.device(f"npu:{local_rank}")
pipe.model.to(device)

from PIL import Image
img = Image.open("input.jpg").convert("RGB")
video = pipe.generate(
    "Your prompt here",
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

# --- Original workflow: Save the generated video (rank 0 only) ---
if rank == 0:
    from wan.utils.utils import save_video
    tag = f"{world_size}card"
    save_video(tensor=video[None], save_file=f"ti2v-5B_{tag}.mp4",
               fps=cfg.sample_fps, nrow=1, normalize=True,
               value_range=(-1, 1))
```

### Launching Multi-Card Inference

Use `torchrun` to launch the script across multiple NPU devices:

```bash
# 2-card inference
ASCEND_RT_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 your_script.py
```

### Expected Output

After a successful run, the generated video is saved as `ti2v-5B_2card.mp4` in the current directory.

### Environment Variables

The following environment variables control the distributed environment and are read by `initialize_usp()`:

| Variable | Description | Default |
|----------|-------------|---------|
| `RANK` | Global rank of the current process | `0` |
| `WORLD_SIZE` | Total number of distributed processes | `1` |
| `LOCAL_RANK` | Local rank on the current node | `0` |
| `MASTER_ADDR` | IP address of the master node | `127.0.0.1` |
| `MASTER_PORT` | Port of the master node | `29502` |
| `NUM_THREADS` | Number of CPU threads per process | `24` |

> When using `torchrun`, `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, and `MASTER_PORT` are set automatically.

### How It Works

When `ParallelManager(pipe)` is called on a pipeline object, it automatically:

1. Detects the model type.
2. For the DiT model: replaces `flash_attention` with an NPU-compatible version, patches each attention block's `forward` with `usp_attn_forward` (inserting `all_to_all` communication pairs), and replaces the model's `forward` with `usp_dit_forward` (entry sequence split + exit `all_gather`).
3. For the VAE model: replaces `vae.encode` and `vae.decode` with DP temporal tiling versions that split the video along the temporal dimension into overlapping chunks, distribute them across devices, and gather results.

The model is modified in-place and returned as-is, so all existing attributes and methods (`.to`, `.cpu`, `.eval`, etc.) continue to work normally.

### Reference

The [generate.py](https://github.com/Wan-Video/Wan2.2/blob/main/generate.py) file from the Wan2.2 repository is the original inference script without LiteBoost. Users can compare it with the LiteBoost code above to understand the differences and modify it accordingly to enable multi-card parallel inference.

## Fusion Operators

Fusion operators currently supported by lite_boost:

| Operator Name                | Hardware       | Operator Interface |
|------------------------------|----------------|------------------------------|
| [RainFusionAttention](https://atomgit.com/mindspore/mindspore-lite/blob/master/mindspore-lite/lite_boost/docs/ops/RainFusionAttention.md) | Atlas 800I A2 | lite_boost.ops.rain_fusion_attention<br/>lite_boost.ops.sparse_attention |
