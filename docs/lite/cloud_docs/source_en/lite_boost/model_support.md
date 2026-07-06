# Model Support List

The following table lists the models currently supported by LiteBoost and their feature support status.

| Model | Hardware | Parallel | Attention | Quantization | Fused Operator | Notes |
|-------|----------|----------|-----------|--------------|----------------|-------|
| [Wan2.1-T2V-1.3B](https://atomgit.com/mindspore/mindspore-lite/blob/r2.10/mindspore-lite/lite_boost/python/model/wan2_1/README.md) | Atlas 300I Duo Inference Card<br>Atlas 800I A2 Inference Server | USP (CP) | NPU Flash Attention<br>(Flash Attention 3→2→`npu_prompt_flash_attention`) | Not supported | Not supported | RoPE rewrite (float32 real-valued arithmetic + cache)<br>Supports VACE variant |
| [Wan2.2-TI2V-5B](https://atomgit.com/mindspore/mindspore-lite/blob/r2.10/mindspore-lite/lite_boost/python/model/wan2_2/README.md) | Atlas 300I Duo Inference Card<br>Atlas 800I A2 Inference Server | USP (CP) + DP (temporal tiling) | NPU Flash Attention<br>(Flash Attention 3→2→`npu_prompt_flash_attention`) | Not supported | Not supported | RoPE rewrite (float32 real-valued arithmetic + cache)<br>VAE DP temporal tiling for encode/decode |

**Column descriptions:**

- **Model**: Model name, linked to the corresponding README in the LiteBoost source tree.
- **Hardware**: Supported Ascend hardware platforms.
- **Parallel**: Parallelism strategies applied by `ParallelManager`. USP (CP) = Ulysses Sequence Parallel (Context Parallel) for DiT; DP = Data Parallel temporal tiling for VAE.
- **Attention**: Attention implementation replacement. The auto-fallback chain is Flash Attention 3 → Flash Attention 2 → `npu_prompt_flash_attention`.
- **Quantization**: Whether quantization is supported.
- **Fused Operator**: Whether C++ fused operators (registered via `TORCH_LIBRARY` and invoking CANN `aclnn` interfaces) are used. RoPE rewrite is a Python-layer optimization and is not classified as a fused operator.
- **Notes**: Additional optimization details.
