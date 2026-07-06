# 模型支持列表

下表列出了 LiteBoost 当前支持的模型及其特性支持情况。

| 模型 | 硬件 | 并行 | Attention | 量化 | 融合算子 | 备注 |
|------|------|------|-----------|------|----------|------|
| [Wan2.1-T2V-1.3B](https://atomgit.com/mindspore/mindspore-lite/blob/r2.10/mindspore-lite/lite_boost/python/model/wan2_1/README.md) | Atlas 300I Duo 推理卡<br>Atlas 800I A2 推理服务器 | USP (CP) | NPU Flash Attention<br>(Flash Attention 3→2→`npu_prompt_flash_attention`) | 不支持 | 不支持 | RoPE改写（float32实数运算+缓存）<br>支持VACE变体 |
| [Wan2.2-TI2V-5B](https://atomgit.com/mindspore/mindspore-lite/blob/r2.10/mindspore-lite/lite_boost/python/model/wan2_2/README.md) | Atlas 300I Duo 推理卡<br>Atlas 800I A2 推理服务器 | USP (CP) + DP（时间切片） | NPU Flash Attention<br>(Flash Attention 3→2→`npu_prompt_flash_attention`) | 不支持 | 不支持 | RoPE改写（float32实数运算+缓存）<br>VAE DP时间切片用于encode/decode |

**列说明：**

- **模型**：模型名称，超链接到 LiteBoost 源码中对应的 README。
- **硬件**：支持的昇腾硬件平台。
- **并行**：`ParallelManager` 应用的并行策略。USP (CP) = Ulysses序列并行（上下文并行）用于DiT；DP = 数据并行时间切片用于VAE。
- **Attention**：Attention 实现替换，自动回退链为 Flash Attention 3 → Flash Attention 2 → `npu_prompt_flash_attention`。
- **量化**：是否支持量化。
- **融合算子**：是否使用 C++ 融合算子（通过 `TORCH_LIBRARY` 注册并调用 CANN `aclnn` 接口）。RoPE改写属于 Python 层优化，不属于融合算子范畴。
- **备注**：其他优化细节。
