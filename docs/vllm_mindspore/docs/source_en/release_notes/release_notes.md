# Release Notes

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.7.0/docs/vllm_mindspore/docs/source_en/release_notes/release_notes.md)

## vLLM-MindSpore Plugin 0.3.0 Release Notes

The vLLM-MindSpore Plugin 0.3.0 version is compatible with vLLM 0.8.3. Below are the new features and models supported in this release.

### New Features

- **Architecture Adaptation**: Supports both vLLM V0 and V1 architectures. Users can switch the architectures using `VLLM_USE_V1`.
- **Service Features**: Supports Chunked Prefill, Automatic Prefix Caching, Async Output, and Reasoning Outputs. The V0 architecture also supports Multi-Step Scheduler and DeepSeek MTP features. For detailed descriptions, refer to the [Feature Support List](../user_guide/supported_features/features_list/features_list.md).
- **Quantization Support**: Supports GPTQ quantization and SmoothQuant quantization. For detailed descriptions, refer to [Quantization Methods](../user_guide/supported_features/quantization/quantization.md).
- **Parallel Strategies**: In the V1 architecture, Tensor Parallel, Data Parallel, and Expert Parallel are supported. For detailed descriptions, refer to [Multi-Machine Parallel Inference](../getting_started/tutorials/deepseek_parallel/deepseek_r1_671b_w8a8_dp4_tp4_ep4.md).
- **Debugging Tools**: Adapted vLLM's profiling tool for performance data collection and model IR graph saving via the MindSpore backend, facilitating model debugging and optimization. Adapted vLLM's benchmark tool for performance testing. For detailed descriptions, refer to [Debugging Methods](../user_guide/supported_features/profiling/profiling.md) and [Performance Testing](../user_guide/supported_features/benchmark/benchmark.md).

### New Models

- DeepSeek Series Models:
    - [Supported] DeepSeek-V3, DeepSeek-R1, DeepSeek-R1 W8A8 quantized models.
- Qwen2.5 Series Models:
    - [Supported] Qwen2.5: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B.
    - [Testing] Qwen2.5-VL: 3B, 7B, 32B, 72B.
- Qwen3 Series Models:
    - [Supported] Qwen3: 32B; Qwen3-MOE: 235B-A22B.
    - [Testing] Qwen3: 0.6B, 1.7B, 4B, 8B, 14B; Qwen3-MOE: Qwen3-30B-A3.
- QwQ Series Models:
    - [Testing] QwQ: 32B.
- Llama Series Models:
    - [Testing] Llama3.1: 8B, 70B, 405B.
    - [Testing] Llama3.2: 1B, 3B.

### Contributors

Thanks to the following contributors for their efforts:

alien_0119, candyhong, can-gaa-hou, ccsszz, cs123abc, dayschan, Erpim, fary86, hangangqiang, horcam, huandong, huzhikun, i-robot, jiahaochen666, JingweiHuang, lijiakun, liu lili, lvhaoyu, lvhaoyu1, moran, nashturing, one_east, panshaowu, pengjingyou, r1chardf1d0, tongl, TrHan, tronzhang, TronZhang, twc, uh, w00521005, wangpingan2, WanYidong, WeiCheng Tan, wusimin, yangminghai, yyyyrf, zhaizhiqiang, zhangxuetong, zhang_xu_hao1230, zhanzhan1, zichun_ye, zlq2020

Contributions to the project in any form are welcome!
