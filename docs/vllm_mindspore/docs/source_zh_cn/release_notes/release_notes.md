# Release Notes

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.0/docs/vllm_mindspore/docs/source_zh_cn/release_notes/release_notes.md)

## vLLM-MindSpore插件 0.3.0 Release Notes

vLLM MindSpore插件0.3.0版本，配套vLLM 0.8.3版本。以下为此版本支持的关键新功能和模型。

### 新特性

- **架构适配**：架构适配vLLM V0与V1架构，用户可通过`VLLM_USE_V1`进行架构切换；
- **服务特性**：支持Chunked Prefill、Automatic Prefix Caching、Async output、Reasoning Outputs等特性；其中V0架构中也支持Multi-step scheduler、DeepSeek MTP特性。详细描述请参考[特性支持列表](../user_guide/supported_features/features_list/features_list.md)；
- **量化支持**：支持GPTQ量化与SmoothQuant量化功能；详细描述请参考[量化方法](../user_guide/supported_features/quantization/quantization.md)；
- **并行策略**：V1架构中，支持张量并行（Tensor Parallel）、数据并行（Data Parallel）、专家并行（Expert Parallel）；详细描述请参考[多机并行推理](../getting_started/tutorials/deepseek_parallel/deepseek_r1_671b_w8a8_dp4_tp4_ep4.md)；
- **调试工具**：适配使用vLLM的profile工具，通过MindSpore后端进行性能数据采集、模型IR图保存，便于用户进行模型的调试与调优；适配使用vLLM的benchmark工具进行性能测试。详细描述请参考[调试方法](../user_guide/supported_features/profiling/profiling.md)与[性能测试](../user_guide/supported_features/benchmark/benchmark.md)；

### 新模型

- DeepSeek 系列模型：
    - [已支持] DeepSeek-V3、DeepSeek-R1、DeepSeek-R1 W8A8量化模型；
- Qwen2.5 系列模型：
    - [已支持] Qwen2.5：0.5B、1.5B、3B、7B、14B、32B、72B；
    - [测试中] Qwen2.5-VL：3B、7B、32B、72B；
- Qwen3 系列模型：
    - [已支持] Qwen3：32B；Qwen3-MOE：235B-A22B；
    - [测试中] Qwen3：0.6B、1.7B、4B、8B、14B；Qwen3-MOE：Qwen3-30B-A3
- QwQ 系列模型：
    - [测试中] QwQ：32B
- Llama 系列模型：
    - [测试中] Llama3.1：8B、70B、405B
    - [测试中] Llama3.2：1B、3B

### 贡献者

感谢以下人员做出的贡献：

alien_0119、candyhong、can-gaa-hou、ccsszz、cs123abc、dayschan、Erpim、fary86、hangangqiang、horcham_zhq、huandong、huzhikun、i-robot、jiahaochen666、JingweiHuang、lijiakun、liu lili、lvhaoyu、lvhaoyu1、moran、nashturing、one_east、panshaowu、pengjingyou、r1chardf1d0、tongl、TrHan、tronzhang、TronZhang、twc、uh、w00521005、wangpingan2、WanYidong、WeiCheng Tan、wusimin、yangminghai、yyyyrf、zhaizhiqiang、zhangxuetong、zhang_xu_hao1230、zhanzhan1、zichun_ye、zlq2020

欢迎以任何形式对项目提供贡献！
