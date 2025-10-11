# 特性支持列表

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.1/docs/vllm_mindspore/docs/source_zh_cn/user_guide/supported_features/features_list/features_list.md)

vLLM-MindSpore插件支持的特性功能与vLLM社区版本保持一致，特性描述和使用请参考[vLLM官方资料](https://docs.vllm.ai/en/latest/)。

以下是vLLM-MindSpore插件的功能支持状态：

| **功能**                          | **vLLM V0** | **vLLM V1** |  
|-----------------------------------|--------------------|--------------------|  
| Chunked Prefill                   | √                  | √                  |  
| Automatic Prefix Caching          | √                  | √                  |  
| Multi step scheduler              | √                  | ×                  |  
| DeepSeek MTP                      | √                  | WIP                |  
| Async output                      | √                  | √                  |  
| Quantization                      | √                  | √                  |  
| LoRA                              | WIP                | WIP                |  
| Tensor Parallel                   | √                  | √                  |  
| Pipeline Parallel                 | WIP                | WIP                |  
| Expert Parallel                   | ×                  | √                  |  
| Data Parallel                     | ×                  | √                  |  
| Prefill Decode Disaggregation     | ×                  | WIP                |  
| Multi Modality                    | WIP                | WIP                |  
| Prompt adapter                    | ×                  | WIP                |  
| Speculative decoding              | ×                  | WIP                |  
| LogProbs                          | ×                  | √                  |  
| Prompt logProbs                   | ×                  | WIP                |  
| Best of                           | ×                  | ×                  |  
| Beam search                       | ×                  | WIP                |  
| Guided Decoding                   | ×                  | WIP                |  
| Pooling                           | ×                  | ×                  |
| Enc-dec                           | ×                  | ×                  |
| Reasoning Outputs                 | √                  | √                  |
| Tool Calling                      | WIP                | √                  |

- √：功能已与vLLM社区版本能力对齐。
- ×：暂无支持计划，建议使用其他方案代替。
- WIP：功能正在开发中或已列入开发计划中。

## 特性说明

- LoRA目前仅支持Qwen2.5 vLLM-MindSpore插件原生模型，其他模型正在适配中；
- Tool Calling目前已支持DeepSeek V3 0324 W8A8模型。
