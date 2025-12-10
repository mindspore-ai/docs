# Supported Features List

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.7.2/docs/vllm_mindspore/docs/source_en/user_guide/supported_features/features_list/features_list.md)

The features supported by vLLM-MindSpore Plugin are consistent with the community version of vLLM. For feature descriptions and usage, please refer to the [vLLM Official Documentation](https://docs.vllm.ai/en/latest/).

The following are the features supported in vLLM-MindSpore Plugin.

| **Features**                          | **vLLM V0** | **vLLM V1** |
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

- √: Feature aligned with the community version of vLLM.
- ×: Currently unsupported; alternative solutions are recommended.
- WIP: Under development or planned for future implementation.

## Feature Description

- LoRA currently only supports the Qwen2.5 vLLM-MindSpore Plugin native model, other models are in the process of adaptation.
- Tool Calling only supports DeepSeek V3 0324 W8A8 model.
- 300I Duo has supported Chunked Prefill, Automatic Prefix Caching and Tensor Parallel，and other features are in the process of adaptation.
