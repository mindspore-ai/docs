# Supported Features List

[![View Source](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/user_guide/supported_features/features_list/features_list.md)

The features supported by vLLM MindSpore are consistent with the community version of vLLM. For feature descriptions and usage, please refer to the [vLLM Official Documentation](https://docs.vllm.ai/en/latest/).

The following is the features supported in vLLM MindSpore.

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
| LogProbs                          | ×                  | WIP                |  
| Prompt logProbs                   | ×                  | WIP                |  
| Best of                           | ×                  | ×                  |  
| Beam search                       | ×                  | WIP                |  
| Guided Decoding                   | ×                  | WIP                |  
| Pooling                           | ×                  | ×                  |
| Enc-dec                           | ×                  | ×                  |  

- √：Feature aligned with the community version of vLLM.
- ×：Currently unsupported; alternative solutions are recommended.
- WIP：Under development or planned for future implementation.
