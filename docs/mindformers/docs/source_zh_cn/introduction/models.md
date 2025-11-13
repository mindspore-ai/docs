# 模型库

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.1/docs/mindformers/docs/source_zh_cn/introduction/models.md)

当前MindSpore Transformers支持的模型列表如下：

| 模型名                                                                                                                                           | 支持规格                          |   模型类型   |     模型架构     | 最新支持版本 |
|:----------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------|:--------:|:------------:|:------:|
| [Qwen3](https://gitee.com/mindspore/mindformers/tree/r1.7.0/configs/qwen3)                          | 0.6B/1.7B/4B/8B/14B/32B       |  稠密LLM   |    Mcore     | 1.7.0  |
| [Qwen3-MoE](https://gitee.com/mindspore/mindformers/tree/r1.7.0/configs/qwen3_moe)                  | 30B-A3B/235B-A22B             |  稀疏LLM   |    Mcore     | 1.7.0  |
| [DeepSeek-V3](https://gitee.com/mindspore/mindformers/tree/r1.7.0/research/deepseek3)               | 671B                          |  稀疏LLM   | Mcore/Legacy | 1.7.0  |
| [GLM4.5](https://gitee.com/mindspore/mindformers/tree/r1.7.0/configs/glm4_moe)                      | 106B-A12B/355B-A32B           |  稀疏LLM   |    Mcore     | 1.7.0  |
| [GLM4](https://gitee.com/mindspore/mindformers/tree/r1.7.0/configs/glm4)                            | 9B                            |  稠密LLM   | Mcore/Legacy | 1.7.0  |
| [Llama3.1](https://gitee.com/mindspore/mindformers/tree/r1.7.0/research/llama3_1)                   | 8B/70B                        |  稠密LLM   |    Legacy    | 1.7.0  |
| [Mixtral](https://gitee.com/mindspore/mindformers/tree/r1.7.0/research/mixtral)                     | 8x7B                          |  稀疏LLM   |    Legacy    | 1.7.0  |
| [Qwen2.5](https://gitee.com/mindspore/mindformers/tree/r1.7.0/research/qwen2_5)                     | 0.5B/1.5B/7B/14B/32B/72B      |  稠密LLM   |    Legacy    | 1.7.0  |
| [TeleChat2](https://gitee.com/mindspore/mindformers/tree/r1.7.0/research/telechat2)                 | 7B/35B/115B                   |  稠密LLM   | Mcore/Legacy | 1.7.0  |
| [CodeLlama](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/codellama.md)          | 34B                           |  稠密LLM   |    Legacy    | 1.5.0  |
| [CogVLM2-Image](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/cogvlm2_image.md)  | 19B                           |    MM    |    Legacy    | 1.5.0  |
| [CogVLM2-Video](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/cogvlm2_video.md)  | 13B                           |    MM    |    Legacy    | 1.5.0  |
| [DeepSeek-V2](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/deepseek2)                   | 236B                          |  稀疏LLM   |    Legacy    | 1.5.0  |
| [DeepSeek-Coder-V1.5](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/deepseek1_5)         | 7B                            |  稠密LLM   |    Legacy    | 1.5.0  |
| [DeepSeek-Coder](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/deepseek)                 | 33B                           |  稠密LLM   |    Legacy    | 1.5.0  |
| [GLM3-32K](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/glm32k)                         | 6B                            |  稠密LLM   |    Legacy    | 1.5.0  |
| [GLM3](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/glm3.md)                    | 6B                            |  稠密LLM   |    Legacy    | 1.5.0  |
| [InternLM2](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/internlm2)                     | 7B/20B                        |  稠密LLM   |    Legacy    | 1.5.0  |
| [Llama3.2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama3_2.md)            | 3B                            |  稠密LLM   |    Legacy    | 1.5.0  |
| [Llama3.2-Vision](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/mllama.md)       | 11B                           |    MM    |    Legacy    | 1.5.0  |
| [Llama3](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/llama3)                           | 8B/70B                        |  稠密LLM   |    Legacy    | 1.5.0  |
| [Llama2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama2.md)                | 7B/13B/70B                    |  稠密LLM   |    Legacy    | 1.5.0  |
| [Qwen2](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/qwen2)                             | 0.5B/1.5B/7B/57B/57B-A14B/72B | 稠密/稀疏LLM |    Legacy    | 1.5.0  |
| [Qwen1.5](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/qwen1_5)                         | 7B/14B/72B                    |  稠密LLM   |    Legacy    | 1.5.0  |
| [Qwen-VL](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/qwenvl)                          | 9.6B                          |    MM    |    Legacy    | 1.5.0  |
| [TeleChat](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/telechat)                       | 7B/12B/52B                    |  稠密LLM   |    Legacy    | 1.5.0  |
| [Whisper](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/whisper.md)              | 1.5B                          |    MM    |    Legacy    | 1.5.0  |
| [Yi](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/yi)                                   | 6B/34B                        |  稠密LLM   |    Legacy    | 1.5.0  |
| [YiZhao](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/yizhao)                           | 12B                           |  稠密LLM   |    Legacy    | 1.5.0  |
| [Baichuan2](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2/baichuan2.md)        | 7B/13B                        |  稠密LLM   |    Legacy    | 1.3.2  |
| [GLM2](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm2.md)                    | 6B                            |  稠密LLM   |    Legacy    | 1.3.2  |
| [GPT2](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/gpt2.md)                    | 124M/13B                      |  稠密LLM   |    Legacy    | 1.3.2  |
| [InternLM](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/internlm/internlm.md)           | 7B/20B                        |  稠密LLM   |    Legacy    | 1.3.2  |
| [Qwen](https://gitee.com/mindspore/mindformers/blob/r1.3.0/research/qwen/qwen.md)                       | 7B/14B                        |  稠密LLM   |    Legacy    | 1.3.2  |
| [CodeGeex2](https://gitee.com/mindspore/mindformers/blob/r1.1.0/docs/model_cards/codegeex2.md)          | 6B                            |  稠密LLM   |    Legacy    | 1.1.0  |
| [WizardCoder](https://gitee.com/mindspore/mindformers/blob/r1.1.0/research/wizardcoder/wizardcoder.md)  | 15B                           |  稠密LLM   |    Legacy    | 1.1.0  |
| [Baichuan](https://gitee.com/mindspore/mindformers/blob/r1.0/research/baichuan/baichuan.md)             | 7B/13B                        |  稠密LLM   |    Legacy    |  1.0   |
| [Blip2](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/blip2.md)                    | 8.1B                          |    MM    |    Legacy    |  1.0   |
| [Bloom](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/bloom.md)                    | 560M/7.1B/65B/176B            |  稠密LLM   |    Legacy    |  1.0   |
| [Clip](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/clip.md)                      | 149M/428M                     |    MM    |    Legacy    |  1.0   |
| [CodeGeex](https://gitee.com/mindspore/mindformers/blob/r1.0/research/codegeex/codegeex.md)             | 13B                           |  稠密LLM   |    Legacy    |  1.0   |
| [GLM](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/glm.md)                        | 6B                            |  稠密LLM   |    Legacy    |  1.0   |
| [iFlytekSpark](https://gitee.com/mindspore/mindformers/blob/r1.0/research/iflytekspark/iflytekspark.md) | 13B                           |  稠密LLM   |    Legacy    |  1.0   |
| [Llama](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/llama.md)                    | 7B/13B                        |  稠密LLM   |    Legacy    |  1.0   |
| [MAE](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/mae.md)                        | 86M                           |    MM    |    Legacy    |  1.0   |
| [Mengzi3](https://gitee.com/mindspore/mindformers/blob/r1.0/research/mengzi3/mengzi3.md)                | 13B                           |  稠密LLM   |    Legacy    |  1.0   |
| [PanguAlpha](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/pangualpha.md)          | 2.6B/13B                      |  稠密LLM   |    Legacy    |  1.0   |
| [SAM](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/sam.md)                        | 91M/308M/636M                 |    MM    |    Legacy    |  1.0   |
| [Skywork](https://gitee.com/mindspore/mindformers/blob/r1.0/research/skywork/skywork.md)                | 13B                           |  稠密LLM   |    Legacy    |  1.0   |
| [Swin](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/swin.md)                      | 88M                           |    MM    |    Legacy    |  1.0   |
| [T5](https://gitee.com/mindspore/mindformers/blob/r1.0/docs/model_cards/t5.md)                          | 14M/60M                       |  稠密LLM   |    Legacy    |  1.0   |
| [VisualGLM](https://gitee.com/mindspore/mindformers/blob/r1.0/research/visualglm/visualglm.md)          | 6B                            |    MM    |    Legacy    |  1.0   |
| [Ziya](https://gitee.com/mindspore/mindformers/blob/r1.0/research/ziya/ziya.md)                         | 13B                           |  稠密LLM   |    Legacy    |  1.0   |
| [Bert](https://gitee.com/mindspore/mindformers/blob/r0.8/docs/model_cards/bert.md)                      | 4M/110M                       |  稠密LLM   |    Legacy    |  0.8   |

*注：**LLM** 表示大语言模型（Large Language Model）；**MM** 表示多模态（Multi-Modal）*
