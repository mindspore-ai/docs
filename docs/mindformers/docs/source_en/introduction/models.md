# Models

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.9.0/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/r2.9.0/docs/mindformers/docs/source_en/introduction/models.md)

The following table lists models supported by MindSpore Transformers.

| Model                                                                                                             | Specifications                |    Model Type     | Model Architecture | Latest Version |
|:------------------------------------------------------------------------------------------------------------------|:------------------------------|:-----------------:|:------------------:|:--------------:|
| [TeleChat3](https://atomgit.com/mindspore/mindformers/blob/r1.9.0/configs/telechat3) `🔥HOT`                      | 36B                           |     Dense LLM     |       Mcore        |     1.9.0      |
| [TeleChat3-MoE](https://atomgit.com/mindspore/mindformers/blob/r1.9.0/configs/telechat3_moe) `🔥HOT`              | 105B-A4.7B                    |    Sparse LLM     |       Mcore        |     1.9.0      |
| [Qwen3](https://atomgit.com/mindspore/mindformers/blob/r1.9.0/configs/qwen3) `🔥HOT`                              | 0.6B/1.7B/4B/8B/14B/32B       |     Dense LLM     |       Mcore        |     1.9.0      |
| [Qwen3-MoE](https://atomgit.com/mindspore/mindformers/blob/r1.9.0/configs/qwen3_moe) `🔥HOT`                      | 30B-A3B/235B-A22B             |    Sparse LLM     |       Mcore        |     1.9.0      |
| [DeepSeek-V3](https://atomgit.com/mindspore/mindformers/blob/r1.9.0/research/deepseek3) `🔥HOT`                   | 671B                          |    Sparse LLM     |    Mcore/Legacy    |     1.9.0      |
| [GLM4.5](https://atomgit.com/mindspore/mindformers/blob/r1.9.0/configs/glm4_moe) `🔥HOT`                          | 106B-A12B/355B-A32B           |    Sparse LLM     |       Mcore        |     1.9.0      |
| [GLM4](https://atomgit.com/mindspore/mindformers/blob/r1.9.0/configs/glm4) `🔥HOT`                                | 9B                            |     Dense LLM     |    Mcore/Legacy    |     1.9.0      |
| [Qwen2.5](https://atomgit.com/mindspore/mindformers/blob/r1.9.0/research/qwen2_5) `🔥HOT`                         | 0.5B/1.5B/7B/14B/32B/72B      |     Dense LLM     |       Legacy       |     1.9.0      |
| [TeleChat2](https://atomgit.com/mindspore/mindformers/blob/r1.9.0/research/telechat2) `🔥HOT`                     | 7B/35B/115B                   |     Dense LLM     |    Mcore/Legacy    |     1.9.0      |
| [Llama3.1](https://atomgit.com/mindspore/mindformers/blob/r1.7.0/research/llama3_1) `⚠️EOL`                       | 8B/70B                        |     Dense LLM     |       Legacy       |     1.7.0      |
| [Mixtral](https://atomgit.com/mindspore/mindformers/blob/r1.7.0/research/mixtral) `⚠️EOL`                         | 8x7B                          |    Sparse LLM     |       Legacy       |     1.7.0      |
| [CodeLlama](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/codellama.md) `⚠️EOL`          | 34B                           |     Dense LLM     |       Legacy       |     1.5.0      |
| [CogVLM2-Image](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/cogvlm2_image.md) `⚠️EOL`  | 19B                           |        MM         |       Legacy       |     1.5.0      |
| [CogVLM2-Video](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/cogvlm2_video.md) `⚠️EOL`  | 13B                           |        MM         |       Legacy       |     1.5.0      |
| [DeepSeek-V2](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/deepseek2) `⚠️EOL`                   | 236B                          |    Sparse LLM     |       Legacy       |     1.5.0      |
| [DeepSeek-Coder-V1.5](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/deepseek1_5) `⚠️EOL`         | 7B                            |     Dense LLM     |       Legacy       |     1.5.0      |
| [DeepSeek-Coder](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/deepseek) `⚠️EOL`                 | 33B                           |     Dense LLM     |       Legacy       |     1.5.0      |
| [GLM3-32K](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/glm32k) `⚠️EOL`                         | 6B                            |     Dense LLM     |       Legacy       |     1.5.0      |
| [GLM3](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/glm3.md) `⚠️EOL`                    | 6B                            |     Dense LLM     |       Legacy       |     1.5.0      |
| [InternLM2](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/internlm2) `⚠️EOL`                     | 7B/20B                        |     Dense LLM     |       Legacy       |     1.5.0      |
| [Llama3.2](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama3_2.md) `⚠️EOL`            | 3B                            |     Dense LLM     |       Legacy       |     1.5.0      |
| [Llama3.2-Vision](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/mllama.md) `⚠️EOL`       | 11B                           |        MM         |       Legacy       |     1.5.0      |
| [Llama3](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/llama3) `⚠️EOL`                           | 8B/70B                        |     Dense LLM     |       Legacy       |     1.5.0      |
| [Qwen2](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/qwen2) `⚠️EOL`                             | 0.5B/1.5B/7B/57B/57B-A14B/72B | Dense /Sparse LLM |       Legacy       |     1.5.0      |
| [Qwen1.5](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/qwen1_5) `⚠️EOL`                         | 7B/14B/72B                    |     Dense LLM     |       Legacy       |     1.5.0      |
| [Qwen-VL](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/qwenvl) `⚠️EOL`                          | 9.6B                          |        MM         |       Legacy       |     1.5.0      |
| [TeleChat](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/telechat) `⚠️EOL`                       | 7B/12B/52B                    |     Dense LLM     |       Legacy       |     1.5.0      |
| [Whisper](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/whisper.md) `⚠️EOL`              | 1.5B                          |        MM         |       Legacy       |     1.5.0      |
| [Yi](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/yi) `⚠️EOL`                                   | 6B/34B                        |     Dense LLM     |       Legacy       |     1.5.0      |
| [YiZhao](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/research/yizhao) `⚠️EOL`                           | 12B                           |     Dense LLM     |       Legacy       |     1.5.0      |
| [Llama2](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md) `⚠️EOL`                | 7B/13B/70B                    |     Dense LLM     |       Legacy       |     1.3.2      |
| [Baichuan2](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2/baichuan2.md) `⚠️EOL`        | 7B/13B                        |     Dense LLM     |       Legacy       |     1.3.2      |
| [GLM2](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm2.md) `⚠️EOL`                    | 6B                            |     Dense LLM     |       Legacy       |     1.3.2      |
| [GPT2](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/gpt2.md) `⚠️EOL`                    | 124M/13B                      |     Dense LLM     |       Legacy       |     1.3.2      |
| [InternLM](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/research/internlm/internlm.md) `⚠️EOL`           | 7B/20B                        |     Dense LLM     |       Legacy       |     1.3.2      |
| [Qwen](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/research/qwen/qwen.md) `⚠️EOL`                       | 7B/14B                        |     Dense LLM     |       Legacy       |     1.3.2      |
| [CodeGeex2](https://atomgit.com/mindspore/mindformers/blob/r1.1.0/docs/model_cards/codegeex2.md) `⚠️EOL`          | 6B                            |     Dense LLM     |       Legacy       |     1.1.0      |
| [WizardCoder](https://atomgit.com/mindspore/mindformers/blob/r1.1.0/research/wizardcoder/wizardcoder.md) `⚠️EOL`  | 15B                           |     Dense LLM     |       Legacy       |     1.1.0      |
| [Baichuan](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/baichuan/baichuan.md) `⚠️EOL`             | 7B/13B                        |     Dense LLM     |       Legacy       |      1.0       |
| [Blip2](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/blip2.md) `⚠️EOL`                    | 8.1B                          |        MM         |       Legacy       |      1.0       |
| [Bloom](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/bloom.md) `⚠️EOL`                    | 560M/7.1B/65B/176B            |     Dense LLM     |       Legacy       |      1.0       |
| [Clip](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/clip.md) `⚠️EOL`                      | 149M/428M                     |        MM         |       Legacy       |      1.0       |
| [CodeGeex](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/codegeex/codegeex.md) `⚠️EOL`             | 13B                           |     Dense LLM     |       Legacy       |      1.0       |
| [GLM](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/glm.md) `⚠️EOL`                        | 6B                            |     Dense LLM     |       Legacy       |      1.0       |
| [iFlytekSpark](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/iflytekspark/iflytekspark.md) `⚠️EOL` | 13B                           |     Dense LLM     |       Legacy       |      1.0       |
| [Llama](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/llama.md) `⚠️EOL`                    | 7B/13B                        |     Dense LLM     |       Legacy       |      1.0       |
| [MAE](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/mae.md) `⚠️EOL`                        | 86M                           |        MM         |       Legacy       |      1.0       |
| [Mengzi3](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/mengzi3/mengzi3.md) `⚠️EOL`                | 13B                           |     Dense LLM     |       Legacy       |      1.0       |
| [PanguAlpha](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/pangualpha.md) `⚠️EOL`          | 2.6B/13B                      |     Dense LLM     |       Legacy       |      1.0       |
| [SAM](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/sam.md) `⚠️EOL`                        | 91M/308M/636M                 |        MM         |       Legacy       |      1.0       |
| [Skywork](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/skywork/skywork.md) `⚠️EOL`                | 13B                           |     Dense LLM     |       Legacy       |      1.0       |
| [Swin](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/swin.md) `⚠️EOL`                      | 88M                           |        MM         |       Legacy       |      1.0       |
| [T5](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/t5.md) `⚠️EOL`                          | 14M/60M                       |     Dense LLM     |       Legacy       |      1.0       |
| [VisualGLM](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/visualglm/visualglm.md) `⚠️EOL`          | 6B                            |        MM         |       Legacy       |      1.0       |
| [Ziya](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/ziya/ziya.md) `⚠️EOL`                         | 13B                           |     Dense LLM     |       Legacy       |      1.0       |
| [Bert](https://atomgit.com/mindspore/mindformers/blob/r0.8/docs/model_cards/bert.md) `⚠️EOL`                      | 4M/110M                       |     Dense LLM     |       Legacy       |      0.8       |

&#42; `⚠️EOL` indicates that the model has been offline from the main branch and can be used with the latest supported version (e.g., 1.7.0).