# 模型支持库

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/introduction/models.md)

本页为 MindSpore Transformers 的统一「模型支持库」。表格中的 **实现形态** 列标注每个模型当前支持的运行模式：

- **动态图（PyNative）**：通过 `--mode 1` 启动，逐算子下发、即时执行，便于调试与开发。当前动态图已支持的模型见下表标注，对应实现位于 `mindformers/models/*/modeling_*_pynative.py`。
- **静态图（GRAPH_MODE）**：图编译后整图执行，相关说明详见[静态图实现](../static_graph/introduction/overview.md)。

动态图当前已支持：**DeepSeek-V3**（MoE + MLA + MTP）、**Qwen3**（Dense）。其余既有模型为静态图实现。

## 模型列表

| 模型名 | 支持规格 | 模型类型 | 模型架构 | 实现形态 | 最新支持版本 |
|:---|:---|:---:|:---:|:---:|:---:|
| [DeepSeek-V3](https://atomgit.com/mindspore/mindformers/tree/master/mindformers/models/deepseek3) | 671B | 稀疏LLM | Mcore/Legacy | **动态图**/静态图 | 1.7.0、在研版本 |
| [Qwen3](https://atomgit.com/mindspore/mindformers/tree/master/mindformers/models/qwen3) | 0.6B/1.7B/4B/8B/14B/32B | 稠密LLM | Mcore | **动态图**/静态图 | 1.7.0、在研版本 |
| [Qwen3-MoE](https://atomgit.com/mindspore/mindformers/tree/master/configs/qwen3_moe) | 30B-A3B/235B-A22B | 稀疏LLM | Mcore | 静态图 | 1.7.0、在研版本 |
| [GLM4.5](https://atomgit.com/mindspore/mindformers/tree/master/configs/glm4_moe) | 106B-A12B/355B-A32B | 稀疏LLM | Mcore | 静态图 | 1.7.0、在研版本 |
| [GLM4](https://atomgit.com/mindspore/mindformers/tree/master/configs/glm4) | 9B | 稠密LLM | Mcore/Legacy | 静态图 | 1.7.0、在研版本 |
| [Qwen2.5](https://atomgit.com/mindspore/mindformers/tree/master/research/qwen2_5) | 0.5B/1.5B/7B/14B/32B/72B | 稠密LLM | Legacy | 静态图 | 1.7.0、在研版本 |
| [TeleChat2](https://atomgit.com/mindspore/mindformers/tree/master/research/telechat2) | 7B/35B/115B | 稠密LLM | Mcore | 静态图 | 1.7.0、在研版本 |
| [Llama3.1](https://atomgit.com/mindspore/mindformers/tree/r1.7.0/research/llama3_1) | 8B/70B | 稠密LLM | Legacy | 静态图 | 1.7.0 |
| [Mixtral](https://atomgit.com/mindspore/mindformers/tree/r1.7.0/research/mixtral) | 8x7B | 稀疏LLM | Legacy | 静态图 | 1.7.0 |
| [CodeLlama](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/codellama.md) | 34B | 稠密LLM | Legacy | 静态图 | 1.5.0 |
| [CogVLM2-Image](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/cogvlm2_image.md) | 19B | MM | Legacy | 静态图 | 1.5.0 |
| [CogVLM2-Video](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/cogvlm2_video.md) | 13B | MM | Legacy | 静态图 | 1.5.0 |
| [DeepSeek-V2](https://atomgit.com/mindspore/mindformers/tree/r1.5.0/research/deepseek2) | 236B | 稀疏LLM | Legacy | 静态图 | 1.5.0 |
| [DeepSeek-Coder-V1.5](https://atomgit.com/mindspore/mindformers/tree/r1.5.0/research/deepseek1_5) | 7B | 稠密LLM | Legacy | 静态图 | 1.5.0 |
| [DeepSeek-Coder](https://atomgit.com/mindspore/mindformers/tree/r1.5.0/research/deepseek) | 33B | 稠密LLM | Legacy | 静态图 | 1.5.0 |
| [GLM3-32K](https://atomgit.com/mindspore/mindformers/tree/r1.5.0/research/glm32k) | 6B | 稠密LLM | Legacy | 静态图 | 1.5.0 |
| [GLM3](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/glm3.md) | 6B | 稠密LLM | Legacy | 静态图 | 1.5.0 |
| [InternLM2](https://atomgit.com/mindspore/mindformers/tree/r1.5.0/research/internlm2) | 7B/20B | 稠密LLM | Legacy | 静态图 | 1.5.0 |
| [Llama3.2](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama3_2.md) | 3B | 稠密LLM | Legacy | 静态图 | 1.5.0 |
| [Llama3.2-Vision](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/mllama.md) | 11B | MM | Legacy | 静态图 | 1.5.0 |
| [Llama3](https://atomgit.com/mindspore/mindformers/tree/r1.5.0/research/llama3) | 8B/70B | 稠密LLM | Legacy | 静态图 | 1.5.0 |
| [Qwen2](https://atomgit.com/mindspore/mindformers/tree/r1.5.0/research/qwen2) | 0.5B/1.5B/7B/57B/57B-A14B/72B | 稠密/稀疏LLM | Legacy | 静态图 | 1.5.0 |
| [Qwen1.5](https://atomgit.com/mindspore/mindformers/tree/r1.5.0/research/qwen1_5) | 7B/14B/72B | 稠密LLM | Legacy | 静态图 | 1.5.0 |
| [Qwen-VL](https://atomgit.com/mindspore/mindformers/tree/r1.5.0/research/qwenvl) | 9.6B | MM | Legacy | 静态图 | 1.5.0 |
| [TeleChat](https://atomgit.com/mindspore/mindformers/tree/r1.5.0/research/telechat) | 7B/12B/52B | 稠密LLM | Legacy | 静态图 | 1.5.0 |
| [Whisper](https://atomgit.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/whisper.md) | 1.5B | MM | Legacy | 静态图 | 1.5.0 |
| [Yi](https://atomgit.com/mindspore/mindformers/tree/r1.5.0/research/yi) | 6B/34B | 稠密LLM | Legacy | 静态图 | 1.5.0 |
| [YiZhao](https://atomgit.com/mindspore/mindformers/tree/r1.5.0/research/yizhao) | 12B | 稠密LLM | Legacy | 静态图 | 1.5.0 |
| [Llama2](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md) | 7B/13B/70B | 稠密LLM | Legacy | 静态图 | 1.3.2 |
| [Baichuan2](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/research/baichuan2/baichuan2.md) | 7B/13B | 稠密LLM | Legacy | 静态图 | 1.3.2 |
| [GLM2](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm2.md) | 6B | 稠密LLM | Legacy | 静态图 | 1.3.2 |
| [GPT2](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/gpt2.md) | 124M/13B | 稠密LLM | Legacy | 静态图 | 1.3.2 |
| [InternLM](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/research/internlm/internlm.md) | 7B/20B | 稠密LLM | Legacy | 静态图 | 1.3.2 |
| [Qwen](https://atomgit.com/mindspore/mindformers/blob/r1.3.0/research/qwen/qwen.md) | 7B/14B | 稠密LLM | Legacy | 静态图 | 1.3.2 |
| [CodeGeex2](https://atomgit.com/mindspore/mindformers/blob/r1.1.0/docs/model_cards/codegeex2.md) | 6B | 稠密LLM | Legacy | 静态图 | 1.1.0 |
| [WizardCoder](https://atomgit.com/mindspore/mindformers/blob/r1.1.0/research/wizardcoder/wizardcoder.md) | 15B | 稠密LLM | Legacy | 静态图 | 1.1.0 |
| [Baichuan](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/baichuan/baichuan.md) | 7B/13B | 稠密LLM | Legacy | 静态图 | 1.0 |
| [Blip2](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/blip2.md) | 8.1B | MM | Legacy | 静态图 | 1.0 |
| [Bloom](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/bloom.md) | 560M/7.1B/65B/176B | 稠密LLM | Legacy | 静态图 | 1.0 |
| [Clip](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/clip.md) | 149M/428M | MM | Legacy | 静态图 | 1.0 |
| [CodeGeex](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/codegeex/codegeex.md) | 13B | 稠密LLM | Legacy | 静态图 | 1.0 |
| [GLM](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/glm.md) | 6B | 稠密LLM | Legacy | 静态图 | 1.0 |
| [iFlytekSpark](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/iflytekspark/iflytekspark.md) | 13B | 稠密LLM | Legacy | 静态图 | 1.0 |
| [Llama](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/llama.md) | 7B/13B | 稠密LLM | Legacy | 静态图 | 1.0 |
| [MAE](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/mae.md) | 86M | MM | Legacy | 静态图 | 1.0 |
| [Mengzi3](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/mengzi3/mengzi3.md) | 13B | 稠密LLM | Legacy | 静态图 | 1.0 |
| [PanguAlpha](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/pangualpha.md) | 2.6B/13B | 稠密LLM | Legacy | 静态图 | 1.0 |
| [SAM](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/sam.md) | 91M/308M/636M | MM | Legacy | 静态图 | 1.0 |
| [Skywork](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/skywork/skywork.md) | 13B | 稠密LLM | Legacy | 静态图 | 1.0 |
| [Swin](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/swin.md) | 88M | MM | Legacy | 静态图 | 1.0 |
| [T5](https://atomgit.com/mindspore/mindformers/blob/r1.0/docs/model_cards/t5.md) | 14M/60M | 稠密LLM | Legacy | 静态图 | 1.0 |
| [VisualGLM](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/visualglm/visualglm.md) | 6B | MM | Legacy | 静态图 | 1.0 |
| [Ziya](https://atomgit.com/mindspore/mindformers/blob/r1.0/research/ziya/ziya.md) | 13B | 稠密LLM | Legacy | 静态图 | 1.0 |
| [Bert](https://atomgit.com/mindspore/mindformers/blob/r0.8/docs/model_cards/bert.md) | 4M/110M | 稠密LLM | Legacy | 静态图 | 0.8 |

*注：**LLM** 表示大语言模型（Large Language Model）；**MM** 表示多模态（Multi-Modal）。**实现形态** 列中标注「动态图/静态图」的模型同时支持两种运行模式；动态图启动方式见 [快速开始](../quick_start/quick_start.md)，静态图实现详见[静态图实现](../static_graph/introduction/overview.md)。*
