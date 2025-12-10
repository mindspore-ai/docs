# Supported Model List

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.7.2/docs/vllm_mindspore/docs/source_en/user_guide/supported_models/models_list/models_list.md)

| Model | Status | Backend Supported | Hardware Supported | Model Download Link |
|-------| ---- |  ---- | --------- | ---- |
| DeepSeek-V3 |   Supported | MindFormers | Atlas 800I A2 | [DeepSeek-V3](https://modelers.cn/models/MindSpore-Lab/DeepSeek-V3) |
| DeepSeek-R1 |   Supported | MindFormers | Atlas 800I A2 | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| DeepSeek-R1 W8A8 |   Supported | MindFormers | Atlas 800I A2 | [DeepSeek-R1-W8A8](https://modelers.cn/models/MindSpore-Lab/DeepSeek-R1-0528-A8W8) |
| DeepSeek-R1 W8A4 |   Supported | MindFormers | Atlas 800I A2 | [DeepSeek-R1-W8A4](https://modelers.cn/models/MindSpore-Lab/R1-0528-A8W4) |
| Telechat2 | Supported | MindFormers | Atlas 800I A2 | [TeleChat2-7B-32K](https://www.modelscope.cn/models/TeleAI/TeleChat2-7B-32K), [TeleChat2-35B-32K](https://www.modelscope.cn/models/TeleAI/TeleChat2-35B-32K) |
| GLM-4.5 | Supported | MindFormers | Atlas 800I A2 | [GLM-4.5](https://huggingface.co/zai-org/GLM-4.5), [GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air) |
| GLM-4 | Supported | MindFormers | Atlas 800I A2 | [GLM-4-9B-0414](https://huggingface.co/zai-org/GLM-4-9B-0414)、[GLM-4-32B-0414](https://huggingface.co/zai-org/GLM-4-32B-0414) |
| Qwen2.5 | Supported | Native, MindFormers | Atlas 800I A2, Atlas 300I Duo(Testing) | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct), [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct), [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| Qwen3 | Supported | Native, MindFormers | Atlas 800I A2, Atlas 300I Duo | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [Qwen3-14B](https://modelers.cn/models/MindSpore-Lab/Qwen3-14B), [Qwen3-32B](https://modelers.cn/models/MindSpore-Lab/Qwen3-32B) |
| Qwen3-235B-A22B | Supported | Native, MindFormers | Atlas 800I A2 | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) |
| Qwen3-30B-A3B | Testing | Native, MindFormers | Atlas 800I A2 | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Qwen2.5-VL | Supported | Native  | Atlas 800I A2 | [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [Qwen2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct), [Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) |
| QwQ-32B | Testing | Native, MindFormers | Atlas 800I A2 | [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B)     |
| Llama3.1 | Testing | Native | Atlas 800I A2 | [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct), [Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct)  |
| Llama3.2 | Testing | Native | Atlas 800I A2 | [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)   |

## Model Description

1. User can refer to [Environment Variable List](../../environment_variables/environment_variables.md), and set the model backend by environment variable `VLLM_MS_MODEL_BACKEND`.
2. The native model backend currently supports the Qwen2.5, Qwen2.5VL, Qwen3 and Llama series models; the MindSpore Transformers backend supports Qwen, DeepSeek, TeleChat and GLM series models.
3. 300I Duo has supported Qwen3 model, and other models are in the process of adaptation.
