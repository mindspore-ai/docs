# Supported Model List

[![View Source](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/user_guide/supported_models/models_list/models_list.md)

| Model | Supported | Download Link | Backend |
|-------| --------- | ------------- | ------- |
| Qwen2.5 |  √ | [Qwen2.5-7B](https://modelers.cn/models/AI-Research/Qwen2.5-7B), [Qwen2.5-32B](https://modelers.cn/models/AI-Research/Qwen2.5-32B), etc. | MindSpore Transformers  |
| Qwen3 |   √ |[Qwen3-32B](https://modelers.cn/models/MindSpore-Lab/Qwen3-32B), etc. | MindSpore Transformers   |
| DeepSeek V3 |   √ | [DeepSeek-V3](https://modelers.cn/models/MindSpore-Lab/DeepSeek-V3), etc. | MindSpore Transformers   |
| DeepSeek R1 |   √ | [DeepSeek-R1](https://modelers.cn/models/MindSpore-Lab/DeepSeek-R1), [Deepseek-R1-W8A8](https://modelers.cn/models/MindSpore-Lab/DeepSeek-r1-w8a8), etc. | MindSpore Transformers   |

The "Backend" refers to the source of the model, which can be either from MindSpore Transformers or vLLM MindSpore native models. It is specified using the environment variable `vLLM_MODEL_BACKEND`:

- If the model source is MindSpore Transformers, the value is `MindFormers`;
- If the model source is vLLM MindSpore, user does not need to set the environment variable.

User can change the model backend by the following command:

```bash
export vLLM_MODEL_BACKEND=MindFormers
```
