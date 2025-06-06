# 模型支持列表

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_zh_cn/user_guide/supported_models/models_list/models_list.md)

| 模型 | 是否支持 | 模型下载链接 | 模型后端 |
|-------| --------- | ---- | ---- |
| Qwen2.5 |  √ | [Qwen2.5-7B](https://modelers.cn/models/AI-Research/Qwen2.5-7B)、[Qwen2.5-32B](https://modelers.cn/models/AI-Research/Qwen2.5-32B) 等 | MINDFORMER_MODELS |
| Qwen3 |   √ | [Qwen3-8B](https://modelers.cn/models/MindSpore-Lab/Qwen3-8B)、[Qwen3-32B](https://modelers.cn/models/MindSpore-Lab/Qwen3-32B) 等 | MINDFORMER_MODELS |
| DeepSeek V3 |   √ | [DeepSeek-V3](https://modelers.cn/models/MindSpore-Lab/DeepSeek-V3) 等 | MINDFORMER_MODELS |
| DeepSeek R1 |   √ | [DeepSeek-R1](https://modelers.cn/models/MindSpore-Lab/DeepSeek-R1)、[Deepseek-R1-W8A8](https://modelers.cn/models/MindSpore-Lab/DeepSeek-r1-w8a8) 等 | MINDFORMER_MODELS |

其中，“模型后端”指模型的来源是来自于MindSpore Transformers和vLLM MindSpore原生模型，使用环境变量`vLLM_MODEL_BACKEND`进行指定：

- 模型来源为MindSpore Transformers时，则取值为`MINDFORMER_MODELS`；
- 模型来源为vLLM MindSpore时，则取值为`NATIVE_MODELS`；

该值默认`NATIVE_MODELS`，当需要更改模型后端时，使用如下命令：

```bash
export vLLM_MODEL_BACKEND=MINDFORMER_MODELS
```
