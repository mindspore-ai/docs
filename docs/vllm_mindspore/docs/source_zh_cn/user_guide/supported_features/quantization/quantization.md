# 量化方法

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_zh_cn/user_guide/supported_features/quantization/quantization.md)

本文档将为用户介绍模型量化与量化推理的方法。量化方法通过牺牲部分模型精度的方式，达到降低模型部署时的资源需求的目的，并提升模型部署时的性能，从而允许模型被部署到更多的设备上。由于大语言模型的规模较大，出于成本考虑，训练后量化成为主流模型量化方案，具体可以参考[后量化技术简介](https://gitee.com/mindspore/golden-stick/blob/master/mindspore_gs/ptq/README_CN.md)。

本文档中，[创建量化模型](#创建量化模型)章节，将以[DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)为例，介绍模型后量化的步骤；[量化模型推理](#量化模型推理)章节，介绍如何使用量化模型进行推理。

## 创建量化模型

以[DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)网络为例，使用OutlierSuppressionLite算法对其进行W8A8量化。

### 使用MindSpore金箍棒量化网络

我们将使用[MindSpore 金箍棒的PTQ算法](https://gitee.com/mindspore/golden-stick/blob/master/mindspore_gs/ptq/ptq/README_CN.md)对DeepSeek-R1网络进行量化，详细方法参考[DeepSeekR1-OutlierSuppressionLite量化样例](https://gitee.com/mindspore/golden-stick/blob/master/example/deepseekv3/a8w8-osl/readme.md)

### 直接下载量化权重

我们已经将量化好的DeepSeek-R1上传到[魔乐社区](https://modelers.cn)：[MindSpore-Lab/DeepSeek-R1-W8A8](https://modelers.cn/models/MindSpore-Lab/DeepSeek-R1-W8A8)，可以参考[魔乐社区文档](https://modelers.cn/docs/zh/openmind-hub-client/0.9/basic_tutorial/download.html)将权重下载到本地。

## 量化模型推理

在上一步中获取到DeepSeek-R1 W8A8量化权重后，保证该权重存放相对路径为`DeepSeek-R1-W8A8`。

### 离线推理

用户可以参考[安装指南](../../../getting_started/installation/installation.md)，进行vLLM MindSpore的环境搭建。环境准备完成后，用户可以使用如下Python代码，进行离线推理服务：

```python
import vllm_mindspore # Add this line on the top of script.
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "I am",
    "Today is",
    "Llama is"
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, top_p=0.95)

# Create a LLM
llm = LLM(model="DeepSeek-R1-W8A8")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}. Generated text: {generated_text!r}")
```

执行成功后，将获得如下推理结果：

```text
Prompt: 'I am'. Generated text: ' trying to create a virtual environment for my Python project, but I am encountering some'
Prompt: 'Today is'. Generated text: ' the 100th day of school. To celebrate, the teacher has'
Prompt: 'Llama is'. Generated text: ' a 100% natural, biodegradable, and compostable alternative'
```
