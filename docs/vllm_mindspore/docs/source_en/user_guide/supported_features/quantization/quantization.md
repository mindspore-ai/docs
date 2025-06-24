# Quantization Methods

[![View Source](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/user_guide/supported_features/quantization/quantization.md)

This document introduces model quantization and quantized inference methods. Quantization reduces inference resource with minor cost of precision, while improving inference performance to enable deployment on more devices. With the large scale of LLMs, post-training quantization has become the mainstream approach for model quantization. For details, refer to [Post-Training Quantization Introduction](https://gitee.com/mindspore/golden-stick/blob/master/mindspore_gs/ptq/README.md).

In this document, the [Creating Quantized Models](#creating-quantized-models) section introduces post-training quantization steps using [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) as an example. A the [Quantized Model Inference](#quantized-model-inference) section explains how to perform inference with quantized models.

## Creating Quantized Models

We use the [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) network as an example to introduce W8A8 quantization with the OutlierSuppressionLite algorithm.

### Quantizing Networks with MindSpore Golden Stick

We employ [MindSpore Golden Stick's PTQ algorithm](https://gitee.com/mindspore/golden-stick/blob/master/mindspore_gs/ptq/ptq/README.md) for quantization of DeepSeek-R1. For detailed methods, refer to [DeepSeekR1-OutlierSuppressionLite Quantization Example](https://gitee.com/mindspore/golden-stick/blob/master/example/deepseekv3/a8w8-osl/readme.md).

### Downloading Quantized Weights

We have uploaded the quantized DeepSeek-R1 to [ModelArts Community](https://modelers.cn): [MindSpore-Lab/DeepSeek-R1-W8A8](https://modelers.cn/models/MindSpore-Lab/DeepSeek-R1-W8A8). Refer to the [ModelArts Community documentation](https://modelers.cn/docs/en/openmind-hub-client/0.9/basic_tutorial/download.html) to download the weights locally.

## Quantized Model Inference

After obtaining the DeepSeek-R1 W8A8 weights, ensure they are stored in the relative path `DeepSeek-R1-W8A8`.

### Offline Inference

Refer to the [Installation Guide](../../../getting_started/installation/installation.md) to set up the vLLM MindSpore environment. Once ready, use the following Python code for offline inference:

```python
import vllm_mindspore  # Add this line at the top of the script
from vllm import LLM, SamplingParams

# Sample prompts
prompts = [
    "I am",
    "Today is",
    "Llama is"
]

# Create sampling parameters
sampling_params = SamplingParams(temperature=0.0, top_p=0.95)

# Initialize LLM
llm = LLM(model="DeepSeek-R1-W8A8")
# Generate text
outputs = llm.generate(prompts, sampling_params)
# Print results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

Successful execution will yield inference results like:

```text
Prompt: 'I am', Generated text: ' trying to create a virtual environment for my Python project, but I am encountering some'
Prompt: 'Today is', Generated text: ' the 100th day of school. To celebrate, the teacher has'
Prompt: 'Llama is', Generated text: ' a 100% natural, biodegradable, and compostable alternative'
```
