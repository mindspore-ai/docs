# Quantization Methods

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/user_guide/supported_features/quantization/quantization.md)

This document introduces model quantization and quantized inference methods. Quantization reduces inference resources with minor cost of precision, while improving inference performance to enable deployment on more devices. With the large scale of LLMs, post-training quantization has become the mainstream approach for model quantization. For details, refer to [Post-Training Quantization Introduction](https://atomgit.com/mindspore/golden-stick/blob/master/mindspore_gs/ptq/README.md).

In this document, the [Creating Quantized Models](#creating-quantized-models) section introduces post-training quantization steps using [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) as an example. The [Quantized Model Inference](#quantized-model-inference) section explains how to perform inference with quantized models. The [W8A8SC Sparse Quantization Models](#w8a8sc-sparse-quantization-models) section introduces the principles, hardware support, and data format requirements of W8A8SC sparse quantization technology.

## Creating Quantized Models

We use the [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) network as an example to introduce W8A8 quantization with the OutlierSuppressionLite algorithm. This chapter requires the MindSpore Golden Stick module. Please refer to [here](https://www.mindspore.cn/golden_stick/docs/en/master/index.html) for details about this module.

### Quantizing Networks with MindSpore Golden Stick

We employ [MindSpore Golden Stick's PTQ algorithm](https://atomgit.com/mindspore/golden-stick/blob/master/mindspore_gs/ptq/ptq/README.md) for quantization of DeepSeek-R1. For detailed methods, refer to [DeepSeekR1-OutlierSuppressionLite Quantization Example](https://atomgit.com/mindspore/golden-stick/blob/master/example/deepseekv3/a8w8-osl/readme.md).

**Note:**

- Currently, quantization calibration is only supported on the Atlas 800I A2.  
- Do **not** install the `accelerate` library in the environment; otherwise, errors will occur during quantization.

### Downloading Quantized Weights

We have uploaded the quantized DeepSeek-R1 to [ModelArts Community](https://modelers.cn): [MindSpore-Lab/DeepSeek-R1-0528-A8W8](https://modelers.cn/models/MindSpore-Lab/DeepSeek-R1-0528-gs-A8W8). Refer to the [ModelArts Community documentation](https://modelers.cn/docs/en/openmind-hub-client/0.9/basic_tutorial/download.html) to download the weights locally.

## Quantized Model Inference

After obtaining the DeepSeek-R1 W8A8 weights, ensure they are stored in the relative path `DeepSeek-R1-W8A8`.

### Offline Inference

Refer to the [Installation Guide](../../../getting_started/installation/installation.md) to set up the vLLM-MindSpore Plugin environment. User needs to set the following environment variables:

```bash
export VLLM_MS_MODEL_BACKEND=MindFormers # use MindSpore Transformers as model backend.
```

Once ready, use the following Python code for offline inference:

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

## W8A8SC Sparse Quantization Models

W8A8SC (Weight 8-bit Activation 8-bit Sparse Compression) is a model compression method that combines sparsity, quantization, and compression technologies. It can achieve higher model compression ratios and faster inference speeds while maintaining inference accuracy.

The overall sparse quantization workflow includes: sparsifying weights and importance pruning → 8-bit quantization of weights and activations → compression encoding of quantized weights and generation of compressed weights and index files. The full workflow can be completed using MindSpore Golden Stick. For details, see [Golden Stick documentation](https://www.mindspore.cn/golden_stick/docs/en/master/index.html) and [Golden Stick repository](https://atomgit.com/mindspore/golden-stick).

### Technical Principles

The large model sparse quantization tool includes three parts: sparsity, quantization, and compression:

1. **Sparsity**: The model sparsity tool uses algorithms to determine the importance of each element in the model weights to the accuracy results, and sets the weight values that have little impact on the final accuracy to zero.

2. **Quantization**: Both weights and activations are quantized, converting high-bit floating-point numbers to 8-bit, which can directly reduce weight volume and bring performance benefits.

3. **Compression**: The weight compression tool further encodes and compresses model weights through compression algorithms, maximizing weight volume reduction and generating compressed weight and index files.

### Hardware Support Description

Compression algorithms are closely related to hardware. The support for sparse quantization on different hardware platforms is as follows:

- **Atlas 300I Duo**: Supports sparse quantization inference. Atlas 300I Duo has a unique UNZIP hardware unit that can perform online decompression of sparsely quantized models. This module is an execution unit of AI Core, subordinate to the MTE module, and is mainly responsible for moving compressed format parameters from HBM, DDR, and L2 buffer for decompression, writing the decompressed parameters to L0 or L1 Buffer. The UNZIP module supports sparse mode and can further perform sparsity + compression/decompression on quantized weights to improve model inference performance.

- **Atlas 800I A2**: Does not support W8A8SC model inference.

For more information about AI Core architecture and hardware features, refer to the following documents:

- [AI Core Architecture](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/developmentguide/opdevg/tbeaicpudevg/atlasopdev_10_0008.html)
- [Ascend AI Processor Hardware Features](https://www.hiascend.com/developer/blog/details/0268202185272112101)

### Data Format Requirements

Atlas 300I Duo does not support the bf16 data format and only supports fp16. Therefore, sparse quantization only supports the fp16 format. If weights are quantized on Atlas 800I A2, the weight format needs to be converted from bf16 to fp16 to support Atlas 300I Duo.

### Online Inference with Sparse Quantization Models

Online inference with sparse quantization (W8A8SC) models refers to deploying models that have been processed through sparsity, quantization, and compression into an inference environment. It leverages the online decompression and efficient inference capabilities provided by hardware platforms (such as Atlas 300I Duo) to achieve low-latency, high-throughput model services. In this process, users only need to configure parameters and service environments according to platform requirements, and can perform online inference on compressed sparse quantization models through specific inference interfaces, achieving resource savings and improved inference efficiency. This section details how to load sparse quantization models and complete the full online inference workflow under the vLLM-MindSpore plugin. The following uses Qwen3-8B-W8A8SC-TP2 as an example to introduce the inference process.

Qwen3-8B-W8A8SC-TP2 model weights can be downloaded from [ModelArts Community](https://modelers.cn/models/sunghajun/Qwen3-8B-W8A8SC-TP2).

#### Parameter Configuration

The following options need to be configured in the startup command `vllm-mindspore serve`:

```bash
export VLLM_MS_MODEL_BACKEND=Native
export MS_ENABLE_LCCL=off
export HCCL_OP_EXPANSION_MODE="AI_CPU"
export MS_ENABLE_INTERNAL_BOOST=off
export MS_ALLOC_CONF=enable_vmm:true
export MS_INTERNAL_ENABLE_CUSTOM_KERNEL_LIST=RmsNormQuant
```

#### Service Startup Example

```bash
vllm-mindspore serve "/path/to/Qwen3-8B-W8A8SC-TP2/" --quantization golden-stick  --load-format sparse_quant  --trust_remote_code --tensor_parallel_size=2 --max-num-seqs=256 --block-size=128 --gpu-memory-utilization=0.8 --max-num-batched-tokens=16384 --max-model-len=32768  2>&1 | tee log_master.txt
```

After the service starts successfully, you will get similar execution results:

```bash
(APIServer pid=919526) INFO:     Started server process [919526]
(APIServer pid=919526) INFO:     Waiting for application startup.
(APIServer pid=919526) INFO:     Application startup complete.
```

#### Sending Requests

```bash
curl http:/127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/Qwen3-8B-W8A8SC-TP2/",
    "prompt": "Introduce the Forbidden City in Beijing",
    "max_tokens": 120,
    "temperature": 0
  }'
```

#### Single Request Accuracy Verification

```bash
{"id":"cmpl-46de75f5f3cf4b83a4161842fa61c6bc","object":"text_completion","created":1770365772,"model":"/path/to/Qwen3-8B-W8A8SC-TP2/","choices":[{"index":0,"text":" 的建筑风格和历史背景。\n北京故宫，又称紫禁城，是中国明清两代的皇家宫殿，位于北京市中心，是世界上现存规模最大、保存最完整的古代宫殿建筑群。它始建于明永乐四年(140","logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null,"prompt_logprobs":null,"prompt_token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":3,"total_tokens":53,"completion_tokens":50,"prompt_tokens_details":null},"kv_transfer_params":null}
```

#### Ceval Dataset Verification

| dataset      | version | metric   | mode | vllm-api-general-chat |
| ------------ | ------- | -------- | ---- | --------------------- |
| cevaldataset | -       | accuracy | gen  | 86.40                 |

The Ceval dataset accuracy reaches 86.40%, proving that the sparse quantization model meets the accuracy requirements. The accuracy of this dataset is affected by parameters such as `batch_size` and `max_out_len`, with a fluctuation range of 0.5%.