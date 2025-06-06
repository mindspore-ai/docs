# Quantization Methods

[![View Source](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/user_guide/supported_features/quantization/quantization.md)

This document introduces model quantization and quantized inference methods. Quantization reduces inference resource with minor cost of precision, while improving inference performance to enable deployment on more devices. With the large scale of LLMs, post-training quantization has become the mainstream approach for model quantization. For details, refer to [Post-Training Quantization Introduction](https://gitee.com/mindspore/golden-stick/blob/master/mindspore_gs/ptq/README_CN.md).

In this document, the [Creating Quantized Models](#creating-quantized-models) section introduces post-training quantization steps using [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) as an example. A the [Quantized Model Inference](#quantized-model-inference) section explains how to perform inference with quantized models.

## Creating Quantized Models

We use the [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) network as an example to introduce A8W8 quantization with the SmoothQuant algorithm.

### Quantizing Networks with MindSpore Golden Stick

We employ [MindSpore Golden Stick's PTQ algorithm](https://gitee.com/mindspore/golden-stick/blob/master/mindspore_gs/ptq/ptq/README_CN.md) for SmoothQuant quantization of Qwen3-8B. For detailed methods, refer to [Qwen3-SmoothQuant Quantization Example](todo).

#### Downloading Qwen3-8B Weights

Users can download the weights using huggingface-cli:

```bash
huggingface-cli download --resume-download Qwen/Qwen3-8B --local-dir Qwen3-8B-bf16
```

Alternatively, use [other download methods](../../../getting_started/quick_start/quick_start.md#download-model).

#### Loading the Network with MindSpore Transformers

Load the network using [MindSpore Transformers](https://gitee.com/mindspore/mindformers) with the following script:

```python
from mindformers import AutoModel
from mindformers import AutoTokenizer

network = AutoModel.from_pretrained("Qwen3-8B-bf16")
tokenizer = AutoTokenizer.from_pretrained("Qwen3-8B-bf16")
```

#### Preparing the CEval Dataset

Download the CEval dataset to the `ceval` directory with the following structure:

```bash
ceval
  ├── dev
  ├── test
  └── val
```

Create a dataset handle using MindSpore:

```python
from mindspore import GeneratorDataset
ds = GeneratorDataset(source="ceval", column_names=["subjects", "input_ids", "labels"])
```

#### Performing Post-Training Quantization with Golden Stick

Use the following Python script for post-training quantization:

```python
from mindspore import dtype as msdtype
from mindspore_gs.ptq import PTQ
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQConfig, PTQMode, OutliersSuppressionType, QuantGranularity, PrecisionRecovery

cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                act_quant_dtype=msdtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH,
                opname_blacklist=['lm_head'])
w2_config = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                      act_quant_dtype=msdtype.int8,
                      outliers_suppression=OutliersSuppressionType.NONE,
                      precision_recovery=PrecisionRecovery.NONE,
                      act_quant_granularity=QuantGranularity.PER_TOKEN,
                      weight_quant_granularity=QuantGranularity.PER_CHANNEL)
layer_policies = OrderedDict({r'.*\.w2.*': w2_config})
ptq = PTQ(config=cfg, layer_policies=layer_policies)
from research.qwen3.qwen3_transformers import Qwen3ParallelTransformerLayer
ptq.decoder_layer_types.append(Qwen3ParallelTransformerLayer)
ptq.apply(network, ds)
ptq.convert(network)
ms.save_checkpoint(network.parameters_dict(), "Qwen3-8B-A8W8", format="safetensors",
                   choice_func=lambda x: "key_cache" not in x and "value_cache" not in x and "float_weight" not in x)
```

Before calibration, add the MindSpore Transformers root directory to the `PYTHONPATH` environment variable, and check Qwen3-related classes have been successfully imported.

### Downloading Quantized Weights

We have uploaded the quantized Qwen3-8B to [ModelArts Community](https://modelers.cn): [MindSpore-Lab/Qwen3-8B-A8W8](https://modelers.cn/models/MindSpore-Lab/Qwen3-8B-A8W8). Refer to the [ModelArts Community documentation](https://modelers.cn/docs/zh/openmind-hub-client/0.9/basic_tutorial/download.html) to download the weights locally.

## Quantized Model Inference

After obtaining the Qwen3-8B SmoothQuant weights, ensure they are stored in the relative path `Qwen3-8B-A8W8`.

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
llm = LLM(model="Qwen3-8B-A8W8", quantization='SmoothQuant')
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
