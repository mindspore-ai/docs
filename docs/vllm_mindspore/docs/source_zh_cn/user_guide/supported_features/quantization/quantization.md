# 量化方法

本文档将为用户介绍模型量化与量化推理的方法。量化方法通过牺牲部分模型精度的方式，达到降低模型部署时的资源需求的目的，并提升模型部署时的性能，从而允许模型被部署到更多的设备上。由于大语言模型的规模较大，出于成本考虑，训练后量化成为主流模型量化方案，具体可以参考[后量化技术简介](https://gitee.com/mindspore/golden-stick/blob/master/mindspore_gs/ptq/README_CN.md)。

本文档中，[创建量化模型](#创建量化模型)章节，将以[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)为例，介绍模型后量化的步骤；[量化模型推理](#量化模型推理)章节，介绍如何使用量化模型进行推理。

## 创建量化模型

以[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)网络为例，使用SmoothQuant算法对其进行A8W8量化。

### 使用MindSpore金箍棒量化网络

我们将使用[MindSpore 金箍棒的PTQ算法](https://gitee.com/mindspore/golden-stick/blob/master/mindspore_gs/ptq/ptq/README_CN.md)对Qwen3-8B网络进行SmoothQuant量化，详细方法参考[Qwen3-SmoothQuant量化样例](todo)

#### Qwen3-8B网络权重下载

用户可使用huggingface-cli下载网络权重：

```bash
huggingface-cli download --resume-download Qwen/Qwen3-8B --local-dir Qwen3-8B-bf16
```

或可以使用[其他下载方式](../../../getting_started/quick_start/quick_start.md#下载模型)，进行权重下载。

#### 使用MindSpore Transformers加载网络

用户可以使用如下的脚本，依赖[MindFormers](https://gitee.com/mindspore/mindformers)，进行网络加载：

```python
from mindformers import AutoModel
from mindformers import AutoTokenizer

network = AutoModel.from_pretrained("Qwen3-8B-bf16")
tokenizer = AutoTokenizer.from_pretrained("Qwen3-8B-bf16")
```

#### 准备CEval数据集

将CEval数据集的下载到ceval目录下，目录结构如下：

```bash
ceval
  ├── dev
  ├── test
  └── val
```

使用MindSpore创建数据集句柄：

```python
from mindspore import GeneratorDataset
ds = GeneratorDataset(source="ceval", column_names=["subjects", "input_ids", "labels"])
```

#### 使用金箍棒进行后量化

用户可使用如下Python脚本，进行模型后量化：

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

执行校准前，需要将MindFormers工程根目录加到`PYTHONPATH`环境变量，从而用户可以成功import Qwen3网络相关类。

### 下载量化权重

我们已经将量化好的Qwen3-8B上传到[魔乐社区](https://modelers.cn)：[MindSpore-Lab/Qwen3-8B-A8W8](https://modelers.cn/models/MindSpore-Lab/Qwen3-8B-A8W8)，可以参考[魔乐社区文档](https://modelers.cn/docs/zh/openmind-hub-client/0.9/basic_tutorial/download.html)将权重下载到本地。

## 量化模型推理

在上一步中获取到Qwen3-8B SmoothQuant量化权重后，保证该权重存放相对路径为`Qwen3-8B-A8W8`。

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
llm = LLM(model="Qwen3-8B-A8W8", quantization='SmoothQuant')
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
