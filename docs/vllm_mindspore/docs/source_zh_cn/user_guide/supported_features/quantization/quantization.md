# 量化方法

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_zh_cn/user_guide/supported_features/quantization/quantization.md)

本文档将为用户介绍模型量化与量化推理的方法。量化方法通过牺牲部分模型精度的方式，达到降低模型部署时的资源需求的目的，并提升模型部署时的性能，从而允许模型被部署到更多的设备上。由于大语言模型的规模较大，出于成本考虑，训练后量化成为主流模型量化方案，具体可以参考[后量化技术简介](https://atomgit.com/mindspore/golden-stick/blob/master/mindspore_gs/ptq/README_CN.md)。

本文档中，[创建量化模型](#创建量化模型)章节将以[DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)为例，介绍模型后量化的步骤；[量化模型推理](#量化模型推理)章节介绍如何使用量化模型进行推理；[W8A8SC稀疏量化模型](#w8a8sc稀疏量化模型)章节介绍W8A8SC稀疏量化技术的原理、硬件支持情况和数据格式要求。

## 创建量化模型

以[DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)网络为例，使用OutlierSuppressionLite算法对其进行W8A8量化。该章节需依赖MindSpore金箍棒模块，请参考[这里](https://www.mindspore.cn/golden_stick/docs/zh-CN/master/index.html)了解该模块。

### 使用MindSpore金箍棒量化网络

我们将使用[MindSpore 金箍棒的PTQ算法](https://atomgit.com/mindspore/golden-stick/blob/master/mindspore_gs/ptq/ptq/README_CN.md)对DeepSeek-R1网络进行量化，详细方法参考[DeepSeekR1-OutlierSuppressionLite量化样例](https://atomgit.com/mindspore/golden-stick/blob/master/example/deepseekv3/a8w8-osl/readme.md)。

注意事项：

- 当前仅支持在Atlas 800I A2上进行量化校准；
- 环境中不能安装accelerate库，否则会导致量化过程中报错。

### 直接下载量化权重

我们已经将量化好的DeepSeek-R1上传到[魔乐社区](https://modelers.cn)：[MindSpore-Lab/DeepSeek-R1-0528-A8W8](https://modelers.cn/models/MindSpore-Lab/DeepSeek-R1-0528-gs-A8W8)，可以参考[魔乐社区文档](https://modelers.cn/docs/zh/openmind-hub-client/0.9/basic_tutorial/download.html)将权重下载到本地。

## 量化模型推理

在上一步中获取到DeepSeek-R1 W8A8量化权重后，保证该权重存放相对路径为`DeepSeek-R1-W8A8`。

### 离线推理

用户可以参考[安装指南](../../../getting_started/installation/installation.md)，进行vLLM-MindSpore插件的环境搭建。用户需设置以下环境变量：

```bash
export VLLM_MS_MODEL_BACKEND=MindFormers # use MindSpore Transformers as model backend.
```

环境准备完成后，用户可以使用如下Python代码，进行离线推理服务：

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

## W8A8SC稀疏量化模型

W8A8SC（Weight 8-bit Activation 8-bit Sparse Compression）是一种结合稀疏、量化和压缩技术的模型压缩方法，能够在保持推理精度的同时，实现更高的模型压缩比和更快的推理速度。

稀疏量化整体流程包括：对权重进行稀疏化与重要性筛选 → 对权重和激活做 8bit 量化 → 对量化后权重做压缩编码并生成压缩权重与索引文件。使用 MindSpore 金箍棒（Golden Stick）可完成上述全流程，详见 [Golden Stick 文档](https://www.mindspore.cn/golden_stick/docs/zh-CN/master/index.html) 与 [Golden Stick 仓库](https://atomgit.com/mindspore/golden-stick)。

### 技术原理

大模型稀疏量化工具包括稀疏、量化和压缩三个部分：

1. **稀疏**：模型稀疏工具通过算法判断模型权重中每个元素对精度结果的重要性，并将模型权重中对最终精度影响小的权重值置零。

2. **量化**：对权重和激活值都做量化，将高位浮点数转为8bit，可以直接降低权重体积，带来性能收益。

3. **压缩**：权重压缩工具将模型权重通过压缩算法进一步编码压缩，最大程度地降低权重体积，生成压缩后权重和索引文件。

### 硬件支持说明

压缩算法和硬件强相关，不同硬件平台对稀疏量化的支持情况如下：

- **Atlas 300I Duo**：支持稀疏量化推理。Atlas 300I Duo特有的UNZIP硬件单元可以对稀疏量化的模型实行在线解压缩。该模块为AI Core的执行单元，下挂于MTE模块，主要负责从HBM、DDR以及L2 buffer中搬移压缩格式的参数进行解压缩，将压缩后的参数写到L0或L1 Buffer中。UNZIP模块支持稀疏模式，可以针对量化后的权重进一步做稀疏+压缩/解压缩从而提升模型的推理性能。

- **Atlas 800I A2**：无法支持W8A8SC模型的推理。

如需了解更多关于AI Core架构和硬件特性的详细信息，可参考以下文档：

- [AI Core架构](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/developmentguide/opdevg/tbeaicpudevg/atlasopdev_10_0008.html)
- [昇腾AI处理器硬件特性](https://www.hiascend.com/developer/blog/details/0268202185272112101)

### 数据格式要求

Atlas 300I Duo不支持bf16的数据格式，仅支持fp16，因此稀疏量化也仅支持fp16格式。如果是由Atlas 800I A2量化的权重，需要对权重格式转化，由bf16格式转换为fp16格式，以支持Atlas 300I Duo。

### 稀疏量化模型在线推理

稀疏量化（W8A8SC）模型在线推理，指的是将经过稀疏、量化和压缩处理后得到的模型部署到推理环境中，利用硬件平台（如Atlas 300I Duo）提供的在线解压缩和高效推理能力，实现低延迟、高吞吐的模型服务。在这一过程中，用户只需按照平台要求配置参数与服务环境，通过特定的推理接口即可对压缩后的稀疏量化模型进行在线推理，实现资源节省和推理效率提升。本节将详细介绍在vLLM-MindSpore插件下，如何加载稀疏量化模型并完成在线推理全流程。下面以Qwen3-8B-W8A8SC-TP2为例，介绍推理流程。

Qwen3-8B-W8A8SC-TP2 模型权重可从 [魔乐社区](https://modelers.cn/models/sunghajun/Qwen3-8B-W8A8SC-TP2) 下载。

#### 参数配置

需要在启动命令`vllm-mindspore serve` 中配置以下选项：

```bash
export VLLM_MS_MODEL_BACKEND=Native
export MS_ENABLE_LCCL=off
export HCCL_OP_EXPANSION_MODE="AI_CPU"
export MS_ENABLE_INTERNAL_BOOST=off
export MS_ALLOC_CONF=enable_vmm:true
export MS_INTERNAL_ENABLE_CUSTOM_KERNEL_LIST=RmsNormQuant
```

#### 服务启动示例

```bash
vllm-mindspore serve "/path/to/Qwen3-8B-W8A8SC-TP2/" --quantization golden-stick  --load-format sparse_quant  --trust_remote_code --tensor_parallel_size=2 --max-num-seqs=256 --block-size=128 --gpu-memory-utilization=0.8 --max-num-batched-tokens=16384 --max-model-len=32768  2>&1 | tee log_master.txt
```

启动服务成功后，可获得类似的执行结果：

```bash
(APIServer pid=919526) INFO:     Started server process [919526]
(APIServer pid=919526) INFO:     Waiting for application startup.
(APIServer pid=919526) INFO:     Application startup complete.
```

#### 发送请求

```bash
curl http:/127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/Qwen3-8B-W8A8SC-TP2/",
    "prompt": "介绍一下北京故宫",
    "max_tokens": 120,
    "temperature": 0
  }'
```

#### 单请求精度验证

```bash
{"id":"cmpl-46de75f5f3cf4b83a4161842fa61c6bc","object":"text_completion","created":1770365772,"model":"/path/to/Qwen3-8B-W8A8SC-TP2/","choices":[{"index":0,"text":"的建筑风格和历史背景。\n北京故宫，又称紫禁城，是中国明清两代的皇家宫殿，位于北京市中心，是世界上现存规模最大、保存最完整的古代宫殿建筑群。它始建于明永乐四年（140","logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null,"prompt_logprobs":null,"prompt_token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":3,"total_tokens":53,"completion_tokens":50,"prompt_tokens_details":null},"kv_transfer_params":null}
```

#### Ceval数据集验证

| dataset      | version | metric   | mode | vllm-api-general-chat |
| ------------ | ------- | -------- | ---- | --------------------- |
| cevaldataset | -       | accuracy | gen  | 86.40                 |

Ceval数据集精度达到86.40%，证明稀疏量化模型的精度达标。该数据集准确率受到batch_size、max_out_len等参数影响，波动范围为0.5%。