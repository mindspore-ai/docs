# 单卡推理（Qwen2.5-7B）

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.1/docs/vllm_mindspore/docs/source_zh_cn/getting_started/tutorials/qwen2.5_7b_singleNPU/qwen2.5_7b_singleNPU.md)

本文档将介绍使用vLLM-MindSpore插件进行单卡推理的流程。以[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)模型为例，用户可通过以下[docker安装](#docker安装)章节或[安装指南](../../installation/installation.md#安装指南)章节进行环境配置，并[下载模型权重](#下载模型权重)。在[设置环境变量](#设置环境变量)之后，可进行[离线推理](#离线推理)与[在线推理](#在线推理)，体验单卡推理功能。

## docker安装

在本章节中，我们推荐使用docker创建的方式，快速部署vLLM-MindSpore插件环境。以下是部署docker的步骤介绍：

### 构建镜像

用户可执行以下命令，拉取vLLM-MindSpore插件代码仓库，并构建镜像：

```bash
git clone -b r0.4.0 https://gitee.com/mindspore/vllm-mindspore.git
bash build_image.sh
```

构建成功后，用户可以得到以下信息：

```text
Successfully built e40bcbeae9fc
Successfully tagged vllm_ms_20250726:latest
```

其中，`e40bcbeae9fc`为镜像ID，`vllm_ms_20250726:latest`为镜像名与tag。用户可执行以下命令，确认docker镜像创建成功：

```bash
docker images
```

### 新建容器

用户在完成[构建镜像](#构建镜像)后，设置`DOCKER_NAME`与`IMAGE_NAME`为容器名与镜像名，并执行以下命令新建容器：

```bash
export DOCKER_NAME=vllm-mindspore-container  # your container name
export IMAGE_NAME=vllm_ms_20250726:latest  # your image name

docker run -itd --name=${DOCKER_NAME} --ipc=host --network=host --privileged=true \
        --device=/dev/davinci0 \
        --device=/dev/davinci1 \
        --device=/dev/davinci2 \
        --device=/dev/davinci3 \
        --device=/dev/davinci4 \
        --device=/dev/davinci5 \
        --device=/dev/davinci6 \
        --device=/dev/davinci7 \
        --device=/dev/davinci_manager \
        --device=/dev/devmm_svm \
        --device=/dev/hisi_hdc \
        -v /usr/local/sbin/:/usr/local/sbin/ \
        -v /var/log/npu/slog/:/var/log/npu/slog \
        -v /var/log/npu/profiling/:/var/log/npu/profiling \
        -v /var/log/npu/dump/:/var/log/npu/dump \
        -v /var/log/npu/:/usr/slog \
        -v /etc/hccn.conf:/etc/hccn.conf \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /etc/vnpu.cfg:/etc/vnpu.cfg \
        --shm-size="250g" \
        ${IMAGE_NAME} \
        bash
```

新建容器成功后，将返回容器ID。用户可执行以下命令，确认容器是否创建成功：

```bash
docker ps
```

### 进入容器

用户在完成[新建容器](#新建容器)后，使用已定义的环境变量`DOCKER_NAME`，启动并进入容器：

```bash
docker exec -it $DOCKER_NAME bash
```

## 下载模型权重

用户可采用[Python工具下载](#python工具下载)或[git-lfs工具下载](#git-lfs工具下载)两种方式，进行模型下载。

### Python工具下载

执行以下 Python 脚本，从[Hugging Face社区](https://huggingface.co/)下载[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)权重及文件：

```python
from openmind_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="/path/to/save/Qwen2.5-7B-Instruct",
    local_dir_use_symlinks=False
)
```

其中`local_dir`为模型保存路径，由用户指定，请确保该路径下有足够的硬盘空间。

### git-lfs工具下载

执行以下代码，以确认[git-lfs](https://git-lfs.com)工具是否可用：

```bash
git lfs install
```

如果可用，将获得如下返回结果：

```text
Git LFS initialized.
```

若工具不可用，则需要先安装[git-lfs](https://git-lfs.com)，可参考[FAQ](../../../faqs/faqs.md)章节中关于[git-lfs安装](../../../faqs/faqs.md#git-lfs安装)的阐述。

工具确认可用后，执行以下命令下载权重：

```bash
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
```

## 设置环境变量

以[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)为例，以下环境变量用于设置内存占用、后端以及模型相关的YAML文件：

```bash
#set environment variables
export VLLM_MS_MODEL_BACKEND=MindFormers # use MindSpore TransFormers as model backend.
export MINDFORMERS_MODEL_CONFIG=$YAML_PATH # Set the corresponding MindSpore Transformers model's YAML file.
```

以下是对上述环境变量的解释：

- `VLLM_MS_MODEL_BACKEND`：所运行的模型后端。目前vLLM-MindSpore插件所支持的模型与模型后端，可在[模型支持列表](../../../user_guide/supported_models/models_list/models_list.md)中进行查询；
- `MINDFORMERS_MODEL_CONFIG`：模型配置文件。用户可以在[MindSpore Transformers工程](https://gitee.com/mindspore/mindformers/tree/r1.7.0/research/qwen2_5)中，找到对应模型的YAML文件。以Qwen2.5-7B为例，其YAML文件为[predict_qwen2_5_7b_instruct.yaml](https://gitee.com/mindspore/mindformers/blob/r1.7.0/research/qwen2_5/predict_qwen2_5_7b_instruct.yaml)。

用户可通过`npu-smi info`查看显存占用情况，并可以使用如下环境变量，设置用于推理的计算卡：

```bash
export NPU_VISIBLE_DEVICES=0
export ASCEND_RT_VISIBLE_DEVICES=$NPU_VISIBLE_DEVICES
```

## 离线推理

vLLM-MindSpore插件环境搭建之后，用户可以使用如下Python代码，进行模型的离线推理：

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
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}. Generated text: {generated_text!r}")
```

若成功执行，则可以获得类似的执行结果：

```text
Prompt: 'I am'. Generated text: ' trying to create a virtual environment for my Python project, but I am encountering some'
Prompt: 'Today is'. Generated text: ' the 100th day of school. To celebrate, the teacher has'
Prompt: 'Llama is'. Generated text: ' a 100% natural, biodegradable, and compostable alternative'
```

## 在线推理

vLLM-MindSpore插件可使用OpenAI的API协议，部署在线推理。以下以[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)为例，介绍模型的[启动服务](#启动服务)和[发送请求](#发送请求)，得到在线推理的推理结果。

### 启动服务

使用如下命令启动vLLM服务：

```bash
python3 -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server --model "Qwen/Qwen2.5-7B-Instruct"
```

用户可以通过`--model`参数，指定模型保存的本地路径。若服务成功启动，则可以获得类似的执行结果：

```text
INFO:   Started server process [6363]
INFO:   Waiting for application startup.
INFO:   Application startup complete.
```

另外，日志中还会打印出服务的性能数据信息，如：

```text
Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```

### 发送请求

使用如下命令发送请求。其中`prompt`字段为模型输入：

```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "prompt": "I am", "max_tokens": 20, "temperature": 0}'
```

其中，用户需确认`"model"`字段与启动服务中的`--model`一致，请求才能成功匹配到模型。若请求处理成功，将获得以下推理结果：

```text
{
    "id":"cmpl-bac2b14c726b48b9967bcfc724e7c2a8","object":"text_completion",
    "create":1748485893,
    "model":"Qwen2.5-7B-Instruct",
    "choices":[
        {
            "index":0,
            "text":"trying to create a virtual environment for my Python project, but I am encountering some issues with setting up",
            "logprobs":null,
            "finish_reason":"length",
            "stop_reason":null,
            "prompt_logprobs":null
        }
    ],
    "usage":{
        "prompt_tokens":2,
        "total_tokens":22,
        "completion_tokens":20,
        "prompt_tokens_details":null
    }
}
```
