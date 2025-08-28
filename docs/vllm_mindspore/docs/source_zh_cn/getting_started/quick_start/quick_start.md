# 快速体验

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.1/docs/vllm_mindspore/docs/source_zh_cn/getting_started/quick_start/quick_start.md)

本文档将为用户提供快速指引，以[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)模型为例，使用[docker](https://www.docker.com/)的安装方式部署vLLM-MindSpore插件，并以[离线推理](#离线推理)与[在线推理](#在线推理)两种方式，快速体验vLLM-MindSpore插件的服务化与推理能力。如用户需要了解更多的安装方式，请参考[安装指南](../installation/installation.md)。

## docker安装

在本章节中，我们推荐使用docker创建方式，快速部署vLLM-MindSpore插件环境。以下是部署docker的步骤介绍：

### 构建镜像

用户可执行以下命令，拉取vLLM-MindSpore插件代码仓库：

```bash
git clone -b r0.4.0 https://gitee.com/mindspore/vllm-mindspore.git
```

根据计算卡类型，构建镜像：

- 若为Atlas 800I A2，则执行

  ```bash
  bash build_image.sh
  ```

- 若为Atlas 300I Duo，则执行

  ```bash
  bash build_image.sh -a 310p
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

新建容器后成功后，将返回容器ID。用户可执行以下命令，确认容器是否创建成功：

```bash
docker ps
```

### 进入容器

用户在完成[新建容器](#新建容器)后，使用已定义的环境变量`DOCKER_NAME`，启动并进入容器：

```bash
docker exec -it $DOCKER_NAME bash
```

## 使用服务

用户在环境部署完毕后，运行模型前需要准备模型文件。用户可通过[下载模型](#下载模型)章节的指引进行模型准备。在[设置环境变量](#设置环境变量)后，可采用[离线推理](#离线推理)或[在线推理](#在线推理)的方式进行模型体验。

### 下载模型

用户可采用[Python工具下载](#python工具下载)或[git-lfs工具下载](#git-lfs工具下载)两种方式，进行模型下载。

#### Python工具下载

执行以下 Python 脚本，从[Hugging Face社区](https://huggingface.co/)下载[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)权重及文件：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="/path/to/save/Qwen2.5-7B-Instruct",
    local_dir_use_symlinks=False
)
```

其中，`local_dir`为模型保存路径，由用户指定，请确保该路径下有足够的硬盘空间。

#### git-lfs工具下载

执行以下代码以确认[git-lfs](https://git-lfs.com)工具是否可用：

```bash
git lfs install
```

如果可用，将获得如下返回结果：

```text
Git LFS initialized.
```

若工具不可用，则需要先安装[git-lfs](https://git-lfs.com)，可参考[FAQ](../../faqs/faqs.md)章节中关于[git-lfs安装](../../faqs/faqs.md#git-lfs安装)的阐述。

工具确认可用后，执行以下命令下载权重：

```bash
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
```

### 设置环境变量

用户在启动模型前，需设置以下环境变量：

```bash
export VLLM_MS_MODEL_BACKEND=MindFormers # use MindSpore Transformers as model backend.
```

以下是对上述环境变量的解释：

- `VLLM_MS_MODEL_BACKEND`：所运行的模型后端。目前vLLM-MindSpore插件所支持的模型与模型后端，可在[模型支持列表](../../user_guide/supported_models/models_list/models_list.md)中进行查询。

### 离线推理

以[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) 为例，用户可以使用如下Python脚本，进行模型的离线推理：

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
llm = LLM(model="Qwen2.5-7B-Instruct")
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

### 在线推理

vLLM-MindSpore插件可使用OpenAI的API协议，进行在线推理部署。以下以[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)为例，介绍模型的[启动服务](#启动服务)和[发送请求](#发送请求)，得到在线推理的推理结果。

#### 启动服务

使用模型`Qwen/Qwen2.5-7B-Instruct`，执行如下命令启动vLLM服务：

```bash
vllm-mindspore serve Qwen/Qwen2.5-7B-Instruct
```

用户可以通过指定模型保存的本地路径作为模型标签。若服务成功启动，则可以获得类似的执行结果：

```text
INFO:   Started server process [6363]
INFO:   Waiting for application startup.
INFO:   Application startup complete.
```

另外，日志中还会打印服务的性能数据信息，如：

```text
Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```

#### 发送请求

使用如下命令发送请求。其中，`prompt`字段为模型输入：

```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "prompt": "I am", "max_tokens": 20, "temperature": 0}'
```

其中，用户需确认`"model"`字段与启动服务中的模型标签一致，请求才能成功匹配到模型。若请求处理成功，将获得以下推理结果：

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
