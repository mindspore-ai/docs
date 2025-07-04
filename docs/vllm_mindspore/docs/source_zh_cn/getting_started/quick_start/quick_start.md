# 快速体验

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_zh_cn/getting_started/quick_start/quick_start.md)

本文档将为用户提供快速指引，以[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)模型为例，使用[docker](https://www.docker.com/)的安装方式部署vLLM MindSpore，并以[离线推理](#离线推理)与[在线服务](#在线服务)两种方式，快速体验vLLM MindSpore的服务化与推理能力。如用户需要了解更多的安装方式，请参考[安装指南](../installation/installation.md)。

## docker安装

在本章节中，我们推荐用docker创建的方式，以快速部署vLLM MindSpore环境，以下是部署docker的步骤介绍：

### 拉取镜像

拉取vLLM MindSpore的docker镜像。执行以下命令进行拉取：

```bash
docker pull hub.oepkgs.net/oedeploy/openeuler/aarch64/mindspore:latest
```

拉取过程中，用户将看到docker镜像各layer的拉取进度。拉取成功后，用户可执行以下命令，确认docker镜像拉取成功：

```bash
docker images
```

### 新建容器

用户在完成[拉取镜像](#拉取镜像)后，设置`DOCKER_NAME`与`IMAGE_NAME`为容器名与镜像名，并执行以下命令新建容器：

```bash
export DOCKER_NAME=vllm-mindspore-container  # your container name
export IMAGE_NAME=hub.oepkgs.net/oedeploy/openeuler/aarch64/mindspore:latest  # your image name

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

用户在环境部署完毕后，在运行模型前，需要准备模型文件，用户可通过[下载模型](#下载模型)章节的指引作模型准备，在[设置环境变量](#设置环境变量)后，可采用[离线推理](#离线推理)或[在线服务](#在线服务)的方式，进行模型体验。

### 下载模型

用户可采用[Python工具下载](#python工具下载)或[git-lfs工具下载](#git-lfs工具下载)两种方式，进行模型下载。

#### Python工具下载

执行以下 Python 脚本，从[Huggingface Face社区](https://huggingface.co/)下载 [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) 权重及文件：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="/path/to/save/Qwen2.5-7B-Instruct",
    local_dir_use_symlinks=False
)
```

其中`local_dir`为模型保存路径，由用户指定，请确保该路径下有足够的硬盘空间。

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

工具确认可用后，执行以下命令，下载权重：

```bash
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
```

### 设置环境变量

用户在拉起模型前，需设置以下环境变量：

```bash
export ASCEND_TOTAL_MEMORY_GB=64 # Please use `npu-smi info` to check the memory.
export vLLM_MODEL_BACKEND=MindFormers # use MindSpore Transformers as model backend.
export vLLM_MODEL_MEMORY_USE_GB=32 # Memory reserved for model execution. Set according to the model's maximum usage, with the remaining environment used for kvcache allocation
export MINDFORMERS_MODEL_CONFIG=$YAML_PATH # Set the corresponding MindSpore Transformers model's YAML file.
```

以下是对上述环境变量的解释：

- `ASCEND_TOTAL_MEMORY_GB`: 每一张计算卡的显存大小。用户可使用`npu-smi info`命令进行查询，该值对应查询结果中的`HBM-Usage(MB)`；
- `vLLM_MODEL_BACKEND`：所运行的模型后端。目前vLLM MindSpore所支持的模型与模型后端，可在[模型支持列表](../../user_guide/supported_models/models_list/models_list.md)中进行查询；
- `vLLM_MODEL_MEMORY_USE_GB`：模型加载时所用空间，根据用户所使用的模型进行设置。若用户在模型加载过程中遇到显存不足时，可适当增大该值并重试；
- `MINDFORMERS_MODEL_CONFIG`：模型配置文件。

另外，用户需要确保MindSpore Transformers已安装。用户可通过

```bash
export PYTHONPATH=/path/to/mindformers:$PYTHONPATH
```

以引入MindSpore Tranformers。

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

vLLM MindSpore可使用OpenAI的API协议，进行在线服务部署。以下是以[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) 为例，介绍模型的[启动服务](#启动服务)，并[发送请求](#发送请求)，得到在线服务的推理结果。

#### 启动服务

使用模型`Qwen/Qwen2.5-7B-Instruct`，并用如下命令拉起vLLM服务：

```bash
python3 -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server --model "Qwen/Qwen2.5-7B-Instruct"
```

若服务成功拉起，则可以获得类似的执行结果：

```text
INFO:   Started server process [6363]
INFO:   Waiting for application startup.
INFO:   Application startup complete.
```

另外，日志中还会打印出服务的性能数据信息，如：

```text
Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg gereration throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```

#### 发送请求

使用如下命令发送请求。其中`prompt`字段为模型输入：

```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "prompt": "I am", "max_tokens": 20, "temperature": 0}'
```

若请求处理成功，将获得以下的推理结果：

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
