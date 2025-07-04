# 多卡推理（Qwen2.5-32B）

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_zh_cn/getting_started/tutorials/qwen2.5_32b_multiNPU/qwen2.5_32b_multiNPU.md)

本文档将为用户介绍使用vLLM MindSpore进行单节点多卡的推理流程。以[Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)模型为例，用户通过以下[docker安装](#docker安装)章节，或[安装指南](../../installation/installation.md#安装指南)进行环境配置，并[下载模型权重](#下载模型权重)。在[设置环境变量](#设置环境变量)之后，可部署[在线推理](#在线推理)，以体验单节点多卡的推理功能。

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

## 下载模型权重

用户可采用[Python工具下载](#python工具下载)或[git-lfs工具下载](#git-lfs工具下载)两种方式，进行模型下载。

### Python工具下载

执行以下 Python 脚本，从[Huggingface Face社区](https://huggingface.co/)下载 [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) 权重及文件：

```python
from openmind_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-32B-Instruct",
    local_dir="/path/to/save/Qwen2.5-32B-Instruct",
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

工具确认可用后，执行以下命令，下载权重：

```bash
git clone https://huggingface.co/Qwen/Qwen2.5-32B-Instruct
```

## 设置环境变量

以[Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)为例，以下环境变量用于设置内存占用，后端以及模型相关的YAML文件。
其中，关于[Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)的环境变量如下：

```bash
#set environment variables
export ASCEND_TOTAL_MEMORY_GB=64 # Please use `npu-smi info` to check the memory.
export vLLM_MODEL_BACKEND=MindFormers # use MindFormers as model backend.
export vLLM_MODEL_MEMORY_USE_GB=32 # Memory reserved for model execution. Set according to the model's maximum usage, with the remaining environment used for kvcache allocation
export MINDFORMERS_MODEL_CONFIG=$YAML_PATH # Set the corresponding MindSpore Transformers model's YAML file.
```

以下是对上述环境变量的解释：

- `ASCEND_TOTAL_MEMORY_GB`: 每一张计算卡的显存大小。用户可使用`npu-smi info`命令进行查询，该值对应查询结果中的`HBM-Usage(MB)`。
- `vLLM_MODEL_BACKEND`：所运行的模型后端。目前vLLM MindSpore所支持的模型与模型后端，可在[模型支持列表](../../../user_guide/supported_models/models_list/models_list.md)中进行查询。
- `vLLM_MODEL_MEMORY_USE_GB`：模型加载时所用空间，根据用户所使用的模型进行设置。若用户在模型加载过程中遇到显存不足时，可适当增大该值并重试。
- `MINDFORMERS_MODEL_CONFIG`：模型配置文件。用户可以在[MindSpore Transformers工程](https://gitee.com/mindspore/mindformers/tree/r1.5.0/research/qwen2_5)中，找到对应模型的yaml文件。以Qwen2.5-32B为例，则其yaml文件为[predict_qwen2_5_32b_instruct.yaml](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/qwen2_5/predict_qwen2_5_32b_instruct.yaml) 。

用户可通过`npu-smi info`查看显存占用情况，并可以使用如下环境变量，设置用于推理的计算卡。以下例子为假设用户使用4,5,6,7卡进行推理：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
```

## 在线推理

vLLM MindSpore可使用OpenAI的API协议，部署为在线服务。以下是在线服务的拉起流程。以下是以[Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) 为例，介绍模型的[启动服务](#启动服务)，并[发送请求](#发送请求)，得到在线服务的推理结果。

### 启动服务

用如下命令拉起服务：

```bash
export TENSOR_PARALLEL_SIZE=4
export MAX_MODEL_LEN=1024
python3 -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server --model "Qwen/Qwen2.5-32B-Instruct" --trust_remote_code --tensor-parallel-size $TENSOR_PARALLEL_SIZE --max-model-len $MAX_MODEL_LEN
```

其中，`TENSOR_PARALLEL_SIZE`为用户指定的卡数，`MAX_MODEL_LEN`为模型最大输出token数。

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

### 发送请求

使用如下命令发送请求。其中`prompt`字段为模型输入：

```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen2.5-32B-Instruct", "prompt": "I am", "max_tokens": 20, "temperature": 0}'
```

若请求处理成功，将获得以下的推理结果：

```text
{
    "id":"cmpl-11fe2898c77d4ff18c879f57ae7aa9ca","object":"text_completion",
    "create":1748568696,
    "model":"Qwen2.5-32B-Instruct",
    "choices":[
        {
            "index":0,
            "text":"trying to create a virtual environment in Python using venv, but I am encountering some issues with setting",
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
