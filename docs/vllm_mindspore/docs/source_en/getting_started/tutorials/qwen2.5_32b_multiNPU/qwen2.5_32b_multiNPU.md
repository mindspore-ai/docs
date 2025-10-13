# Multi-Card Inference (Qwen2.5-32B)

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.7.1/docs/vllm_mindspore/docs/source_en/getting_started/tutorials/qwen2.5_32b_multiNPU/qwen2.5_32b_multiNPU.md)

This document introduces single-node multi-card inference process by vLLM-MindSpore Plugin. Taking the [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) model as an example, users can configure the environment through the [Docker Installation](#docker-installation) section or the [Installation Guide](../../installation/installation.md#installation-guide), and then [download the model weights](#downloading-model-weights). After [setting environment variables](#setting-environment-variables), users can perform [online inference](#online-inference) to experience single-node multi-card inference capabilities.

## Docker Installation

In this section, we recommend using Docker for quick deployment of the vLLM-MindSpore Plugin environment. Below are the steps for Docker deployment:

### Building the Image

User can execute the following commands to clone the vLLM-MindSpore Plugin code repository and build the image:

```bash
git clone -b r0.4.0 https://gitee.com/mindspore/vllm-mindspore.git
bash build_image.sh
```

After a successful build, user will get the following output:

```text
Successfully built e40bcbeae9fc
Successfully tagged vllm_ms_20250726:latest
```

Here, `e40bcbeae9fc` is the image ID, and `vllm_ms_20250726:latest` is the image name and tag. User can run the following command to confirm that the Docker image has been successfully created:

```bash
docker images
```

### Creating a Container

After [building the image](#building-the-image), set `DOCKER_NAME` and `IMAGE_NAME` as the container and image names, then create the container:

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

After successful creation, the container ID will be returned. Verify the container by running:

```bash
docker ps
```

### Entering the Container

After [creating the container](#creating-a-container), start and enter the container using the predefined `DOCKER_NAME`:

```bash
docker exec -it $DOCKER_NAME bash
```

## Downloading Model Weights

Users can download the model using either [Python Tools](#downloading-with-python-tool) or [git-lfs Tools](#downloading-with-git-lfs-tool).

### Downloading with Python Tool

Execute the following Python script to download the [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) weights and files from [Hugging Face](https://huggingface.co/):

```python
from openmind_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-32B-Instruct",
    local_dir="/path/to/save/Qwen2.5-32B-Instruct",
    local_dir_use_symlinks=False
)
```

`local_dir` is the user-specified path to save the model. Ensure sufficient disk space is available.

### Downloading with git-lfs Tool

Run the following command to verify if [git-lfs](https://git-lfs.com) is available:

```bash
git lfs install
```

If available, the following output will be displayed:

```text
Git LFS initialized.
```

If the tool is unavailable, install [git-lfs](https://git-lfs.com) first. Refer to [git-lfs installation](../../../faqs/faqs.md#git-lfs-installation) guidance in the [FAQ](../../../faqs/faqs.md) section.

Once confirmed, execute the following command to download the weights:

```bash
git clone https://huggingface.co/Qwen/Qwen2.5-32B-Instruct
```

## Setting Environment Variables

For [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct), the following environment variables configure memory allocation, backend, and model-related YAML files:

```bash
#set environment variables
export VLLM_MS_MODEL_BACKEND=MindFormers # Use MindSpore TransFormers as the model backend.
export MINDFORMERS_MODEL_CONFIG=$YAML_PATH # Set the corresponding MindSpore Transformers model YAML file.
```

Here is an explanation of these environment variables:

- `VLLM_MS_MODEL_BACKEND`: The model backend. Currently supported models and backends are listed in the [Model Support List](../../../user_guide/supported_models/models_list/models_list.md).
- `MINDFORMERS_MODEL_CONFIG`: Model configuration file. User can find the corresponding YAML file in the [MindSpore Transformers repository](https://gitee.com/mindspore/mindformers/tree/r1.7.0/research/qwen2_5). For Qwen2.5-32B, the YAML file is [predict_qwen2_5_32b_instruct.yaml](https://gitee.com/mindspore/mindformers/blob/r1.7.0/research/qwen2_5/predict_qwen2_5_32b_instruct.yaml).

Users can check memory usage with `npu-smi info` and set the NPU cards for inference using the following example (assuming cards 4,5,6,7 are used):

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
```

## Online Inference

vLLM-MindSpore Plugin supports online inference deployment with the OpenAI API protocol. The following section would introduce how to [starting the service](#starting-the-service) and [send requests](#sending-requests) to obtain inference results, using [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) as an example.

### Starting the Service

Use the model `Qwen/Qwen2.5-32B-Instruct` and start the vLLM service with the following command:

```bash
export TENSOR_PARALLEL_SIZE=4
export MAX_MODEL_LEN=1024
python3 -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server --model "Qwen/Qwen2.5-32B-Instruct" --trust_remote_code --tensor-parallel-size $TENSOR_PARALLEL_SIZE --max-model-len $MAX_MODEL_LEN
```

Here, `TENSOR_PARALLEL_SIZE` specifies the number of NPU cards, and `MAX_MODEL_LEN` sets the maximum output token length. User can also set the local model path by `--model` argument.

If the service starts successfully, similar output will be obtained:

```text
INFO:   Started server process [6363]
INFO:   Waiting for application startup.
INFO:   Application startup complete.
```

Additionally, performance metrics will be logged, such as:

```text
Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```

### Sending Requests

Use the following command to send a request, where `prompt` is the model input:

```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen2.5-32B-Instruct", "prompt": "I am", "max_tokens": 20, "temperature": 0}'
```

User needs to ensure that the `"model"` field matches the `--model` in the service startup, and the request can successfully match the model.

If processed successfully, the inference result will be:

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
