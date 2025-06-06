# Quick Start  

[![View Source](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/getting_started/quick_start/quick_start.md)  

This document provides a quick guide to deploy vLLM MindSpore by [docker](https://www.docker.com/), with the [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) model as an example. User can quickly experience the serving and inference abilities of vLLM MindSpore by [offline inference](#offline-inference) and [online serving](#online-serving). For more information about installation, please refer to the [Installation Guide](../installation/installation.md).  

## Docker Installation  

In this section, we recommend to use docker to deploy the vLLM MindSpore environment. The following sections are the steps for deployment:

### Pulling the Image  

Pull the vLLM MindSpore docker image by executing the following command:  

```bash  
docker pull hub.oepkgs.net/oedeploy/openeuler/aarch64/mindspore:latest  
```  

During the pull process, user will see the progress of each layer of the docker image. User can verify the image by executing the following command:  

```bash  
docker images  
```  

### Creating a Container  

After [pulling the image](#pulling-the-image), set `DOCKER_NAME` and `IMAGE_NAME` as the container and image names, and create the container by running:  

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

After successfully creating the container, the container ID will be returned. User can verify the creation by executing the following command:  

```bash  
docker ps  
```  

### Entering the Container  

After [creating the container](#creating-a-container), use the environment variable `DOCKER_NAME` to start and enter the container by executing the following command:  

```bash  
docker exec -it $DOCKER_NAME bash  
```  

## Using the Service

After deploying the environment, user need to prepare the model files before running the model. Refer to the [Download Model](#downloading-model) section for guidance. After [setting environment variables](#setting-environment-variables), user can experience the model bt [offline inference](#offline-inference) or [online serving](#online-serving).  

### Downloading Model

User can download the model using either the [Python Tool](#downloading-with-python-tool) or [git-lfs Tool](#downloading-with-git-lfs-tool).  

#### Downloading with Python Tool  

Execute the following Python script to download the [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) weights and files from [Hugging Face](https://huggingface.co/):  

```python  
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="/path/to/save/Qwen2.5-7B-Instruct",
    local_dir_use_symlinks=False
)
```  

`local_dir` is the model save path specified by the user. Please ensure the disk space is sufficient.  

#### Downloading with git-lfs Tool

Execute the following command to check if [git-lfs](https://git-lfs.com) is available:  

```bash  
git lfs install  
```  

If available, the following output will be displayed:  

```text  
Git LFS initialized.  
```  

If the tool is unavailable, please install [git-lfs](https://git-lfs.com) first. Refer to the [FAQ](../../faqs/faqs.md) section for guidance on [git-lfs installation](../../faqs/faqs.md#git-lfs-installation).  

Once confirmed, download the weights by executing the following command:  

```bash  
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct  
```  

### Setting Environment Variables  

Before launching the model, user need to set the following environment variables:  

```bash  
export ASCEND_TOTAL_MEMORY_GB=64 # Please use `npu-smi info` to check the memory.  
export vLLM_MODEL_BACKEND=MindFormers # use MindSpore Transformers as model backend.  
export vLLM_MODEL_MEMORY_USE_GB=32 # Memory reserved for model execution. Set according to the model's maximum usage, with the remaining environment used for kvcache allocation.  
export MINDFORMERS_MODEL_CONFIG=$YAML_PATH # Set the corresponding MindSpore Transformers model's YAML file.  
```  

Here is an explanation of these environment variables:  

- `ASCEND_TOTAL_MEMORY_GB`: The memory size of each card. User can check the memory by using `npu-smi info`, where the value corresponds to `HBM-Usage(MB)` in the query results.
- `vLLM_MODEL_BACKEND`: The backend of the model to run. User could find supported models and backends for vLLM MindSpore in the [Model Support List](../../user_guide/supported_models/models_list/models_list.md).  
- `vLLM_MODEL_MEMORY_USE_GB`: The memory reserved for model loading. Adjust this value if insufficient memory error occurs during model loading.  
- `MINDFORMERS_MODEL_CONFIG`: The model configuration file.  

### Offline Inference  

Taking [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) as an example, user can perform offline inference with the following Python script:  

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

If offline inference runs successfully, similar results will be obtained:  

```text  
Prompt: 'I am'. Generated text: ' trying to create a virtual environment for my Python project, but I am encountering some'
Prompt: 'Today is'. Generated text: ' the 100th day of school. To celebrate, the teacher has'
Prompt: 'Llama is'. Generated text: ' a 100% natural, biodegradable, and compostable alternative'
```  

### Online Serving  

vLLM MindSpore supports online serving deployment with the OpenAI API protocol. The following section would introduce how to [starting the service](#starting-the-service) and [send requests](#sending-requests) to obtain inference results, using [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) as an example.  

#### Starting the Service  

Use the model `Qwen/Qwen2.5-7B-Instruct` and start the vLLM service with the following command:  

```bash  
python3 -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server --model "Qwen/Qwen2.5-7B-Instruct"
```  

If the service starts successfully, similar output will be obtained:  

```text  
INFO:   Started server process [6363]  
INFO:   Waiting for application startup.  
INFO:   Application startup complete.  
```  

Additionally, performance metrics will be logged, such as:  

```text  
Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg gereration throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```  

#### Sending Requests  

Use the following command to send a request, where `prompt` is the model input:  

```bash  
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "prompt": "I am", "max_tokens": 15, "temperature": 0}'  
```  

If the request is processed successfully, the following inference result will be returned:  

```text  
{  
    "id":"cmpl-5e6e314861c24ba79fea151d86c1b9a6","object":"text_completion",  
    "create":1747398389,  
    "model":"Qwen2.5-7B-Instruct",  
    "choices":[  
        {  
            "index":0,  
            "trying to create a virtual environment for my Python project, but I am encountering some",  
            "logprobs":null,  
            "finish_reason":"length",  
            "stop_reason":null,  
            "prompt_logprobs":null  
        }  
    ],  
    "usage":{  
        "prompt_tokens":2,  
        "total_tokens":17,  
        "completion_tokens":15,  
        "prompt_tokens_details":null  
    }  
}  
```
