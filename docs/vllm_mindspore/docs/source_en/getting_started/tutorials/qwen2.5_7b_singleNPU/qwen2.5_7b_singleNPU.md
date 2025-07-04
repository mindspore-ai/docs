# Single-Card Inference (Qwen2.5-7B)

[![View Source](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/getting_started/tutorials/qwen2.5_7b_singleNPU/qwen2.5_7b_singleNPU.md)  

This document introduces single NPU inference process by vLLM MindSpore. Taking the [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) model as an example, user can configure the environment through the [Docker Installation](#docker-installation) or the [Installation Guide](../../installation/installation.md#installation-guide), and [downloading model weights](#downloading-model-weights). After [setting environment variables](#setting-environment-variables), user can perform [offline inference](#offline-inference) and [online inference](#online-inference) to experience single NPU inference abilities.  

## Docker Installation

In this section, we recommend using Docker for quick deployment of the vLLM MindSpore environment. Below are the steps for Docker deployment:  

### Pulling the Image

Pull the vLLM MindSpore Docker image by executing the following command:  

```bash  
docker pull hub.oepkgs.net/oedeploy/openeuler/aarch64/mindspore:latest  
```  

During the pull process, user will see the progress of each layer. After successful completion, use can also check the image by running:  

```bash  
docker images  
```  

### Creating a Container

After [pulling the image](#pulling-the-image), set `DOCKER_NAME` and `IMAGE_NAME` as the container and image names, then create the container:  

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

User can download the model using either [Python Tool](#downloading-with-python-tool) or [git-lfs Tool](#downloading-with-git-lfs-tool).  

### Downloading with Python Tool

Execute the following Python script to download the [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) weights and files from [Hugging Face](https://huggingface.co/):  

```python  
from huggingface_hub import snapshot_download  
snapshot_download(  
    repo_id="Qwen/Qwen2.5-7B-Instruct",  
    local_dir="/path/to/save/Qwen2.5-7B-Instruct",  
    local_dir_use_symlinks=False  
)  
```  

`local_dir` is the user-specified model save path. Ensure sufficient disk space is available.  

### Downloading with git-lfs Tool

Run the following command to check if [git-lfs](https://git-lfs.com) is available:  

```bash  
git lfs install  
```  

If available, the following output will be displayed:  

```text  
Git LFS initialized.  
```  

If the tool is unavailable, install [git-lfs](https://git-lfs.com) first. Refer to [git-lfs installation](../../../faqs/faqs.md#git-lfs-installation) guidance in the [FAQ](../../../faqs/faqs.md) section.  

Once confirmed, download the weights by executing the following command:  

```bash  
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct  
```  

## Setting Environment Variables

For [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), the following environment variables configure memory allocation, backend, and model-related YAML files:  

```bash  
#set environment variables  
export ASCEND_TOTAL_MEMORY_GB=64 # Please use `npu-smi info` to check the memory.  
export vLLM_MODEL_BACKEND=MindFormers # use MindFormers as model backend.  
export vLLM_MODEL_MEMORY_USE_GB=32 # Memory reserved for model execution. Set according to the model's maximum usage, with the remaining environment used for kvcache allocation  
export MINDFORMERS_MODEL_CONFIG=$YAML_PATH # Set the corresponding MindSpore Transformers model's YAML file.  
```  

Here is an explanation of these variables:  

- `ASCEND_TOTAL_MEMORY_GB`: The memory size of each compute card. Query using `npu-smi info`, corresponding to `HBM-Usage(MB)` in the results.  
- `vLLM_MODEL_BACKEND`: The model backend. Currently supported models and backends are listed in the [Model Support List](../../../user_guide/supported_models/models_list/models_list.md).  
- `vLLM_MODEL_MEMORY_USE_GB`: Memory reserved for model loading. Adjust this if encountering insufficient memory.  
- `MINDFORMERS_MODEL_CONFIG`: Model configuration file. User can find the corresponding YAML file in the [MindSpore Transformers repository](https://gitee.com/mindspore/mindformers/tree/dev/research/qwen2_5). For Qwen2.5-7B, the YAML file is [predict_qwen2_5_7b_instruct.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/qwen2_5/predict_qwen2_5_7b_instruct.yaml).  

User can check memory usage with `npu-smi info` and set the compute card for inference using:  

```bash  
export NPU_VISIBE_DEVICES=0  
export ASCEND_RT_VISIBLE_DEVICES=$NPU_VISIBE_DEVICES  
```  

## Offline Inference

Taking [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) as an example, user can perform offline inference with the following Python code:  

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
# Generate texts from the prompts.  
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

## Online Inference

vLLM MindSpore supports online serving deployment with the OpenAI API protocol. The following section would introduce how to [starting the service](#starting-the-service) and [send requests](#sending-requests) to obtain inference results, using [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) as an example.  

### Starting the Service

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
Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%  
```  

### Sending Requests

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
            "text":"trying to create a virtual environment for my Python project, but I am encountering some",  
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
