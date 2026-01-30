# Multi-modal Single-Card Inference (Qwen3-VL-8B-Instruct)

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/getting_started/tutorials/qwen3_vl_8b_singleNPU/qwen3_vl_8b_singleNPU.md)  

This document introduces single NPU multimodal inference process by vLLM-MindSpore Plugin. Taking the [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) model as an example, user can configure the environment through the [Docker Installation](#docker-installation) or the [Installation Guide](../../installation/installation.md#installation-guide), and [downloading model weights](#downloading-model-weights). After [setting environment variables](#setting-environment-variables), user can perform [offline inference](#offline-inference) and [online inference](#online-inference) to experience single NPU inference abilities.

## Docker Installation

In this section, we recommend using Docker for quick deployment of the vLLM-MindSpore Plugin environment. Below are the steps for Docker deployment:  

### Building the Image  

User can execute the following commands to clone the vLLM-MindSpore Plugin code repository:

```bash  
git clone https://atomgit.com/mindspore/vllm-mindspore.git
```  

To build the image according to your npu type, follow these steps:

- For Atlas 800I A2:

  ```bash
  bash build_image.sh
  ```

- For Atlas 300I Duo:

  ```bash
  bash build_image.sh -a 310p
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

After [building the image](#building-the-image), set `DOCKER_NAME` and `IMAGE_NAME` as the container and image names. Run the following command to create a new container:

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
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /var/log/npu/:/usr/slog \
        -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
        -v /etc/hccn.conf:/etc/hccn.conf \
        --shm-size="250g" \
        ${IMAGE_NAME} \
        bash
```

For docker run parameters, please refer to the "Running MindSpore Image" section in the [MindSpore Installation Guide](https://www.mindspore.cn/install/en/).

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

User can download the model using either Web or [git-lfs Tool](#downloading-with-git-lfs-tool).  

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
git clone https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
```  

## Setting Environment Variables

For [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), the following environment variables configure memory allocation, backend, and model-related YAML files:  

```bash  
#set environment variables  
export VLLM_MS_MODEL_BACKEND=Native # use Native Model as model backend.
```  

Here is an explanation of these variables:  

- `VLLM_MS_MODEL_BACKEND`: The model backend. Currently supported models and backends are listed in the [Model Support List](../../../user_guide/supported_models/models_list/models_list.md).

User can check memory usage with `npu-smi info` and set the compute card for inference using:  

```bash  
export ASCEND_RT_VISIBLE_DEVICES=0  
```  

## Offline Inference

After setting up the vLLM-MindSpore Plugin environment, user can use the following python code to perform offline inference on the model:

```python
from PIL import Image
import vllm_mindspore # Add this line on the top of script.
from vllm import LLM, SamplingParams

# Sample prompts.
PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
    "\n<|im_start|>user\nDescribe the content of the image"
    "<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"
    "<|im_start|>assistant\n")

image_path = "/path/to/image.jpg"

def pil_image() -> Image.Image:
    return Image.open(image_path)

inputs = [
    {
        "prompt": PROMPT_TEMPLATE,
        "multi_modal_data": {
            "image": pil_image()
        },
    },
]

# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=512)
model_path = "/path/to/save/Qwen3-VL-8B-Instruct"
# Create a LLM
llm = LLM(model=model_path,
          max_model_len=32768,
          gpu_memory_utilization=0.85)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(inputs, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")
```

If offline inference runs successfully, similar results will be obtained:

```text
"Generated text: This image captures a Pallas's cat (also known as the manul or Otocolobus manul) walking through a snowy landscape.\n\nKey features of the cat:\n* Distinctive Appearance: The cat has a stocky, robust build with a very thick, dense coat of fur that is a mix of gray, brown, and buff colors. This thick fur, dusted with snowflakes, is a key adaptation for its cold, high-altitude habitat.\n* Round Head and Face: It has a wide, rounded head and a remarkably flattened face with short, broad muzzle and large, prominent ears.\n* Gaze: The cat is looking down and to its left, seemingly focused on something in the snow.\n* Posture: It is walking or stalking with one front paw lifted, indicating movement.\n* Paws: Its paws are large and furry, which help it walk on snow.\n\nEnvironment:\n* The cat is on a blanket of fresh white snow.\n* The background is composed of the characteristic white, peeling bark of birch trees and a dark, possibly wrought iron, fence.\n* The lighting suggests an overcast day, which is common in winter mountain environments where this species lives.\n\nIn summary, the image provides a clear, naturalistic view of a Pallas's cat navigating its snowy, wooded habitat, showcasing its unique and endearing physical characteristics."
```

## Online Inference

vLLM-MindSpore Plugin supports online inference deployment with the OpenAI API protocol. The following section would introduce how to [starting the service](#starting-the-service) and [send requests](#sending-requests) to obtain inference results, using [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) as an example.

### Starting the Service

Start the vLLM service with the following command:

```bash
nohup vllm-mindspore serve /path/to/save/Qwen3-VL-8B-Instruct --max-model-len 32768 --gpu-memory-utilization 0.85 &
```

User can also set the local model path as model tag. If the service starts successfully, similar output will be obtained:

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

Use the Python command to send a request, where `prompt` is the model input:

```python
import base64
import requests
from concurrent.futures import ThreadPoolExecutor

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def send_request():
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    print(response.json())

image_path = "/path/to/image.jpg"

base64_image = encode_image(image_path)

# make a request
payload = {
    "model": "/home/ckpt/Qwen3-VL-8B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                },
                {"type": "text", "text": "Describe the content of the image"}
            ]
        }
    ],
    "max_tokens": 512,
}

send_request()
```

User needs to ensure that the `"model"` field matches the model tag in the service startup, and the request can successfully match the model.

If online inference runs successfully, similar results will be obtained:

```text
{'id': 'chatcmpl-0f0a85dcbe7343f89539200e1a201e04', 'object': 'chat.completion', 'created': 1769408795, 'model': '/home/ckpt/Qwen3-VL-8B-Instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'This is a photograph of a Pallas's cat (also known as a "manul") walking through a snowy landscape.\n\nHere are the key details:\n\n* Subject: The central focus is a Pallas's cat, a small wild feline native to Central Asia. It is covered in thick, fluffy fur that is a mix of gray, brown, and tan, with distinct dark markings around its eyes and on its cheeks.\n* Action: The cat is captured mid-stride, walking forward through the snow. Its body is low to the ground, and its front left paw is lifted, indicating movement.\n* Environment: The setting is a winter scene. The ground is covered in white snow, and the background consists of the distinctive white, peeling bark of birch trees. There are also some dark, vertical elements in the background, possibly a fence or another structure.\n* Atmosphere: The image conveys a sense of quiet wilderness. The cat's thick fur is dusted with snow, suggesting it has been moving through the snow for some time. The overall mood is calm and natural.\n\nThe Pallas's cat's unique, somewhat "pug-faced" appearance, with its large, rounded ears and dense coat, is clearly visible, making it a striking and memorable subject.', 'refusal': None, 'annotations': None, 'audio': None, 'function_call': None, 'tool_calls': [], 'reasoning_content': None}, 'logprobs': None, 'finish_reason': 'stop', 'stop_reason': None, 'token_ids': None}], 'service_tier': None, 'system_fingerprint': None, 'usage': {'prompt_tokens': 646, 'total_tokens': 917, 'completion_tokens': 271, 'prompt_tokens_details': None}, 'prompt_logprobs': None, 'prompt_token_ids': None, 'kv_transfer_params': None}
```
