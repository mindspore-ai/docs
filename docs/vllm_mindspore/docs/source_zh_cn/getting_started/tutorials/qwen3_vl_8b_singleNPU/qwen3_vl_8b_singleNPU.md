# 多模态单卡推理（Qwen3-VL-8B-Instruct）

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_zh_cn/getting_started/tutorials/qwen3_vl_8b_singleNPU/qwen3_vl_8b_singleNPU.md)

本文档将介绍使用vLLM-MindSpore插件进行多模态单卡推理的流程。以[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)模型为例，用户可通过以下[docker安装](#docker安装)章节或[安装指南](../../installation/installation.md#安装指南)章节进行环境配置，并[下载模型权重](#下载模型权重)。在[设置环境变量](#设置环境变量)之后，可进行[离线推理](#离线推理)与[在线推理](#在线推理)，体验单卡推理功能。

## docker安装

在本章节中，我们推荐使用docker创建的方式，快速部署vLLM-MindSpore插件环境。以下是部署docker的步骤介绍：

### 构建镜像

用户可执行以下命令，拉取vLLM-MindSpore插件代码仓库：

```bash
git clone https://atomgit.com/mindspore/vllm-mindspore.git
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

关于docker运行参数，可以参考文档：[MindSpore安装指南](https://www.mindspore.cn/install/)的“运行MindSpore镜像”部分。

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

用户可采用Hugging Face网页下载或者[git-lfs工具下载](#git-lfs工具下载)两种方式，进行模型下载。

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
git clone https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
```

## 设置环境变量

以[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)为例，以下环境变量用于设置内存占用、后端以及模型相关的YAML文件：

```bash
#set environment variables
export VLLM_MS_MODEL_BACKEND=Native # use Native Model as model backend.
```

以下是对上述环境变量的解释：

- `VLLM_MS_MODEL_BACKEND`：所运行的模型后端。目前vLLM-MindSpore插件所支持的模型与模型后端，可在[模型支持列表](../../../user_guide/supported_models/models_list/models_list.md)中进行查询。

用户可通过`npu-smi info`查看显存占用情况，并可以使用如下环境变量，设置用于推理的计算卡：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
```

## 离线推理

vLLM-MindSpore插件环境搭建之后，用户可以使用如下Python代码，进行模型的离线推理：

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

若成功执行，则可以获得类似的执行结果：

```text
"Generated text: This image captures a Pallas's cat (also known as the manul or Otocolobus manul) walking through a snowy landscape.\n\nKey features of the cat:\n* Distinctive Appearance: The cat has a stocky, robust build with a very thick, dense coat of fur that is a mix of gray, brown, and buff colors. This thick fur, dusted with snowflakes, is a key adaptation for its cold, high-altitude habitat.\n* Round Head and Face: It has a wide, rounded head and a remarkably flattened face with short, broad muzzle and large, prominent ears.\n* Gaze: The cat is looking down and to its left, seemingly focused on something in the snow.\n* Posture: It is walking or stalking with one front paw lifted, indicating movement.\n* Paws: Its paws are large and furry, which help it walk on snow.\n\nEnvironment:\n* The cat is on a blanket of fresh white snow.\n* The background is composed of the characteristic white, peeling bark of birch trees and a dark, possibly wrought iron, fence.\n* The lighting suggests an overcast day, which is common in winter mountain environments where this species lives.\n\nIn summary, the image provides a clear, naturalistic view of a Pallas's cat navigating its snowy, wooded habitat, showcasing its unique and endearing physical characteristics."
```

## 在线推理

vLLM-MindSpore插件可使用OpenAI的API协议，部署在线推理。以下以[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)为例，介绍模型的[启动服务](#启动服务)和[发送请求](#发送请求)，得到在线推理的推理结果。

### 启动服务

使用如下命令启动vLLM服务：

```bash
nohup vllm-mindspore serve /path/to/save/Qwen3-VL-8B-Instruct --max-model-len 32768 --gpu-memory-utilization 0.85 &
```

用户可以通过指定模型保存的本地路径作为模型标签。若服务成功启动，则可以获得类似的执行结果：

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

使用如下Python脚本发送请求。其中`prompt`字段为模型输入：

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

# 构造请求
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

其中，用户需确认`"model"`字段与启动服务中的模型标签一致，请求才能成功匹配到模型。

若成功执行，则可以获得类似的执行结果：

```text
{'id': 'chatcmpl-0f0a85dcbe7343f89539200e1a201e04', 'object': 'chat.completion', 'created': 1769408795, 'model': '/home/ckpt/Qwen3-VL-8B-Instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'This is a photograph of a Pallas's cat (also known as a "manul") walking through a snowy landscape.\n\nHere are the key details:\n\n* Subject: The central focus is a Pallas's cat, a small wild feline native to Central Asia. It is covered in thick, fluffy fur that is a mix of gray, brown, and tan, with distinct dark markings around its eyes and on its cheeks.\n* Action: The cat is captured mid-stride, walking forward through the snow. Its body is low to the ground, and its front left paw is lifted, indicating movement.\n* Environment: The setting is a winter scene. The ground is covered in white snow, and the background consists of the distinctive white, peeling bark of birch trees. There are also some dark, vertical elements in the background, possibly a fence or another structure.\n* Atmosphere: The image conveys a sense of quiet wilderness. The cat's thick fur is dusted with snow, suggesting it has been moving through the snow for some time. The overall mood is calm and natural.\n\nThe Pallas's cat's unique, somewhat "pug-faced" appearance, with its large, rounded ears and dense coat, is clearly visible, making it a striking and memorable subject.', 'refusal': None, 'annotations': None, 'audio': None, 'function_call': None, 'tool_calls': [], 'reasoning_content': None}, 'logprobs': None, 'finish_reason': 'stop', 'stop_reason': None, 'token_ids': None}], 'service_tier': None, 'system_fingerprint': None, 'usage': {'prompt_tokens': 646, 'total_tokens': 917, 'completion_tokens': 271, 'prompt_tokens_details': None}, 'prompt_logprobs': None, 'prompt_token_ids': None, 'kv_transfer_params': None}
```
