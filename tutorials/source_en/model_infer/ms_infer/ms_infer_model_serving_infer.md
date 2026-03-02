
# Service-oriented Model Inference

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/tutorials/source_en/model_infer/ms_infer/ms_infer_model_serving_infer.md)

## Background

MindSpore is an AI model development framework that provides efficient model development capabilities. Generally, the following code is used for model inference:

```python

input_str = "I love Beijing, because"

model = Qwen2Model(config)
model.load_weight("/path/to/model")

input_ids = tokenizer(input_str)["input_ids"]

logits = model(input_ids)

next_token = ops.argmax(logits)

generate_text = tokenizer.decode(next_token)

print(generate_text)
```

This model inference mode is simple, but the model and weight need to be reloaded each time inference is performed. As a result, the inference efficiency is low in actual applications. To solve this problem, a model inference backend service is usually deployed to receive inference requests online and send requests to the model for computing. This inference mode is called service-oriented inference. MindSpore does not provide the service-oriented inference capability. If service-oriented inference is required in actual applications, users need to develop a service backend and integrate the related model.

To help users easily deploy out-of-the-box model inference capabilities in the production environment, MindSpore provides full-stack service-oriented model inference capabilities based on the popular vLLM model inference open-source software. Service-oriented inference supports real-time online inference and efficiently improves the overall throughput of model inference and reduces inference costs through efficient user request scheduling.

## Main Features

As an efficient service-oriented model inference backend, it should provide the following capabilities to maximize the deployment and running efficiency of models:

- **Quick startup**: Quick loading and initialization of LLMs are implemented through technologies such as compilation cache and parallel loading, reducing the extra startup overhead caused by the continuous increase of model weights.

- **Batch inference**: A proper batch grouping mechanism is used to implement optimal user experience in the case of massive concurrent requests.

- **Efficient scheduling**: Full and incremental request scheduling is used to address full and incremental inference requirements of LLMs, maximizing resource computing efficiency and improving system throughput.

## Inference Tutorial

MindSpore inference works with the vLLM community solution to provide users with full-stack end-to-end inference service capabilities. The vLLM-MindSpore Plugin implements seamless interconnection of the vLLM community service capabilities in the MindSpore framework. For details, see [vLLM-MindSpore Plugin](https://www.mindspore.cn/vllm_mindspore/docs/en/master/index.html).

This section describes the basic usage of vLLM-MindSpore Plugin service-oriented inference.

### Setting Up the Environment

The vLLM-MindSpore Plugin provides [Docker Installation](https://www.mindspore.cn/vllm_mindspore/docs/en/master/getting_started/installation/installation.html#docker-installation) and [Source Code Installation](https://www.mindspore.cn/vllm_mindspore/docs/en/master/getting_started/installation/installation.html#source-code-installation) for users to do installation. The belows are steps for docker installation:

**Building the Image**
User can execute the following commands to clone the vLLM-MindSpore Plugin code repository and build the image:

```bash  
git clone https://atomgit.com/mindspore/vllm-mindspore.git
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

**Creating a Container**

After building the image, set `DOCKER_NAME` and `IMAGE_NAME` as the container and image names, then execute the following command to create the container:  

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

After successfully creating the container, the container ID will be returned. User can verify the creation by executing the following command:

```bash  
docker ps
```  

**Entering the Container**

After creating the container, user can start and enter the container, using the environment variable `DOCKER_NAME`:

```bash  
docker exec -it $DOCKER_NAME bash
```

### Preparing a Model

The service-oriented vLLM-MindSpore Plugin supports the direct running of the native Hugging Face model. Therefore, users can directly download the model from the [Hugging Face](https://huggingface.co/). The following uses the [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) model as an example:

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
```

If `git lfs install` fails during the pull process, refer to the vLLM-MindSpore Plugin [FAQ](https://www.mindspore.cn/vllm_mindspore/docs/en/master/faqs/faqs.html) for a solution.

### Starting a Service

Before launching the model, user need to set the following environment variables:  

```bash
export VLLM_MS_MODEL_BACKEND=MindFormers # use MindSpore Transformers as model backend.
```

Here is an explanation of these environment variables:

- `VLLM_MS_MODEL_BACKEND`: The backend of the model to run. User could find supported models and backends for vLLM-MindSpore Plugin in the [Model Support List](https://www.mindspore.cn/vllm_mindspore/docs/en/master/user_guide/supported_models/models_list/models_list.html).

vLLM-MindSpore Plugin supports online inference deployment with the OpenAI API protocol. Users can run the following command to start the vLLM-MindSpore Plugin online inference service:

```bash
nohup vllm-mindspore serve /path/to/save/Qwen2.5-7B-Instruct &
```

User can also pass the local model path to `vllm-mindspore serve` as model tag. If the service starts successfully, similar output will be obtained:

```text  
INFO:   Started server process [6363]
INFO:   Waiting for application startup.
INFO:   Application startup complete.
```  

Additionally, performance metrics will be logged, such as:

```text  
Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```

### Sending a Request

Use the following command to send a request, where `prompt` is the model input:

```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "prompt": "I am", "max_tokens": 20, "temperature": 0}'
```

User needs to ensure that the `"model"` field matches the model tag in the service startup, and the request can successfully match the model.

If the request is processed successfully, the following inference result will be returned:

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