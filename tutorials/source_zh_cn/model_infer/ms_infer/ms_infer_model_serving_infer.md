
# 服务化模型推理

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.1/tutorials/source_zh_cn/model_infer/ms_infer/ms_infer_model_serving_infer.md)

## 特性背景

MindSpore作为AI模型开发框架，可以提供模型的高效开发能力，通常我们会用下面的代码进行模型推理：

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

这种模型推理方式比较简单，但是每次推理需要重新加载模型和权重，在实际应用中使用效率较低。为解决这一问题，通常会部署一个模型推理后端服务，在线接收用户的推理请求，并将请求发给模型进行计算，这种推理方式被称为服务化推理。MindSpore本身不提供服务化推理能力，如果要在实际应用中实现服务化推理，需要自行开发服务后端并集成相关模型。

为了帮助用户更便捷地在生产环境中部署“开箱即用”的模型推理能力，MindSpore结合当前流行的vLLM模型推理开源软件，提供全栈服务化模型推理能力。服务化推理不仅支持实时在线推理，还能通过高效的用户请求调度，有效地提高模型推理整体吞吐，降低推理成本。

## 主要特性

作为一个高效的服务化模型推理后端，应该提供以下能力，以最大化提升模型的部署和运行效率：

- **快速启动**：通过编译缓存、并行加载等技术，实现大语言模型快速加载和初始化，减少模型权重不断增大带来的额外启动开销。

- **Batch推理**：合理的批处理机制，实现海量并发请求时最优的用户体验。

- **高效调度**：面向大语言模型的全量和增量推理特性，通过全量和增量请求调度，最大化资源计算效能，提升系统吞吐量。

## 推理教程

MindSpore推理结合vLLM社区方案，为用户提供了全栈端到端的推理服务化能力，通过vLLM-MindSpore插件实现vLLM社区的服务化能力在MindSpore框架下的无缝对接，具体可以参考[vLLM-MindSpore插件文档](https://www.mindspore.cn/vllm_mindspore/docs/zh-CN/master/index.html)。

本章主要简单介绍vLLM-MindSpore插件服务化推理的基础使用。

### 环境准备

vLLM-MindSpore插件提供了[docker安装](https://www.mindspore.cn/vllm_mindspore/docs/zh-CN/master/getting_started/installation/installation.html#docker%E5%AE%89%E8%A3%85)与[源码安装](https://www.mindspore.cn/vllm_mindspore/docs/zh-CN/master/getting_started/installation/installation.html#%E6%BA%90%E7%A0%81%E5%AE%89%E8%A3%85)的方式，让用户可以便捷地安装使用vLLM-MindSpore插件。以下是部署docker的步骤介绍：

**构建镜像**

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

**新建容器**

用户在构建镜像后，设置`DOCKER_NAME`与`IMAGE_NAME`为容器名与镜像名，并执行以下命令新建容器：

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

**进入容器**

用户在新建容器后，使用已定义的环境变量`DOCKER_NAME`，启动并进入容器：

```bash
docker exec -it $DOCKER_NAME bash
```

### 模型准备

vLLM-MindSpore插件服务化支持原生Hugging Face的模型直接运行，因此直接从[Hugging Face社区](https://huggingface.co/)下载模型即可，此处我们以[Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B)模型为例。

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen2-7B
```

若在拉取过程中，执行`git lfs install失败`，可以参考vLLM-MindSpore插件 [FAQ](https://www.mindspore.cn/vllm_mindspore/docs/zh-CN/master/faqs/faqs.html) 进行解决。

### 启动服务

在启动后端服务前，需要设置对应的环境变量。

```bash
export vLLM_MS_MODEL_BACKEND=MindFormers # use MindSpore Transformers as model backend.
```

以下是对上述环境变量的解释：

- `vLLM_MS_MODEL_BACKEND`：所运行的模型后端。目前vLLM-MindSpore插件所支持的模型与模型后端，可在[模型支持列表](https://www.mindspore.cn/vllm_mindspore/docs/zh-CN/master/user_guide/supported_models/models_list/models_list.html)与[环境变量清单](https://www.mindspore.cn/vllm_mindspore/docs/zh-CN/master/user_guide/environment_variables/environment_variables.html)中进行查询。

另外，用户需要确保MindSpore Transformers已安装。用户可通过以下方式引入MindSpore Transformers：

```bash
export PYTHONPATH=/path/to/mindformers:$PYTHONPATH
```

vLLM-MindSpore插件可使用OpenAI的API协议，进行在线推理部署。执行如下命令，启动vLLM-MindSpore插件的在线推理服务：

```bash
vllm-mindspore serve --model=/path/to/model/Qwen2-7B --trust_remote_code --max-num-seqs=256 --max-model-len=32768 --max-num-batched-tokens=4096 --block_size=128 --gpu-memory-utilization=0.9
```

用户可以通过`--model`参数，指定模型保存的本地路径。若服务成功启动，则可以获得类似的执行结果：

```text
INFO:   Started server process [6363]
INFO:   Waiting for application startup.
INFO:   Application startup complete.
```

另外，日志中还会打印服务的性能数据信息，如：

```text
Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```

### 发送请求

使用如下命令发送请求。其中，`prompt`字段为模型输入：

```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "/path/to/model/Qwen2-7B", "prompt": "I love Beijing, because", "max_tokens": 128, "temperature": 1.0, "top_p": 1.0, "top_k": 1, "repetition_penalty": 1.0}'
```

其中，用户需确认`"model"`字段与启动服务中的`--model`一致，请求才能成功匹配到模型。若请求处理成功，将获得以下推理结果：

```text
{
    "id":"cmpl-1c30caf453154b5ab4a579b7b06cea19",
    "object":"text_completion",
    "created":1754103773,
    "model":"/path/to/model/Qwen2-7B",
    "choices":[
        {
            "index":0,
            "text":" it is a city with a long history and rich culture. I have been to many places of interest in Beijing, such as the Great Wall, the Forbidden City, the Summer Palace, and the Temple of Heaven. I also visited the National Museum of China, where I learned a lot about Chinese history and culture. The food in Beijing is also amazing, especially the Peking duck and the dumplings. I enjoyed trying different types of local cuisine and experiencing the unique flavors of Beijing. The people in Beijing are friendly and welcoming, and they are always willing to help tourists. I had a great time exploring the city and interacting with the locals",
            "logprobs":null,
            "finish_reason":"length",
            "stop_reason":null,
            "prompt_logprobs":null
        }
    ],
    "usage":{
        "prompt_tokens":5,
        "total_tokens":133,
        "completion_tokens":128,
        "prompt_tokens_details":null
    }
}
```
