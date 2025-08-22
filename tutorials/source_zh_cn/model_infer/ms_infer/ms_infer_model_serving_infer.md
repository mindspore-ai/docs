
# 服务化模型推理

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/model_infer/ms_infer/ms_infer_model_serving_infer.md)

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

- **batch推理**：合理的组batch机制，实现海量并发请求时最优的用户体验。

- **高效调度**：面向大语言模型的全量和增量推理特性，通过全量和增量请求调度，最大化资源计算效能，提升系统吞吐。

## 推理教程

MindSpore推理结合vLLM社区方案，为用户提供了全栈端到端的推理服务化能力，通过vLLM-MindSpore插件适配层，实现vLLM社区的服务化能力在MindSpore框架下的无缝对接，具体可以参考[vLLM-MindSpore插件文档](https://www.mindspore.cn/vllm_mindspore/docs/zh-CN/master/index.html)。

本章主要简单介绍vLLM-MindSpore插件服务化推理的基础使用。

### 环境准备

vLLM-MindSpore插件适配层提供了环境安装脚本，用户可以执行如下命令创建一个vLLM-MindSpore插件的运行环境：

```shell
# download vllm-mindspore code
git clone https://gitee.com/mindspore/vllm-mindspore.git
cd vllm-mindspore

# create conda env
conda create -n vllm-mindspore-py311 python=3.11
conda activate vllm-mindspore-py311

# install extra dependent packages
pip install setuptools_scm
pip install numba

# run install dependences script
bash install_depend_pkgs.sh

# install vllm-mindspore
python setup.py install
```

vLLM-MindSpore插件的运行环境创建后，还需要安装以下依赖包：

- **mindspore**：MindSpore开发框架，模型运行基础。

- **vllm**：vLLM服务化软件。

- **vllm-mindspore**：适配MindSpore框架能力的vLLM插件，运行MindSpore模型必备。

- **msadapter**：MindSpore对接PyTorch的适配层，部分vLLM的功能依赖PyTorch能力，需要MSAdapter进行适配。

- **golden-stick**：MindSpore模型量化框架，如果要使用量化能力，需要安装此软件。

- **mindformers**：MindSpore框架提供的Transformers模型库，用户可以直接使用这些模型，也可以自己对接MindSpore原生的模型。

### 模型准备

vLLM-MindSpore插件服务化支持原生Hugging Face的模型直接运行，因此直接从Hugging Face官网下载模型即可，此处我们仍然以Qwen2-7B-Instruct模型为例。

```shell
git lfs install
git clone https://huggingface.co/Qwen/Qwen2-7B-Instruct
```

若在拉取过程中，执行`git lfs install失败`，可以参考vLLM-MindSpore插件 [FAQ](https://www.mindspore.cn/vllm_mindspore/docs/zh-CN/master/faqs/faqs.html) 进行解决。

### 启动服务

在启动后端服务前，需要按照实际环境设置对应的环境变量。

```shell
# set Ascend CANN tools envs
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_PATH=${ASCEND_HOME_PATH}/../
export ASCEND_RT_VISIBLE_DEVICES=3
export ASCEND_TOTAL_MEMORY_GB=32

# mindspore envs
export MS_ALLOC_CONF=enable_vmm:true
export CPU_AFFINITY=0

# vLLM envs
export VLLM_MODEL_MEMORY_USE_GB=26

# backend envs
export VLLM_MASTER_IP=127.0.0.1
export VLLM_RPC_PORT=12390
export VLLM_HTTP_PORT=8080
unset vLLM_MODEL_BACKEND

# model envs
export MODEL_ID="/path/to/model/Qwen2-7B-Instruct"
```

执行如下命令可以启动vLLM-MindSpore插件的服务后端。

```shell
vllm-mindspore serve --model=${MODEL_ID} --port=${VLLM_HTTP_PORT} --trust_remote_code --max-num-seqs=256 --max_model_len=32768 --max-num-batched-tokens=4096 --block_size=128 --gpu-memory-utilization=0.9 --tensor-parallel-size 1 --data-parallel-size 1 --data-parallel-size-local 1 --data-parallel-start-rank 0  --data-parallel-address ${VLLM_MASTER_IP} --data-parallel-rpc-port ${VLLM_RPC_PORT} &> vllm-mindspore.log &
```

后端服务加载完成后，会显示后端服务监听的端口和提供的接口。

### 发送请求

用户可以通过发送http请求来实现模型推理，具体可以执行如下命令：

```shell
curl http://${VLLM_MASTER_IP}:${VLLM_HTTP_PORT}/v1/completions -H "Content-Type: application/json" -d "{\"model\": \"${MODEL_ID}\", \"prompt\": \"I love Beijing, because\", \"max_tokens\": 128, \"temperature\": 1.0, \"top_p\": 1.0, \"top_k\": 1, \"repetition_penalty\": 1.0}"
```

服务后端收到推理请求后，计算后会返回如下结果：

```json
{
    "id":"cmpl-1c30caf453154b5ab4a579b7b06cea19",
    "object":"text_completion",
    "created":1754103773,
    "model":"/path/to/model/Qwen2-7B-Instruct",
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
