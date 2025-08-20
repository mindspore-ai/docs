
# Service-oriented Model Inference

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.7.0/tutorials/source_en/model_infer/ms_infer/ms_infer_model_serving_infer.md)

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

This model inference mode is simple, but the model and weight need to be reloaded each time inference is performed. As a result, the inference efficiency is low in actual applications. To solve this problem, a model inference backend service is usually deployed to receive inference requests online and send requests to the model for computing. This inference mode is called service-oriented inference. MindSpore does not provide the service-oriented inference capability. If service-oriented inference is required in actual applications, you need to develop a service backend and integrate the related model.

To help users easily deploy out-of-the-box model inference capabilities in the production environment, MindSpore provides full-stack service-oriented model inference capabilities based on the popular vLLM model inference open-source software. Service-oriented inference supports real-time online inference and efficiently improves the overall throughput of model inference and reduces inference costs through efficient user request scheduling.

## Main Features

As an efficient service-oriented model inference backend, it should provide the following capabilities to maximize the deployment and running efficiency of models:

- **Quick startup**: Quick loading and initialization of LLMs are implemented through technologies such as compilation cache and parallel loading, reducing the extra startup overhead caused by the continuous increase of model weights.

- **Batch inference**: A proper batch grouping mechanism is used to implement optimal user experience in the case of massive concurrent requests.

- **Efficient scheduling**: Full and incremental request scheduling is used to address full and incremental inference requirements of LLMs, maximizing resource computing efficiency and improving system throughput.

## Inference Tutorial

MindSpore inference works with the vLLM community solution to provide users with full-stack end-to-end inference service capabilities. The vLLM-MindSpore Plugin adaptation layer implements seamless interconnection of the vLLM community service capabilities in the MindSpore framework. For details, see [vLLM-MindSpore Plugin](https://www.mindspore.cn/vllm_mindspore/docs/en/master/index.html).

This section describes the basic usage of vLLM-MindSpore Plugin service-oriented inference.

### Setting Up the Environment

The vLLM-MindSpore Plugin adaptation layer provides an environment installation script. You can run the following commands to create a vLLM-MindSpore Plugin operating environment:

```shell
# download vllm-mindspore code
git clone -b r0.3.0 https://gitee.com/mindspore/vllm-mindspore.git
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

After the vLLM-MindSpore Plugin operating environment is created, you need to install the following dependency packages:

- **mindspore**: MindSpore development framework, which is the basis for model running.

- **vllm**: vLLM service software.

- **vllm-mindspore**: vLLM extension that adapts to the MindSpore framework. It is required for running MindSpore models.

- **msadapter**: adaptation layer for MindSpore to connect to PyTorch. Some vLLM functions depend on the PyTorch capabilities and need to be adapted by MSAdapter.

- **golden-stick**: MindSpore model quantization framework. If the quantization capability is required, install this software.

- **mindformers**: Transformer model library provided by the MindSpore framework. You can use the models directly or connect to the native models of MindSpore.

### Preparing a Model

The service-oriented vLLM-MindSpore Plugin supports the direct running of the native Hugging Face model. Therefore, you can directly download the model from the Hugging Face official website. The following uses the Qwen2-7B-Instruct model as an example:

```shell
git lfs install
git clone https://huggingface.co/Qwen/Qwen2-7B-Instruct
```

If `git lfs install` fails during the pull process, refer to the vLLM-MindSpore Plugin [FAQ](https://www.mindspore.cn/vllm_mindspore/docs/en/master/faqs/faqs.html) for a solution.

### Starting a Service

Before starting the backend service, you need to set the environment variables based on the actual environment.

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

Run the following command to start the vLLM-MindSpore Plugin service backend:

```shell
vllm-mindspore serve --model=${MODEL_ID} --port=${VLLM_HTTP_PORT} --trust_remote_code --max-num-seqs=256 --max_model_len=32768 --max-num-batched-tokens=4096 --block_size=128 --gpu-memory-utilization=0.9 --tensor-parallel-size 1 --data-parallel-size 1 --data-parallel-size-local 1 --data-parallel-start-rank 0  --data-parallel-address ${VLLM_MASTER_IP} --data-parallel-rpc-port ${VLLM_RPC_PORT} &> vllm-mindspore.log &
```

After the backend service is loaded, the listening port and provided APIs of the backend service are displayed.

### Sending a Request

You can run the following command to send an HTTP request to implement model inference:

```shell
curl http://${VLLM_MASTER_IP}:${VLLM_HTTP_PORT}/v1/completions -H "Content-Type: application/json" -d "{\"model\": \"${MODEL_ID}\", \"prompt\": \"I love Beijing, because\", \"max_tokens\": 128, \"temperature\": 1.0, \"top_p\": 1.0, \"top_k\": 1, \"repetition_penalty\": 1.0}"
```

After receiving the inference request, the service backend calculates and returns the following results:

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
