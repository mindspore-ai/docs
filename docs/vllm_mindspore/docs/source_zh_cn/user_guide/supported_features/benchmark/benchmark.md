# 性能测试

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_zh_cn/user_guide/supported_features/benchmark/benchmark.md)

vLLM MindSpore的性能测试能力，继承自vLLM所提供的性能测试能力，详情可参考[vLLM BenchMark](https://github.com/vllm-project/vllm/blob/main/benchmarks/README.md)文档。该文档将介绍[在线性能测试](#在线性能测试)与[离线性能测试](#离线性能测试)，用户可以根据所介绍步骤进行性能测试。

## 在线性能测试

若用户使用单卡推理，以[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)为例，可按照文档[单卡推理（Qwen2.5-7B）](../../../getting_started/tutorials/qwen2.5_7b_singleNPU/qwen2.5_7b_singleNPU.md#在线推理)进行环境准备，设置以下环境变量：

```bash
export ASCEND_TOTAL_MEMORY_GB=64 # Please use `npu-smi info` to check the memory.
export vLLM_MODEL_BACKEND=MindFormers # use MindSpore Transformers as model backend.
export vLLM_MODEL_MEMORY_USE_GB=32 # Memory reserved for model execution. Set according to the model's maximum usage, with the remaining environment used for kvcache allocation
export MINDFORMERS_MODEL_CONFIG=$YAML_PATH # Set the corresponding MindSpore Transformers model's YAML file.
```

并以下命令启动在线服务：

```bash
vllm-mindspore serve Qwen/Qwen2.5-7B-Instruct --device auto --disable-log-requests
```

若使用多卡推理，以[Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) 为例，可按照文档[多卡推理（Qwen2.5-32B）](../../../getting_started/tutorials/qwen2.5_32b_multiNPU/qwen2.5_32b_multiNPU.md#在线推理)进行环境准备，则可用以下命令启动在线服务：

```bash
export TENSOR_PARALLEL_SIZE=4
export MAX_MODEL_LEN=1024
python3 -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server --model "Qwen/Qwen2.5-32B-Instruct" --trust_remote_code --tensor-parallel-size $TENSOR_PARALLEL_SIZE --max-model-len $MAX_MODEL_LEN
```

当返回以下日志时，则服务已成功拉起：

```text
INFO:     Started server process [21349]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

拉取vLLM代码仓，导入vLLM MindSpore插件，复用其中benchmark功能：

```bash
export VLLM_BRANCH=v0.8.3
git clone https://github.com/vllm-project/vllm.git -b ${VLLM_BRANCH}
cd vllm
sed -i '1i import vllm_mindspore' benchmarks/benchmark_serving.py
```

其中，$VLLM_BRANCH$为vLLM的分支名，其需要与vLLM MindSpore相配套。配套关系可以参考[这里](../../../getting_started/installation/installation.md#版本配套)。

执行测试脚本：

```bash
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# single-card, take Qwen2.5-7B as example:
python3 benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --endpoint /v1/chat/completions  \
    --model Qwen/Qwen2.5-7B-Instruct  \
    --dataset-name sharegpt  \
    --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json  \
    --num-prompts 10

# multi-card, take Qwen2.5-32B as example:
python3 benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --endpoint /v1/chat/completions  \
    --model Qwen/Qwen2.5-32B-Instruct  \
    --dataset-name sharegpt  \
    --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json  \
    --num-prompts 10
```

若成功执行测试，则可以返回如下结果：

```text
============ Serving Benchmark Result ============
Successful requests:                     ....
Benchmark duration (s):                  ....
Total input tokens:                      ....
Total generated tokens:                  ....
Request throughput (req/s):              ....
Output token throughput (tok/s):         ....
Total Token throughput (tok/s):          ....
---------------Time to First Token----------------
Mean TTFT (ms):                          ....
Median TTFT (ms):                        ....
P99 TTFT (ms):                           ....
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          ....
Median TPOT (ms):                        ....
P99 TPOT (ms):                           ....
---------------Inter-token Latency----------------
Mean ITL (ms):                           ....
Median ITL (ms):                         ....
P99 ITL (ms):                            ....
==================================================
```

## 离线性能测试

用户使用离线性能测试时，以[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)为例，可按照文档[单卡推理（Qwen2.5-7B）](../../../getting_started/tutorials/qwen2.5_7b_singleNPU/qwen2.5_7b_singleNPU.md#离线推理)进行环境准备，设置以下环境变量：

```bash
export ASCEND_TOTAL_MEMORY_GB=64 # Please use `npu-smi info` to check the memory.
export vLLM_MODEL_BACKEND=MindFormers # use MindSpore Transformers as model backend.
export vLLM_MODEL_MEMORY_USE_GB=32 # Memory reserved for model execution. Set according to the model's maximum usage, with the remaining environment used for kvcache allocation
export MINDFORMERS_MODEL_CONFIG=$YAML_PATH # Set the corresponding MindSpore Transformers model's YAML file.
```

并拉取vLLM代码仓，导入vLLM MindSpore插件，复用其中benchmark功能：

```bash
export VLLM_BRANCH=v0.8.3
git clone https://github.com/vllm-project/vllm.git -b ${VLLM_BRANCH}
cd vllm
sed -i '1i import vllm_mindspore' benchmarks/benchmark_throughput.py
```

其中，`VLLM_BRANCH`为vLLM的分支名，其需要与vLLM MindSpore相配套。配套关系可以参考[这里](../../../getting_started/installation/installation.md#版本配套)。

用户可通过以下命令，运行测试脚本。该脚本将启动模型，并执行测试，用户不需要再拉起模型：

```bash
python3 benchmarks/benchmark_throughput.py \  
    --model Qwen/Qwen2.5-7B-Instruct \  
    --dataset-name sonnet \  
    --dataset-path benchmarks/sonnet.txt \  
    --num-prompts 10
```

若成功执行测试，则可以返回如下结果：

```text
Throughput: ... requests/s, ... total tokens/s, ... output tokens/s
Total num prompt tokens:  ...
Total num output tokens:  ...
```
