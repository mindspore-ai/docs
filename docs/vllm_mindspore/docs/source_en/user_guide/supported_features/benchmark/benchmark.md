# Benchmark

[![View Source](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/user_guide/supported_features/benchmark/benchmark.md)  

The benchmark tool of vLLM MindSpore is inherited from vLLM. You can refer to the [vLLM BenchMark](https://github.com/vllm-project/vllm/blob/main/benchmarks/README.md) documentation for more details. This document introduces [Online Benchmark](#online-benchmark) and [Offline Benchmark](#offline-benchmark). Users can follow the steps to conduct performance tests.  

## Online Benchmark

For single-GPU inference, we take [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) as an example. You can prepare the environment by following the guide [NPU Single-GPU Inference (Qwen2.5-7B)](../../../getting_started/tutorials/qwen2.5_7b_singleNPU/qwen2.5_7b_singleNPU.md#online-inference), then start the online service with the following command:  

```bash
vllm-mindspore serve Qwen/Qwen2.5-7B-Instruct --device auto --disable-log-requests  
```  

For multi-GPU inference, we take [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) as an example. You can prepare the environment by following the guide [NPU Single-Node Multi-GPU Inference (Qwen2.5-32B)](../../../getting_started/tutorials/qwen2.5_32b_multiNPU/qwen2.5_32b_multiNPU.md#online-inference), then start the online service with the following command:  

```bash  
export TENSOR_PARALLEL_SIZE=4
export MAX_MODEL_LEN=1024
python3 -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server --model "Qwen/Qwen2.5-32B-Instruct" --trust_remote_code --tensor-parallel-size $TENSOR_PARALLEL_SIZE --max-model-len $MAX_MODEL_LEN
```  

If the service is successfully started, the following inference result will be returned:

```text
INFO:     Started server process [21349]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Clone the vLLM repository and import the vLLM MindSpore plugin to reuse the benchmark tools:  

```bash  
export VLLM_BRANCH=v0.8.3
git clone https://github.com/vllm-project/vllm.git -b ${VLLM_BRANCH}
cd vllm
sed -i '1i import vllm_mindspore' benchmarks/benchmark_serving.py
```  

Here, $VLLM_BRANCH$ refers to the branch name of vLLM, which needs to be compatible with vLLM MindSpore. For compatibility details, please refer to [here](../../../getting_started/installation/installation.md#version-compatibility).

Execute the test script:  

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

If the test runs successfully, the following results will be returned:  

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

## Offline Benchmark

For offline performance benchmark, take [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) as an example. Prepare the environment by following the guide [NPU Single-GPU Inference (Qwen2.5-7B)](../../../getting_started/tutorials/qwen2.5_7b_singleNPU/qwen2.5_7b_singleNPU.md#offline-inference).  

Clone the vLLM repository and import the vLLM-MindSpore plugin to reuse the benchmark tools:

```bash
export VLLM_BRANCH=v0.8.3
git clone https://github.com/vllm-project/vllm.git -b ${VLLM_BRANCH}
cd vllm
sed -i '1i import vllm_mindspore' benchmarks/benchmark_throughput.py
```  

Here, $VLLM_BRANCH$ refers to the branch name of vLLM, which needs to be compatible with vLLM MindSpore. For compatibility details, please refer to [here](../../../getting_started/installation/installation.md#version-compatibility).

Run the test script with the following command. The script below will start the model automatically, and user does not need to start the model manually:  

```bash  
python3 benchmarks/benchmark_throughput.py \  
    --model Qwen/Qwen2.5-7B-Instruct \  
    --dataset-name sonnet \  
    --dataset-path benchmarks/sonnet.txt \  
    --num-prompts 10
```  

If the test runs successfully, the following results will be returned:  

```text  
Throughput: ... requests/s, ... total tokens/s, ... output tokens/s
Total num prompt tokens:  ...
Total num output tokens:  ...
```
