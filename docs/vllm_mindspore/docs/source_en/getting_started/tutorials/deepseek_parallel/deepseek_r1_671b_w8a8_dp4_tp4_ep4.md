# Multi-machine Parallel Inference (DeepSeek R1)

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/getting_started/tutorials/deepseek_parallel/deepseek_r1_671b_w8a8_dp4_tp4_ep4.md)  

vLLM-MindSpore Plugin supports hybrid parallel inference with configurations of tensor parallelism (TP), data parallelism (DP), expert parallelism (EP), and their combinations. For the applicable scenarios of different parallel strategies, refer to the [vLLM official documentation](https://docs.vllm.ai/en/latest/configuration/optimization.html#parallelism-strategies).  

This document uses the DeepSeek R1 671B W8A8 model as an example to introduce the inference workflows for [tensor parallelism (TP16)](#tp16-tensor-parallel-inference) and [hybrid parallelism](#hybrid-parallel-inference). The DeepSeek R1 671B W8A8 model requires multiple nodes to run inference. To ensure consistent execution configurations (including model configuration file paths, Python environments, etc.) across all nodes, it is recommended to use Docker containers to eliminate execution differences.  

Users can configure the environment by following the [Docker Installation](#docker-installation) section below.  

## Docker Installation

In this section, we recommend to use docker to deploy the vLLM-MindSpore Plugin environment. The following sections are the steps for deployment:

### Building the Image  

User can execute the following commands to clone the vLLM-MindSpore Plugin code repository and build the image:

```bash  
git clone https://gitee.com/mindspore/vllm-mindspore.git
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

### Entering the Container

After [building the image](#building-the-image) step, use the predefined environment variable `DOCKER_NAME` to start and enter the container:  

```bash  
docker exec -it $DOCKER_NAME bash
```  

## Downloading Model Weights

User can download the model using either [Python Tool](#downloading-with-python-tool) or [git-lfs Tool](#downloading-with-git-lfs-tool).  

### Downloading with Python Tool

Execute the following Python script to download the MindSpore-compatible DeepSeek-R1 W8A8 weights and files from [Modelers Community](https://modelers.cn):

```python  
from openmind_hub import snapshot_download
snapshot_download(repo_id="MindSpore-Lab/DeepSeek-R1-0528-A8W8FA3",
                  local_dir="/path/to/save/deepseek_r1_0528_a8w8fa3",
                  local_dir_use_symlinks=False)
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

```shell  
git clone https://modelers.cn/models/MindSpore-Lab/DeepSeek-R1-0528-A8W8.git  
```  

## TP16 Tensor Parallel Inference

vLLM manages and runs multi-node resources through Ray. This example corresponds to a scenario with Tensor Parallelism (TP) set to 16.

### Setting Environment Variables

Environment variables must be set before creating the Ray cluster. If the environment changes, stop the cluster with `ray stop` and recreate it; otherwise, the environment variables will not take effect.

Configure the following environment variables on the master and worker nodes:  

```bash  
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export GLOO_SOCKET_IFNAME=enp189s0f0
export HCCL_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
export MS_ENABLE_LCCL=off
export HCCL_OP_EXPANSION_MODE=AIV
export MS_ALLOC_CONF=enable_vmm:true
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_MS_MODEL_BACKEND=MindFormers
```  

Environment variable descriptions:  

- `GLOO_SOCKET_IFNAME`: GLOO backend port. Use `ifconfig` to find the network interface name corresponding to the IP.  
- `HCCL_SOCKET_IFNAME`: Configure the HCCL port. Use `ifconfig` to find the network interface name corresponding to the IP.  
- `TP_SOCKET_IFNAME`: Configure the TP port. Use `ifconfig` to find the network interface name corresponding to the IP.  
- `MS_ENABLE_LCCL`: Disable LCCL and enable HCCL communication.  
- `HCCL_OP_EXPANSION_MODE`: Configure the communication algorithm expansion location to the AI Vector Core (AIV) computing unit on the device side.  
- `MS_ALLOC_CONF`: Set the memory policy. Refer to the [MindSpore documentation](https://www.mindspore.cn/docs/en/master/api_python/env_var_list.html).  
- `ASCEND_RT_VISIBLE_DEVICES`: Configure the available device IDs for each node. Use the `npu-smi info` command to check.  
- `VLLM_MS_MODEL_BACKEND`: The backend of the model to run. Currently supported models and backends for vLLM-MindSpore Plugin can be found in the [Model Support List](../../../user_guide/supported_models/models_list/models_list.md).  

Additionally, users need to ensure that MindSpore Transformers is installed. Users can add it by running the following command:  

```bash  
export PYTHONPATH=/path/to/mindformers:$PYTHONPATH  
```  

This will include MindSpore Transformers in the Python path.

### Starting Ray for Multi-Node Cluster Management

On Ascend, the pyACL package must be installed to adapt Ray. Additionally, the CANN dependency versions on all nodes must be consistent.  

#### Installing pyACL

pyACL (Python Ascend Computing Language) encapsulates AscendCL APIs via CPython, enabling management of Ascend AI processors and computing resources.  

In the corresponding environment, obtain the Ascend-cann-nnrt installation package for the required version, extract the pyACL dependency package, install it separately, and add the installation path to the environment variables:  

```shell  
./Ascend-cann-nnrt_8.0.RC1_linux-aarch64.run --noexec --extract=./
cd ./run_package
./Ascend-pyACL_8.0.RC1_linux-aarch64.run --full --install-path=<install_path>
export PYTHONPATH=<install_path>/CANN-<VERSION>/python/site-packages/:$PYTHONPATH
```

If you encounter permission issues during installation, you can grant permissions using:  

```bash  
chmod -R 777 ./Ascend-pyACL_8.0.RC1_linux-aarch64.run  
```

Download the Ascend runtime package from the [Ascend homepage](https://www.hiascend.cn/developer/download/community/result?module=cann&version=8.0.RC1.beta1).  

#### Multi-Node Cluster

Before managing a multi-node cluster, ensure that the hostnames of all nodes are unique. If they are the same, set different hostnames using `hostname <new-host-name>`.  

1. Start the master node: `ray start --head --port=<port-to-ray>`. After successful startup, the connection method for worker nodes will be displayed. For example, running `ray start --head --port=6379` on a node with IP `192.5.5.5` will display:  

    ```text
    Local node IP: 192.5.5.5

    --------------------
    Ray runtime started.
    --------------------

    Next steps
      To add another node to this Ray cluster, run
        ray start --address='192.5.5.5:6379'

      To connect to this Ray cluster:
        import ray
        ray.init()

      To terminate the Ray runtime, run
        ray stop

      To view the status of the cluster, use
        ray status
    ```

2. Connect worker nodes to the master node: `ray start --address=<head_node_ip>:<port>`.  
3. Check the cluster status with `ray status`. If the total number of NPUs displayed matches the sum of all nodes, the cluster is successfully created.  

    For example, with two nodes, each with 8 NPUs, the output will be:  

   ```shell
   ======== Autoscaler status: 2025-05-19 00:00:00.000000 ========
   Node status
   ---------------------------------------------------------------
   Active:
    1 node_efa0981305b1204810c3080c09898097099090f09ee909d0ae12545
    1 node_184f44c4790135907ab098897c878699d89098e879f2403bc990112
   Pending:
    (no pending nodes)
   Recent failures:
    (no failures)

   Resources
   ---------------------------------------------------------------
   Usage:
    0.0/384.0 CPU
    0.0/16.0 NPU
    0B/2.58TiB memory
    0B/372.56GiB object_store_memory

   Demands:
    (no resource demands)
   ```

### Online Inference

#### Starting the Service

vLLM-MindSpore Plugin can deploy online inference using the OpenAI API protocol. Below is the workflow for launching the service.  

```bash  
# Service launch parameter explanation  
vllm-mindspore serve  
 --model=[Model Config/Weights Path]  
 --quantization [Source of weight quantification] # golden-stick/ascend are optional, respectively indicating that the quantified weights come from the golden-stick or modelslim quantification tools
 --trust-remote-code # Use locally downloaded model files  
 --max-num-seqs [Maximum Batch Size]  
 --max-model-len [Maximum Input/Output Length]  
 --max-num-batched-tokens [Maximum Tokens per Iteration, recommended: 4096]  
 --block-size [Block Size, recommended: 128]  
 --gpu-memory-utilization [GPU Memory Utilization, recommended: 0.9]  
 --tensor-parallel-size [TP Parallelism Degree]  
```  

Execution example:  

```bash  
# Master node:  
vllm-mindspore serve --model="MindSpore-Lab/DeepSeek-R1-0528-A8W8" --quantization ascend --trust-remote-code --max-num-seqs=256 --max_model_len=32768 --max-num-batched-tokens=4096 --block-size=128 --gpu-memory-utilization=0.9 --tensor-parallel-size 16 --distributed-executor-backend=ray
```  

In tensor parallel scenarios, the `--tensor-parallel-size` parameter overrides the `model_parallel` configuration in the model YAML file. User can also set the local model path by `--model` argument.

#### Sending Requests

Use the following command to send requests, where `prompt` is the model input:  

```bash  
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "MindSpore-Lab/DeepSeek-R1-0528-A8W8", "prompt": "I am", "max_tokens": 20, "temperature": 0, "top_p": 1.0, "top_k": 1, "repetition_penalty": 1.0}'  
```

User needs to ensure that the `"model"` field matches the `--model` in the service startup, and the request can successfully match the model.

## Hybrid Parallel Inference

vLLM manages and operates resources across multiple nodes through Ray. This example corresponds to the following parallel strategy:  

- Data Parallelism (DP): 4;  
- Tensor Parallelism (TP): 4;  
- Expert Parallelism (EP): 4.

### Setting Environment Variables

Configure the following environment variables on the master and worker nodes:  

```bash  
source /usr/local/Ascend/ascend-toolkit/set_env.sh  

export MS_ENABLE_LCCL=off  
export HCCL_OP_EXPANSION_MODE=AIV  
export MS_ALLOC_CONF=enable_vmm:true  
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  
export VLLM_MS_MODEL_BACKEND=MindFormers
```  

Environment variable descriptions:  

- `MS_ENABLE_LCCL`: Disable LCCL and enable HCCL communication.  
- `HCCL_OP_EXPANSION_MODE`: Configure the communication algorithm expansion location to the AI Vector Core (AIV) computing unit on the device side.  
- `MS_ALLOC_CONF`: Set the memory policy. Refer to the [MindSpore documentation](https://www.mindspore.cn/docs/en/master/api_python/env_var_list.html).  
- `ASCEND_RT_VISIBLE_DEVICES`: Configure the available device IDs for each node. Use the `npu-smi info` command to check.  
- `VLLM_MS_MODEL_BACKEND`: The backend of the model to run. Currently supported models and backends for vLLM-MindSpore Plugin can be found in the [Model Support List](../../../user_guide/supported_models/models_list/models_list.md).

### Online Inference

#### Starting the Service

`vllm-mindspore` can deploy online inference using the OpenAI API protocol. Below is the workflow for launching the service:  

```bash  
# Parameter explanations for service launch  
vllm-mindspore serve  
 --model=[Model Config/Weights Path]
 --quantization [Source of weight quantification] # golden-stick/ascend are optional, respectively indicating that the quantified weights come from the golden-stick or modelslim quantification tools
 --trust-remote-code # Use locally downloaded model files  
 --max-num-seqs [Maximum Batch Size]  
 --max-model-len [Maximum Input/Output Length]  
 --max-num-batched-tokens [Maximum Tokens per Iteration, recommended: 4096]  
 --block-size [Block Size, recommended: 128]  
 --gpu-memory-utilization [GPU Memory Utilization, recommended: 0.9]  
 --tensor-parallel-size [TP Parallelism Degree]  
 --headless # Required only for worker nodes, indicating no service-side content  
 --data-parallel-size [DP Parallelism Degree]  
 --data-parallel-size-local [DP count on the current service node, sum across all nodes equals data-parallel-size]  
 --data-parallel-start-rank [Offset of the first DP handled by the current service node]  
 --data-parallel-address [Master node communication IP]  
 --data-parallel-rpc-port [Master node communication port]  
 --enable-expert-parallel # Enable expert parallelism
 --additional-config '{"expert_parallel": [EP Parallelism Degree]}'
```

`data-parallel-size` and `tensor-parallel-size` specify the parallel policies for the attn and ffn-dense parts, and `expert_parallel` specifies the parallel policies for the routing experts in the moe part. And it must satisfy that `data-parallel-size * tensor-parallel-size` is divisible by `expert_parallel`.

User can also set the local model path by `--model` argument. The following is an execution example:  

```bash  
# Master node:  
vllm-mindspore serve --model="MindSpore-Lab/DeepSeek-R1-0528-A8W8" --quantization ascend --trust-remote-code --max-num-seqs=256 --max-model-len=32768 --max-num-batched-tokens=4096 --block-size=128 --gpu-memory-utilization=0.9 --tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-start-rank 0 --data-parallel-address 192.10.10.10 --data-parallel-rpc-port 12370 --enable-expert-parallel --additional-config '{"expert_parallel": 4}'

# Worker node:  
vllm-mindspore serve --headless --model="MindSpore-Lab/DeepSeek-R1-0528-A8W8" --quantization ascend --trust-remote-code --max-num-seqs=256 --max-model-len=32768 --max-num-batched-tokens=4096 --block-size=128 --gpu-memory-utilization=0.9 --tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-start-rank 2 --data-parallel-address 192.10.10.10 --data-parallel-rpc-port 12370 --enable-expert-parallel --additional-config '{"expert_parallel": 4}'
```

#### Sending Requests

Use the following command to send requests, where `prompt` is the model input:  

```bash  
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "MindSpore-Lab/DeepSeek-R1-0528-A8W8", "prompt": "I am", "max_tokens": 20, "temperature": 0}'  
```

User needs to ensure that the `"model"` field matches the `--model` in the service startup, and the request can successfully match the model.
