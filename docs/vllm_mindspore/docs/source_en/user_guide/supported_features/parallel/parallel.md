# Parallel Inference Methods

[![View Source](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/user_guide/supported_features/parallel/parallel.md)

The vLLM-MindSpore plugin supports hybrid parallel inference configurations combining Tensor Parallelism (TP), Data Parallelism (DP), and Expert Parallelism (EP), and can be launched for multi-node multi-card setups using `Ray` or `multiprocess`. For applicable scenarios of different parallel strategies, refer to the [vLLM Official Documentation](https://docs.vllm.ai/en/latest/configuration/optimization.html#parallelism-strategies). The following sections will detail the usage scenarios, parameter configuration, and [Online Inference](#online-inference) for [Tensor Parallelism](#tensor-parallelism), [Data Parallelism](#data-parallelism), [Expert Parallelism](#expert-parallelism), and [Hybrid Parallelism](#hybrid-parallelism).

## Tensor Parallelism

Tensor Parallelism shards the model weight parameters within each model layer across multiple NPUs. Tensor Parallelism is the recommended strategy for large model inference when the model size exceeds the capacity of a single NPU, or when there is a need to reduce the pressure on a single NPU and free up more space for the KV cache to achieve higher throughput. For more information, see the [Introduction to Tensor Parallelism in vLLM](https://docs.vllm.ai/en/v0.9.1/configuration/optimization.html?h=expert#tensor-parallelism-tp).

### Parameter Configuration

To use Tensor Parallelism (TP), configure the following options in the launch command `vllm-mindspore serve`:

- `--tensor-parallel-size`: The TP parallelism degree.

### Single-Node Example

The following command is an example of launching Tensor Parallelism for Qwen-2.5 on a single node with four cards:

```bash
TENSOR_PARALLEL_SIZE=4       # TP parallelism degree

vllm-mindspore serve /path/to/Qwen2.5/model --trust-remote-code --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
```

### Multi-Node Example

Multi-node Tensor Parallelism relies on Ray for launch. Please refer to [Ray Multi-Node Cluster Management](#ray-multi-node-cluster-management) for Ray environment configuration.

The following command is an example of launching Tensor Parallelism for Qwen-2.5 across two nodes with four cards total:

```bash
# Master Node:

TENSOR_PARALLEL_SIZE=4       # TP parallelism degree
vllm-mindspore serve /path/to/Qwen2.5/model --trust-remote-code --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
```

## Data Parallelism

Data Parallelism fully replicates the model across multiple groups of NPUs, and processes requests from different batches in parallel during inference. Data Parallelism is the recommended strategy for large model inference when there are sufficient NPUs to fully replicate the model, when there is a need to improve throughput rather than scale the model size, or when maintaining isolation between request batches in a multi-user environment is required. Data Parallelism can be combined with other parallel strategies. Please note: MoE (Mixture of Experts) layers will be sharded based on the product of the Tensor Parallelism degree and the Data Parallelism degree. For more information, see the [Introduction to Data Parallelism in vLLM](https://docs.vllm.ai/en/v0.9.1/configuration/optimization.html?h=expert#data-parallelism-dp).

### Parameter Configuration

To use Data Parallelism (DP), configure the following options in the launch command `vllm-mindspore serve`:

- `--data-parallel-size`: The DP parallelism degree.
- `--data-parallel-backend`: Sets the DP deployment method. Options are `mp` and `ray`. By default, DP is deployed using multiprocess:
    - `mp`: Deploy with multiprocess.
    - `ray`: Deploy with Ray.
- When `--data-parallel-backend` is set to `mp`, the following options must also be configured in the launch command:
    - `--data-parallel-size-local`: The number of DP workers on the current service node. The sum across all nodes should equal to `--data-parallel-size`.
    - `--data-parallel-start-rank`: The offset of the first DP worker responsible for the current service node.
    - `--data-parallel-address`: The communication IP address of the master node.
    - `--data-parallel-rpc-port`: The communication port of the master node.

### Service Startup Examples

#### Ray Startup Example (Recommended)

Ray simplifies startup in multi-node scenarios and is the recommended method. Please refer to [Ray Multi-Node Cluster Management](#ray-multi-node-cluster-management) for Ray environment configuration. The following command is an example of launching Data Parallelism for Qwen-2.5 across two nodes with four cards using Ray:

```bash
DATA_PARALLEL_SIZE=4       # DP parallelism degree
DATA_PARALLEL_SIZE_LOCAL=2 # Number of DP workers on the current service node. The sum across all nodes should equal to `--data-parallel-size`

vllm-mindspore serve /path/to/Qwen2.5/model --trust-remote-code --data-parallel-size ${DATA_PARALLEL_SIZE} --data-parallel-size-local ${DATA_PARALLEL_SIZE_LOCAL} --data-parallel-backend=ray
```

#### Multiprocess Startup

When users need to simplify dependencies, they can use multiprocess startup. The following command is an example of launching Data Parallelism for Qwen-2.5 across two nodes with four cards using multiprocess:

```bash
# Master Node:
vllm-mindspore serve /path/to/Qwen2.5/model --trust-remote-code --data-parallel-size ${DATA_PARALLEL_SIZE} --data-parallel-size-local ${DATA_PARALLEL_SIZE_LOCAL}

# Worker Node:
vllm-mindspore serve /path/to/Qwen2.5/model --headless --trust-remote-code --data-parallel-size ${DATA_PARALLEL_SIZE} --data-parallel-size-local ${DATA_PARALLEL_SIZE_LOCAL}
```

## Expert Parallelism

Expert Parallelism is a parallelization form specific to Mixture of Experts (MoE) models, achieved by distributing different expert networks across multiple NPUs. This parallel mode is suitable for MoE models (such as DeepSeekV3, Qwen3-MoE, Llama-4, etc.) and is the recommended strategy for large model inference when there is a need to balance the computational load of expert networks across NPUs. Enable Expert Parallelism by setting `enable_expert_parallel=True` and `--additional-config`. This setting will cause the MoE layers to adopt the Expert Parallelism strategy instead of Tensor Parallelism. The parallelism degree for Expert Parallelism will remain consistent with the already set Tensor Parallelism degree. For more information, see the [Introduction to Expert Parallelism in vLLM](https://docs.vllm.ai/en/v0.9.1/configuration/optimization.html?h=expert#expert-parallelism-ep).

### Parameter Configuration

To use Expert Parallelism (EP), configure the following options in the launch command `vllm-mindspore serve`:

- `--enable-expert-parallel`: Enable Expert Parallelism.

- `--additional-config`: Configure the `expert_parallel` field to set the EP parallelism degree. For example, to configure EP as 4:

  ```bash
  --additional-config '{"expert_parallel": 4}'
  ```

> - If `--enable-expert-parallel` is not configured, EP is not enabled, and configuring `--additional-config '{"expert_parallel": 4}'` will not take effect.
> - If `--enable-expert-parallel` is configured, but `--additional-config '{"expert_parallel": 4}'` is not configured, then the EP parallelism degree equals to the TP parallelism degree multiplied by the DP parallelism degree.
> - If `--enable-expert-parallel` is configured, and `--additional-config '{"expert_parallel": 4}'` is also configured, then the EP parallelism degree equals to `4`.

### Service Startup Examples

#### Single-Node Example

The following command is an example of launching Expert Parallelism for Qwen-3 MOE on a single node with eight cards:

```bash
vllm-mindspore serve /path/to/Qwen3-MOE --trust-remote-code --enable-expert-parallel --additional-config '{"expert_parallel": 8}
```

#### Multi-Node Example

Multi-node Expert Parallelism relies on Ray for launch. Please refer to [Ray Multi-Node Cluster Management](#ray-multi-node-cluster-management) for Ray environment configuration. The following command is an example of launching Expert Parallelism for Qwen-3 MOE across two nodes with eight cards total using Ray:

```bash
vllm-mindspore serve /path/to/Qwen3-MOE --trust-remote-code --enable-expert-parallel --additional-config '{"expert_parallel": 8} --data-parallel-backend ray
```

## Hybrid Parallelism

Users can flexibly combine and adjust parallel strategies based on the model used and the available machine resources. For example, in a DeepSeek-R1 scenario, the following hybrid strategy can be used:

- Tensor Parallelism: 4
- Data Parallelism: 4
- Expert Parallelism: 4

Based on the introductions above, the configurations for the three parallel strategies can be combined and enabled in the `vllm-mindspore serve` launch command. Multi-node Hybrid Parallelism relies on Ray for launch. Please refer to [Ray Multi-Node Cluster Management](#ray-multi-node-cluster-management) for Ray environment configuration. The combined Ray launch command for Hybrid Parallelism is as follows:

```bash
vllm-mindspore serve /path/to/DeepSeek-R1 --trust-remote-code --max-num-seqs=256 --max-model-len=32768 --max-num-batched-tokens=4096 --block-size=128 --gpu-memory-utilization=0.9 --tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-size-local 2 --enable-expert-parallel --additional-config '{"expert_parallel": 4}' --data-parallel-backend ray
```

## Appendix

### Ray Multi-Node Cluster Management

On Ascend, there are two startup methods: multiprocess and Ray. In multi-node scenarios, if using Ray, an additional pyACL package needs to be installed to adapt Ray, and the CANN dependency versions on all nodes must be consistent.

#### Installing pyACL

pyACL (Python Ascend Computing Language) wraps the corresponding API interfaces of AscendCL through CPython. Using these interfaces allows management of Ascend AI processors and their corresponding computing resources.

In the target environment, after obtaining the appropriate version of the Ascend-cann-nnrt installation package, extract the pyACL dependency package and install it separately. Then add the installation path to the environment variables:

```bash
./Ascend-cann-nnrt_*_linux-aarch64.run --noexec --extract=./
cd ./run_package
./Ascend-pyACL_*_linux-aarch64.run --full --install-path=<install_path>
export PYTHONPATH=<install_path>/CANN-<VERSION>/python/site-packages/:$PYTHONPATH
```

If there are permission issues during installation, use the following command to add permissions:

```bash
chmod -R 777 ./Ascend-pyACL_*_linux-aarch64.run
```

The Ascend runtime package can be downloaded from the Ascend homepage. For example, you can refer to [installation](../../../getting_started/installation/installation.md) and download the runtime package.

#### Multi-Node Cluster

Before managing a multi-node cluster, check that the hostnames of all nodes are different. If any are the same, set different hostnames using `hostname <new-host-name>`.

1. Start the head node: `ray start --head --port=<port-to-ray>`. Upon successful startup, the connection method for worker nodes will be displayed. Configure as follows, replacing `IP` and `address` with the actual environment information.

  ```text
  Local node IP: *.*.*.*

  -------------------
  Ray runtime started.
  --------------------

  Next steps
    To add another node to this Ray cluster, run
      ray start --address='*.*.*.*:*'

    To connect to this Ray cluster:
      import ray
      ray.init()

    To terminate the Ray runtime, run
      ray stop

    To view the status of the cluster, use
      ray status
   ```

2. Connect worker nodes to the head node: `ray start --address=<head_node_ip>:<port>`.

3. Check the cluster status using `ray status`. If the displayed total number of NPUs matches the sum across all nodes, the cluster is successful.

   When there are two nodes, each with 8 NPUs, the result is as follows:

   ```text
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

#### Setting Environment Variables

Configure the following environment variables on both the head and worker nodes:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export MS_ENABLE_LCCL=off
export HCCL_OP_EXPANSION_MODE=AIV
export MS_ALLOC_CONF=enable_vmm:true
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_MS_MODEL_BACKEND=MindFormers
```

Environment Variable Descriptions:

- `MS_ENABLE_LCCL`: Disables LCCL and enables HCCL communication.
- `HCCL_OP_EXPANSION_MODE`: Configures the scheduling and expansion location of the communication algorithm to be the AI Vector Core computing unit on the Device side.
- `MS_ALLOC_CONF`: Sets the memory policy. Refer to the [MindSpore Official Documentation](https://www.mindspore.cn/docs/en/master/api_python/env_var_list.html).
- `ASCEND_RT_VISIBLE_DEVICES`: Configures the available device IDs for each node. Users can query this using the `npu-smi info` command.
- `VLLM_MS_MODEL_BACKEND`: The backend of the model being run. The models and model backends currently supported by the vLLM-MindSpore plugin can be queried in the [Model Support List](../../../user_guide/supported_models/models_list/models_list.md).

**If users need to use the Ray deployment method, the following environment variables must be additionally set:**

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export GLOO_SOCKET_IFNAME=enp189s0f0
export HCCL_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
```

Environment Variable Descriptions:

- `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION`: Used when there are version compatibility issues.
- `GLOO_SOCKET_IFNAME`: The GLOO backend port name, used for the network interface name when using gloo communication between multiple machines. Find the network card name corresponding to the IP via `ifconfig`.
- `HCCL_SOCKET_IFNAME`: Configures the HCCL port name, used for the network interface name when using HCCL communication between multiple machines. Find the network card name corresponding to the IP via `ifconfig`.
- `TP_SOCKET_IFNAME`: Configures the TP port name, used for the network interface name when using TP communication between multiple machines. Find the network card name corresponding to the IP via `ifconfig`.

#### Starting the Service

The vLLM-MindSpore plugin can deploy online inference using the OpenAI API protocol. The following is the startup process for online inference:

```bash
# Launch configuration parameter explanation
vllm-mindspore serve
 [Model Tag: Path to model Config and weight files]
 --trust-remote-code # Use the locally downloaded model file
 --max-num-seqs [Maximum Batch Size]
 --max-model-len [Model Context Length]
 --max-num-batched-tokens [Maximum number of tokens supported per iteration, recommended 4096]
 --block-size [Block Size, recommended 128]
 --gpu-memory-utilization [Memory utilization rate, recommended 0.9]
 --tensor-parallel-size [TP parallelism degree]
 --headless # Only needed for worker nodes, indicates no server-side related content is needed
 --data-parallel-size [DP parallelism degree]
 --data-parallel-size-local [Number of DP workers on the current service node. The sum across all nodes equals data-parallel-size]
 --data-parallel-start-rank [The offset of the first DP worker responsible for the current service node, used when using the multiprocess startup method]
 --data-parallel-address [The communication IP address of the master node, used when using the multiprocess startup method]
 --data-parallel-rpc-port [The communication port of the master node, used when using the multiprocess startup method]
 --enable-expert-parallel # Enable Expert Parallelism
 --data-parallel-backend [ray, mp] # Specify the dp deployment method as Ray or mp (i.e., multiprocess)
 --additional-config # Parallel features and additional configurations
```

- Users can specify the local path where the model is saved as the model tag.

The following are execution examples for the multiprocess and Ray startup methods respectively, taking DP4-EP4-TP4 as example:

**Multiprocess Startup Method**

```bash
# Master Node:
vllm-mindspore serve MindSpore-Lab/DeepSeek-R1-0528-A8W8 --trust-remote-code --max-num-seqs=256 --max-model-len=32768 --max-num-batched-tokens=4096 --block-size=128 --gpu-memory-utilization=0.9 --tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-start-rank 0 --data-parallel-address 127.0.0.1 --data-parallel-rpc-port 29550 --enable-expert-parallel --additional-config '{"expert_parallel": 4}' --quantization ascend

# Worker Node:
vllm-mindspore serve MindSpore-Lab/DeepSeek-R1-0528-A8W8 --trust-remote-code --max-num-seqs=256 --max-model-len=32768 --max-num-batched-tokens=4096 --block-size=128 --gpu-memory-utilization=0.9 --headless --tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-start-rank 2 --data-parallel-address 127.0.0.1 --data-parallel-rpc-port 29550 --enable-expert-parallel --additional-config '{"expert_parallel": 4}' --quantization ascend
```

Specifically, `data-parallel-address` and `--data-parallel-rpc-port` must be configured with the actual environment information for the running instance.

**Ray Startup Method**

```bash
# Master Node:
vllm-mindspore serve MindSpore-Lab/DeepSeek-R1-0528-A8W8 --trust-remote-code --max-num-seqs=256 --max-model-len=32768 --max-num-batched-tokens=4096 --block-size=128 --gpu-memory-utilization=0.9 --tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-size-local 2 --enable-expert-parallel --additional-config '{"expert_parallel": 4}' --data-parallel-backend ray --quantization ascend
```

#### Sending Requests

Use the following command to send a request. The `prompt` field is the model input:

```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "MindSpore-Lab/DeepSeek-R1-0528-A8W8", "prompt": "I am, "max_tokens": 120, "temperature": 0}'
```

Users must ensure that the `"model"` field matches the model tag used when starting the service for the request to successfully match the model.
