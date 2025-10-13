# Multi-machine Parallel Inference (DeepSeek R1)

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.7.1/docs/vllm_mindspore/docs/source_en/getting_started/tutorials/deepseek_parallel/deepseek_r1_671b_w8a8_dp4_tp4_ep4.md)

This document describes the parallel inference startup process for the DeepSeek R1 671B W8A8 model. The DeepSeek R1 671B W8A8 model requires resources from multiple nodes to run the inference model. To ensure consistent execution configurations (including model configuration file paths, Python environment, etc.) across all nodes, it is recommended to use a Docker image to create containers and avoid execution discrepancies. Users can configure the environment by following the instructions in the [Docker Installation](#docker-installation) section below.

The vLLM-MindSpore plugin supports hybrid parallel inference configurations combining [Tensor Parallelism (TP)](https://docs.vllm.ai/en/v0.9.1/configuration/optimization.html?h=expert#tensor-parallelism-tp), [Data Parallelism (DP)](https://docs.vllm.ai/en/v0.9.1/configuration/optimization.html?h=expert#data-parallelism-dp), [Expert Parallelism (EP)](https://docs.vllm.ai/en/v0.9.1/configuration/optimization.html?h=expert#expert-parallelism-ep), and their combinations. For more information on multi-node parallel inference, refer to the [Parallel Inference Methods Introduction](../../../user_guide/supported_features/parallel/parallel.md).

This document's example requires two Atlas 800 A2 server nodes, providing a total of 16 available NPUs, each with 64GB specifications.

## Docker Installation

In this section, we recommend using Docker creation to quickly deploy the vLLM-MindSpore plugin environment. The following are the steps for deploying Docker:

### Building the Image

Users can execute the following commands to pull the vLLM-MindSpore plugin code repository and build the image:

```bash
git clone -b r0.4.0 https://gitee.com/mindspore/vllm-mindspore.git
bash build_image.sh
```

After a successful build, users will receive the following information:

```text
Successfully built e40bcbeae9fc
Successfully tagged vllm_ms_20250726:latest
```

Here, `e40bcbeae9fc` is the image ID, and `vllm_ms_20250726:latest` is the image name and tag. Users can execute the following command to confirm the Docker image was created successfully:

```bash
docker images
```

### Creating a New Container

After completing the [Building the Image](#building-the-image) step, set `DOCKER_NAME` and `IMAGE_NAME` as the container name and image name, respectively, and execute the following command to create a new container:

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

After successfully creating the container, the container ID will be returned. Users can execute the following command to confirm if the container was created successfully:

```bash
docker ps
```

### Entering the Container

After completing the [Creating a New Container](#creating-a-new-container) step, use the predefined environment variable `DOCKER_NAME` to start and enter the container:

```bash
docker exec -it $DOCKER_NAME bash
```

## Ray Multi-Node Cluster Management

On Ascend, if using Ray, an additional pyACL package needs to be installed to adapt Ray. The CANN dependency versions on all nodes must be consistent. This example relies on Ray for multi-node startup. For Ray installation instructions, please see the [Ray Installation Process Introduction](#ray-installation-process).

## Downloading Model Weights

Users can download the model using either the [Python Tool Download](#python-tool-download) method or the [git-lfs Tool Download](#git-lfs-tool-download) method.

### Python Tool Download

Execute the following Python script to download the MindSpore version of the DeepSeek-R1 W8A8 weights and files from [Modelers Community](https://modelers.cn):

```python
from openmind_hub import snapshot_download
snapshot_download(repo_id="MindSpore-Lab/DeepSeek-R1-0528-A8W8FA3",
                  local_dir="/path/to/save/deepseek_r1_0528_a8w8fa3",
                  local_dir_use_symlinks=False)
```

Where `local_dir` is the path to save the model, specified by the user. Please ensure there is sufficient hard disk space in this path.

### Git-lfs Tool Download

Execute the following code to confirm if the [git-lfs](https://git-lfs.com) tool is available:

```bash
git lfs install
```

If available, you will get a return result similar to the following:

```text
Git LFS initialized.
```

If the tool is not available, you need to install [git-lfs](https://git-lfs.com) first. Please refer to the instructions for [git-lfs installation](../../../faqs/faqs.md#git-lfs-installation) in the [FAQ](../../../faqs/faqs.md) section.

After confirming the tool is available, execute the following command to download the weights:

```bash
git clone https://modelers.cn/models/MindSpore-Lab/DeepSeek-R1-0528-A8W8.git
```

## Starting the Model Service

The following example uses the DeepSeek R1 671B W8A8 model to demonstrate starting the model service.

### Setting Environment Variables

Configure the following environment variables on both the head and worker nodes:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export MS_ENABLE_LCCL=off
export HCCL_OP_EXPANSION_MODE=AIV
export MS_ALLOC_CONF=enable_vmm:true
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_MS_MODEL_BACKEND=MindFormers
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export GLOO_SOCKET_IFNAME=enp189s0f0
export HCCL_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
```

Environment Variable Descriptions:

- `MS_ENABLE_LCCL`: Disables LCCL and enables HCCL communication.
- `HCCL_OP_EXPANSION_MODE`: Configures the scheduling and expansion location of the communication algorithm to be the AI Vector Core computing unit on the Device side.
- `MS_ALLOC_CONF`: Sets the memory policy. Refer to the [MindSpore Official Documentation](https://www.mindspore.cn/docs/en/master/api_python/env_var_list.html).
- `ASCEND_RT_VISIBLE_DEVICES`: Configures the available device IDs for each node. Users can query this using the `npu-smi info` command.
- `VLLM_MS_MODEL_BACKEND`: The backend of the model being run. The models and model backends currently supported by the vLLM-MindSpore plugin can be queried in the [Model Support List](../../../user_guide/supported_models/models_list/models_list.md).
- `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION`: Used when there are version compatibility issues.
- `GLOO_SOCKET_IFNAME`: The GLOO backend port name, used for the network interface name when using gloo communication between multiple machines. Find the network card name corresponding to the IP via `ifconfig`.
- `HCCL_SOCKET_IFNAME`: Configures the HCCL port name, used for the network interface name when using HCCL communication between multiple machines. Find the network card name corresponding to the IP via `ifconfig`.
- `TP_SOCKET_IFNAME`: Configures the TP port name, used for the network interface name when using TP communication between multiple machines. Find the network card name corresponding to the IP via `ifconfig`.

### Online Inference

#### Starting the Service

The vLLM-MindSpore plugin can deploy online inference using the OpenAI API protocol. The following is the startup process for online inference.

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
- Users can configure parallelism and other features using the `--additional-config` parameter.

The following is the Ray startup command:

```bash
# Master Node:
vllm-mindspore serve MindSpore-Lab/DeepSeek-R1-0528-A8W8 --trust-remote-code --max-num-seqs=256 --max-model-len=32768 --max-num-batched-tokens=4096 --block-size=128 --gpu-memory-utilization=0.9 --tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-size-local 2 --enable-expert-parallel --addition-config '{"expert_parallel": 4}' --data-parallel-backend=ray
```

For the multiprocess startup command, please refer to the [Multiprocess Startup Method](../../../user_guide/supported_features/parallel/parallel.md#starting-the-service).

#### Sending Requests

Use the following command to send a request. The `prompt` field is the model input:

```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "MindSpore-Lab/DeepSeek-R1-0528-A8W8", "prompt": "I am", "max_tokens": 120, "temperature": 0}'
```

Users must ensure that the `"model"` field matches the model tag used when starting the service for the request to successfully match the model.

## Appendix

### Ray Multi-Node Cluster Management

On Ascend, there are two startup methods: multiprocess and Ray. In multi-node scenarios, if using Ray, an additional pyACL package needs to be installed to adapt Ray, and the CANN dependency versions on all nodes must be consistent.

#### Installing pyACL

pyACL (Python Ascend Computing Language) wraps the corresponding API interfaces of AscendCL through CPython. Using these interfaces allows management of Ascend AI processors and their corresponding computing resources.

In the target environment, after obtaining the appropriate version of the Ascend-cann-nnrt installation package, extract the pyACL dependency package and install it separately. Then add the installation path to the environment variables:

```bash
./Ascend-cann-nnrt_8.0.RC1_linux-aarch64.run --noexec --extract=./
cd ./run_package
./Ascend-pyACL_8.0.RC1_linux-aarch64.run --full --install-path=<install_path>
export PYTHONPATH=<install_path>/CANN-<VERSION>/python/site-packages/:$PYTHONPATH
```

If there are permission issues during installation, use the following command to add permissions:

```bash
chmod -R 777 ./Ascend-pyACL_8.0.RC1_linux-aarch64.run
```

The Ascend runtime package can be downloaded from the Ascend homepage. For example, you can download the runtime package for version [8.0.RC1.beta1](https://www.hiascend.cn/developer/download/community/result?module=cann&version=8.0.RC1.beta1).

#### Multi-Node Cluster

Before managing a multi-node cluster, check that the hostnames of all nodes are different. If any are the same, set different hostnames using `hostname <new-host-name>`.

1. Start the head node: `ray start --head --port=<port-to-ray>`. Upon successful startup, the connection method for worker nodes will be displayed. For example, in an environment with IP `192.5.5.5`, running `ray start --head --port=6379` will prompt:

  ```text
  Local node IP: 192.5.5.5

  -------------------
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
