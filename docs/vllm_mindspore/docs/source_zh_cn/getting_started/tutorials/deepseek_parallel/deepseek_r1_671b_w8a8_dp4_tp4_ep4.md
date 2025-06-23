# 并行推理（DeepSeek R1）

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_zh_cn/getting_started/tutorials/deepseek_parallel/deepseek_r1_671b_w8a8_dp4_tp4_ep4.md)

vLLM MindSpore支持张量并行（TP）、数据并行（DP）、专家并行（EP）及其组合配置的混合并行推理，不同并行策略的适用场景可参考[vLLM官方文档](https://docs.vllm.ai/en/latest/configuration/optimization.html#parallelism-strategies)。

本文档将以DeepSeek R1 671B W8A8为例介绍[张量并行](#tp16-张量并行推理)及[混合并行](#dp4tp4ep4-混合并行推理)推理流程。DeepSeek R1 671B W8A8模型需使用多个节点资源运行推理模型。为确保各个节点的执行配置（包括模型配置文件路径、Python环境等）一致，推荐通过 docker 镜像创建容器的方式避免执行差异。

用户可通过以下[新建容器](#新建容器)章节或参考[安装指南](../../installation/installation.md#安装指南)进行环境配置。

## 新建容器

```bash
docker pull hub.oepkgs.net/oedeploy/openeuler/aarch64/mindspore:latest

# 分别在主从节点新建docker容器
docker run -itd --name=mindspore_vllm --ipc=host --network=host --privileged=true \
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
        hub.oepkgs.net/oedeploy/openeuler/aarch64/mindspore:latest \
        bash
```

新建容器后成功后，将返回容器ID。用户可执行以下命令，确认容器是否创建成功：

```bash
docker ps
```

### 进入容器

用户在完成[新建容器](#新建容器)后，使用已定义的环境变量`DOCKER_NAME`，启动并进入容器：

```bash
docker exec -it $DOCKER_NAME bash
```

## 下载模型权重

用户可采用[Python工具下载](#python工具下载)或[git-lfs工具下载](#git-lfs工具下载)两种方式，进行模型下载。

### Python工具下载

执行以下 Python 脚本，从[魔乐社区](https://modelers.cn)下载 MindSpore版本的DeepSeek-R1 W8A8权重及文件：

```python
from openmind_hub import snapshot_download
snapshot_download(repo_id="MindSpore-Lab/DeepSeek-R1-W8A8",
                  local_dir="/path/to/save/deepseek_r1_w8a8",
                  local_dir_use_symlinks=False)
```

其中`local_dir`为模型保存路径，由用户指定，请确保该路径下有足够的硬盘空间。

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

工具确认可用后，执行以下命令，下载权重：

```shell
git clone https://modelers.cn/MindSpore-Lab/DeepSeek-R1-W8A8.git
```

## TP16 张量并行推理

vLLM 通过 Ray 对多个节点资源进行管理和运行。该样例对应张量并行（TP）为16的场景。

### 设置环境变量

环境变量必须设置在 Ray 创建集群前，且当环境有变更时，需要通过 `ray stop` 将主从节点集群停止，并重新创建集群，否则环境变量将不生效。

分别在主从节点配置如下环境变量：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export GLOO_SOCKET_IFNAME=enp189s0f0
export HCCL_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
export MS_ENABLE_LCCL=off
export HCCL_OP_EXPANSION_MODE=AIV
export MS_ALLOC_CONF=enable_vmm:true
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export vLLM_MODEL_BACKEND=MindFormers
export MINDFORMERS_MODEL_CONFIG=/path/to/research/deepseek3/deepseek_r1_671b/predict_deepseek_r1_671b_w8a8.yaml
```

环境变量说明：

- `GLOO_SOCKET_IFNAME`: GLOO后端端口。可通过`ifconfig`查找ip对应网卡的网卡名。
- `HCCL_SOCKET_IFNAME`: 配置HCCL端口。可通过`ifconfig`查找ip对应网卡的网卡名。
- `TP_SOCKET_IFNAME`: 配置TP端口。可通过`ifconfig`查找ip对应网卡的网卡名。
- `MS_ENABLE_LCCL`: 关闭LCCL，使能HCCL通信。
- `HCCL_OP_EXPANSION_MODE`: 配置通信算法的编排展开位置为Device侧的AI Vector Core计算单元。
- `MS_ALLOC_CONF`: 设置内存策略。可参考[MindSpore官网文档](https://www.mindspore.cn/docs/zh-CN/master/api_python/env_var_list.html)。
- `ASCEND_RT_VISIBLE_DEVICES`: 配置每个节点可用device id。用户可使用`npu-smi info`命令进行查询。
- `vLLM_MODEL_BACKEND`：所运行的模型后端。目前vLLM MindSpore所支持的模型与模型后端，可在[模型支持列表](../../../user_guide/supported_models/models_list/models_list.md)中进行查询。
- `MINDFORMERS_MODEL_CONFIG`：模型配置文件。用户可以在[MindSpore Transformers工程](https://gitee.com/mindspore/mindformers/tree/dev/research/deepseek3/deepseek_r1_671b)中，找到对应模型的yaml文件[predict_deepseek_r1_671b_w8a8.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/deepseek_r1_671b/predict_deepseek_r1_671b_w8a8.yaml) 。

模型并行策略通过配置文件中的`parallel_config`指定，例如TP16 张量并行配置如下所示：

```text
# default parallel of device num = 16 for Atlas 800T A2
parallel_config:
  data_parallel: 1
  model_parallel: 16
  pipeline_stage: 1
  expert_parallel: 1
```

### 启动 Ray 进行多节点集群管理

在 Ascend 上，需要额外安装 pyACL 包来适配 Ray。且所有节点的 CANN 依赖版本需要保持一致。

#### 安装 pyACL

pyACL (Python Ascend Computing Language) 通过 CPython 封装了 AscendCL 对应的 API 接口，使用接口可以管理 Ascend AI 处理器和对应的计算资源。

在对应环境中，获取相应版本的 Ascend-cann-nnrt 安装包后，解压出 pyACL 依赖包并单独安装，并将安装路径添加到环境变量中：

```shell
./Ascend-cann-nnrt_8.0.RC1_linux-aarch64.run --noexec --extract=./
cd ./run_package
./Ascend-pyACL_8.0.RC1_linux-aarch64.run --full --install-path=<install_path>
export PYTHONPATH=<install_path>/CANN-<VERSION>/python/site-packages/:$PYTHONPATH
```

在 Ascend 的首页中可以下载 Ascend 运行包。如, 可以下载 [8.0.RC1.beta1](https://www.hiascend.cn/developer/download/community/result?module=cann&version=8.0.RC1.beta1) 对应版本的运行包。

#### 多节点间集群

多节点集群管理前，需要检查各节点的 hostname 是否各异，如果存在相同的，需要通过 `hostname <new-host-name>` 设置不同的 hostname。

1. 启动主节点 `ray start --head --port=<port-to-ray>`，启动成功后，会提示从节点的连接方式。如在 ip 为 `192.5.5.5` 的环境中，通过 `ray start --head --port=6379`，提示如下：

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

2. 从节点连接主节点 `ray start --address=<head_node_ip>:<port>`。
3. 通过 `ray status` 查询集群状态，显示的NPU总数为节点总合，则表示集群成功。

   当有两个节点，每个节点有8个NPU时，其结果如下：

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

### 启动在线服务

#### 启动服务

vLLM MindSpore可使用OpenAI的API协议，部署为在线服务。以下是在线服务的拉起流程。

```bash
# 启动配置参数说明

vllm-mindspore serve
 --model=[模型Config/权重路径]
 --trust-remote-code # 使用本地下载的model文件
 --max-num-seqs [最大Batch数]
 --max-model-len [输出输出最大长度]
 --max-num-batched-tokens [单次迭代最大支持token数, 推荐4096]
 --block-size [Block Size 大小, 推荐128]
 --gpu-memory-utilization [显存利用率, 推荐0.9]
 --tensor-parallel-size [TP 并行数]
```

执行示例

```bash
# 主节点：
vllm-mindspore serve --model="/path/to/save/deepseek_r1_w8a8" --trust-remote-code --max-num-seqs=256 --max_model_len=32768 --max-num-batched-tokens=4096 --block-size=128 --gpu-memory-utilization=0.9 --tensor-parallel-size 16 --distributed-executor-backend=ray
```

张量并行场景下，`--tensor-parallel-size`参数会覆盖模型yaml文件中`parallel_config`的`model_parallel`配置。

#### 发起请求

使用如下命令发送请求。其中`prompt`字段为模型输入：

```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "/path/to/save/deepseek_r1_w8a8", "prompt": "I am", "max_tokens": 20, "temperature": 0, "top_p": 1.0, "top_k": 1, "repetition_penalty": 1.0}'
```

## DP4TP4EP4 混合并行推理

vLLM 通过 Ray 对多个节点资源进行管理和运行。该样例对应以下并行策略场景：

- 数据并行（DP）为4；
- 张量并行（TP）为4；
- 专家并行（EP）为4。

### DP4TP4EP4 设置环境变量

分别在主从节点配置如下环境变量：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export MS_ENABLE_LCCL=off
export HCCL_OP_EXPANSION_MODE=AIV
export MS_ALLOC_CONF=enable_vmm:true
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export vLLM_MODEL_BACKEND=MindFormers
export MINDFORMERS_MODEL_CONFIG=/path/to/research/deepseek3/deepseek_r1_671b/predict_deepseek_r1_671b_w8a8_ep4tp4.yaml
```

环境变量说明：

- `MS_ENABLE_LCCL`: 关闭LCCL，使能HCCL通信。
- `HCCL_OP_EXPANSION_MODE`: 配置通信算法的编排展开位置为Device侧的AI Vector Core计算单元。
- `MS_ALLOC_CONF`: 设置内存策略。可参考[MindSpore官网文档](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/env_var_list.html)。
- `ASCEND_RT_VISIBLE_DEVICES`: 配置每个节点可用device id。用户可使用`npu-smi info`命令进行查询。
- `vLLM_MODEL_BACKEND`：所运行的模型后端。目前vLLM MindSpore所支持的模型与模型后端，可在[模型支持列表](../../../user_guide/supported_models/models_list/models_list.md)中进行查询。
- `MINDFORMERS_MODEL_CONFIG`：模型配置文件。用户可以在[MindSpore Transformers工程](https://gitee.com/mindspore/mindformers/tree/dev/research/deepseek3/deepseek_r1_671b)中，找到对应模型的yaml文件[predict_deepseek_r1_671b_w8a8.yaml](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/deepseek_r1_671b/predict_deepseek_r1_671b_w8a8_ep4tp4.yaml)。

模型并行策略通过配置文件中的`parallel_config`指定，例如DP4TP4EP4 混合并行配置如下所示：

```text
# default parallel of device num = 16 for Atlas 800T A2
parallel_config:
  data_parallel: 4
  model_parallel: 4
  pipeline_stage: 1
  expert_parallel: 4
```

`data_parallel`及`model_parallel`指定attn及ffn-dense部分的并行策略，`expert_parallel`指定moe部分路由专家并行策略，且需满足`data_parallel` * `model_parallel`可被`expert_parallel`整除。

### DP4TP4EP4 启动在线服务

`vllm-mindspore`可使用OpenAI的API协议部署在线服务。以下是在线服务的拉起流程：

```bash
# 启动配置参数说明
vllm-mindspore serve
 --model=[模型Config/权重路径]
 --trust-remote-code # 使用本地下载的model文件
 --max-num-seqs [最大Batch数]
 --max-model-len [输出输出最大长度]
 --max-num-batched-tokens [单次迭代最大支持token数, 推荐4096]
 --block-size [Block Size 大小, 推荐128]
 --gpu-memory-utilization [显存利用率, 推荐0.9]
 --tensor-parallel-size [TP 并行数]
 --headless # 仅从节点需要配置，表示不需要服务侧相关内容
 --data-parallel-size [DP 并行数]
 --data-parallel-size-local [当前服务节点中的DP数，所有节点求和等于data-parallel-size]
 --data-parallel-start-rank [当前服务节点中负责的首个DP的偏移量]
 --data-parallel-address [主节点的通讯IP]
 --data-parallel-rpc-port [主节点的通讯端口]
 --enable-expert-parallel # 使能专家并行
```

执行示例：

```bash
# 主节点：
vllm-mindspore serve --model="/path/to/save/deepseek_r1_w8a8" --trust-remote-code --max-num-seqs=256 --max-model-len=32768 --max-num-batched-tokens=4096 --block-size=128 --gpu-memory-utilization=0.9 --tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-start-rank 0 --data-parallel-address 192.10.10.10 --data-parallel-rpc-port 12370 --enable-expert-parallel

# 从节点：
vllm-mindspore serve --headless --model="/path/to/save/deepseek_r1_w8a8" --trust-remote-code --max-num-seqs=256 --max-model-len=32768 --max-num-batched-tokens=4096 --block-size=128 --gpu-memory-utilization=0.9 --tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-start-rank 2 --data-parallel-address 192.10.10.10 --data-parallel-rpc-port 12370 --enable-expert-parallel
```

## 发送请求

使用如下命令发送请求。其中`$PROMPT`为模型输入：

```bash
PROMPT="I am"
MAX_TOKEN=120
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "/path/to/save/deepseek_r1_w8a8", "prompt": "$PROMPT", "max_tokens": $MAX_TOKEN, "temperature": 0}'
```
