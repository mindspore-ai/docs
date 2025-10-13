# 并行推理方法

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_zh_cn/user_guide/supported_features/parallel/parallel.md)

vLLM-MindSpore插件支持张量并行（TP）、数据并行（DP）、专家并行（EP）及其组合配置的混合并行推理，并可以使用`Ray`或者`multiprocess`进行多机多卡启动。不同并行策略的适用场景可参考[vLLM官方文档](https://docs.vllm.ai/en/latest/configuration/optimization.html#parallelism-strategies)。下面将展开介绍[张量并行](#张量并行)、[数据并行](#数据并行)、[专家并行](#专家并行)、[混合并行](#混合并行)的使用场景、参数配置与[在线推理](#在线推理)。

## 张量并行

张量并行将模型权重参数，在每个模型层内跨多个NPU进行分片。当模型过大超过单个NPU容量，或需要降低单个NPU压力、为KV缓存腾出更多空间以实现更高吞吐量时，张量并行是进行大模型推理时的推荐策略。更多信息可查看[vLLM中关于张量并行的介绍](https://docs.vllm.ai/en/v0.9.1/configuration/optimization.html?h=expert#tensor-parallelism-tp)。

### 参数配置

使用张量并行（TP），需要在启动命令`vllm-mindspore serve`中配置以下选项：

- `--tensor-parallel-size`：TP 并行数

### 单机示例

以下命令为单机四卡，启动Qwen-2.5的张量并行示例：

```bash
TENSOR_PARALLEL_SIZE=4       # TP 并行数

vllm-mindspore serve /path/to/Qwen2.5/model --trust-remote-code --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
```

### 多机示例

多机张量并行依赖Ray进行启动。请参考[Ray多节点集群管理](#ray多节点集群管理)进行Ray环境配置。

以下命令为双机四卡，启动Qwen-2.5的张量并行示例：

```bash
# 主节点：

TENSOR_PARALLEL_SIZE=4       # TP 并行数
vllm-mindspore serve /path/to/Qwen2.5/model --trust-remote-code --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
```

## 数据并行

数据并行通过在多组NPU间完整复制模型副本，推理时并行处理不同batch的请求。当具备充足NPU可完整复制模型时，需要提升吞吐量而非扩大模型规模，在多用户环境中需要保持请求批次间隔离性时，数据并行是进行大模型推理时的推荐策略。数据并行可与其他并行策略组合使用。请注意：MoE（混合专家）层将根据张量并行规模与数据并行规模的乘积进行分片。更多信息可查看[vLLM中关于数据并行的介绍](https://docs.vllm.ai/en/v0.9.1/configuration/optimization.html?h=expert#data-parallelism-dp)。

### 参数配置

使用数据并行（DP），需要在启动命令`vllm-mindspore serve`中配置以下选项：

- `--data-parallel-size`：DP 并行数；
- `--data-parallel-backend`：设置DP部署方式，可选项为 `mp` 和 `ray`。默认行为下，DP 将以 multiprocess 方式部署：
    - `mp`：以multiprocess的方式部署；
    - `ray`：以 Ray 方式部署。
- 当`--data-parallel-backend`取值为`mp`时，还需要在启动命令中，配置以下选项：
    - `--data-parallel-size-local`：当前服务节点中的DP数，所有节点求和等于`--data-parallel-size`；
    - `--data-parallel-start-rank`：当前服务节点中负责的首个DP的偏移量；
    - `--data-parallel-address`：主节点的通讯IP；
    - `--data-parallel-rpc-port`：主节点的通讯端口。

### 服务启动示例

#### Ray启动示例（推荐）

Ray在多机场景中可以简化启动，是推荐的启动方式，请参考[Ray多节点集群管理](#ray多节点集群管理)进行Ray环境配置。以下命令为双机四卡，Ray启动Qwen-2.5的数据并行示例：

```bash
DATA_PARALLEL_SIZE=4       # DP 并行数
DATA_PARALLEL_SIZE_LOCAL=2 # 当前服务节点中的DP数，所有节点求和等于`--data-parallel-size`

vllm-mindspore serve /path/to/Qwen2.5/model --trust-remote-code --data-parallel-size ${DATA_PARALLEL_SIZE} --data-parallel-size-local ${DATA_PARALLEL_SIZE_LOCAL} --data-parallel-backend=ray
```

#### multiprocess启动

当用户有简化依赖的需要，可进行multiprocess启动。以下命令为双机四卡、multiprocess启动Qwen-2.5的数据并行示例：

```bash
# 主节点：
vllm-mindspore serve /path/to/Qwen2.5/model --trust-remote-code --data-parallel-size ${DATA_PARALLEL_SIZE} --data-parallel-size-local ${DATA_PARALLEL_SIZE_LOCAL}

# 从节点：
vllm-mindspore serve /path/to/Qwen2.5/model --headless --trust-remote-code --data-parallel-size ${DATA_PARALLEL_SIZE} --data-parallel-size-local ${DATA_PARALLEL_SIZE_LOCAL}
```

## 专家并行

专家并行是混合专家（MoE）模型特有的并行化形式，通过将不同专家网络分布到多个NPU上实现。该并行模式适用于MoE模型（如DeepSeekV3、Qwen3MoE、Llama-4等），且需要跨NPU平衡专家网络计算负载时，是进行大模型推理时的推荐策略。通过设置`enable_expert_parallel=True`和`--additional-config`启用专家并行，该设置将使MoE层采用专家并行而非张量并行策略。专家并行的并行度将保持与已设置的张量并行度一致。更多信息可查看[vLLM中关于专家并行的介绍](https://docs.vllm.ai/en/v0.9.1/configuration/optimization.html?h=expert#expert-parallelism-ep)。

### 参数配置

使用专家并行（EP）时，需要在启动命令`vllm-mindspore serve`中配置以下选项：

- `--enable-expert-parallel`：使能专家并行

- `--additional-config`：配置`expert_parallel`字段为EP并行数。例如配置EP为4，则

  ```bash
  --addition-config '{"expert_parallel": 4}
  ```

> - 如果不配置`--enable-expert-parallel`则不使能EP，配置 `--additional-config '{"expert_parallel": 4}'`不会生效；
> - 如果配置`--enable-expert-parallel`，但不配置`--additional-config '{"expert_parallel": 4'}`，则EP并行数等于TP并行数乘以DP并行数；
> - 如果配置`--enable-expert-parallel`，且配置`--additional-config '{"expert_parallel": 4'}`，则EP并行数等于`4`。

### 服务启动示例

#### 单机示例

以下命令为单机八卡，启动Qwen-3 MOE的专家并行示例：

```bash
vllm-mindspore serve /path/to/Qwen3-MOE --trust-remote-code --enable-expert-parallel --addition-config '{"expert_parallel": 8}
```

#### 多机示例

多机专家并行依赖Ray进行启动。请参考[Ray多节点集群管理](#ray多节点集群管理)进行Ray环境配置。以下命令为双机四卡，Ray启动Qwen-3 MOE的专家并行示例：

```bash
vllm-mindspore serve /path/to/Qwen3-MOE --trust-remote-code --enable-expert-parallel --addition-config '{"expert_parallel": 8} --data-parallel-backend=ray
```

## 混合并行

用户可根据所使用的模型与机器资源情况，灵活叠加调整并行策略。例如在DeepSeek-R1场景中，可以使用以下混合策略：

- 张量并行：4
- 数据并行：4
- 专家并行：4

可根据上述介绍，分别将三种并行策略的配置叠加，在启动命令`vllm-mindspore serve`中使能。多机混合并行依赖Ray进行启动。请参考[Ray多节点集群管理](#ray多节点集群管理)进行Ray环境配置。其叠加后混合并行的Ray启动命令如下：

```bash
vllm-mindspore serve /path/to/DeepSeek-R1 --trust-remote-code --max-num-seqs=256 --max-model-len=32768 --max-num-batched-tokens=4096 --block-size=128 --gpu-memory-utilization=0.9 --tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-size-local 2 --enable-expert-parallel --addition-config '{"expert_parallel": 4}' --data-parallel-backend=ray
```

## 附录

### Ray多节点集群管理

在 Ascend 上，有multiprocess和Ray两种启动方式。由于在多机场景中，如果使用Ray，则需要额外安装 pyACL 包来适配 Ray，所有节点的 CANN 依赖版本需要保持一致。

#### 安装 pyACL

pyACL (Python Ascend Computing Language) 通过 CPython 封装了 AscendCL 对应的 API 接口，使用该接口可以管理 Ascend AI 处理器和对应的计算资源。

在对应环境中，获取相应版本的 Ascend-cann-nnrt 安装包后，解压出 pyACL 依赖包并单独安装，并将安装路径添加到环境变量中：

```shell
./Ascend-cann-nnrt_8.0.RC1_linux-aarch64.run --noexec --extract=./
cd ./run_package
./Ascend-pyACL_8.0.RC1_linux-aarch64.run --full --install-path=<install_path>
export PYTHONPATH=<install_path>/CANN-<VERSION>/python/site-packages/:$PYTHONPATH
```

若安装过程有权限问题，可以使用以下命令加权限：

```bash
chmod -R 777 ./Ascend-pyACL_8.0.RC1_linux-aarch64.run
```

在 Ascend 的首页中可以下载 Ascend 运行包。例如，可以下载 [8.0.RC1.beta1](https://www.hiascend.cn/developer/download/community/result?module=cann&version=8.0.RC1.beta1) 对应版本的运行包。

#### 多节点间集群

多节点集群管理前，需要检查各节点的 hostname 是否各异。如果存在相同的，需要通过 `hostname <new-host-name>` 设置不同的 hostname。

1. 启动主节点 `ray start --head --port=<port-to-ray>`，启动成功后，会提示从节点的连接方式。如在 IP 为 `192.5.5.5` 的环境中，通过 `ray start --head --port=6379`，提示如下：

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

2. 从节点连接主节点 `ray start --address=<head_node_ip>:<port>`。

3. 通过 `ray status` 查询集群状态。显示的NPU总数为节点总和，则表示集群成功。

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

### 在线推理

#### 设置环境变量

分别在主从节点配置如下环境变量：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export MS_ENABLE_LCCL=off
export HCCL_OP_EXPANSION_MODE=AIV
export MS_ALLOC_CONF=enable_vmm:true
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_MS_MODEL_BACKEND=MindFormers
```

环境变量说明：

- `MS_ENABLE_LCCL`：关闭LCCL，使能HCCL通信。
- `HCCL_OP_EXPANSION_MODE`：配置通信算法的编排展开位置为Device侧的AI Vector Core计算单元。
- `MS_ALLOC_CONF`：设置内存策略。可参考[MindSpore官网文档](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/env_var_list.html)。
- `ASCEND_RT_VISIBLE_DEVICES`：配置每个节点可用device id。用户可使用`npu-smi info`命令进行查询。
- `VLLM_MS_MODEL_BACKEND`：所运行的模型后端。目前vLLM-MindSpore插件所支持的模型与模型后端，可在[模型支持列表](../../../user_guide/supported_models/models_list/models_list.md)中进行查询。

**如用户需要使用Ray方式部署，需要额外设置以下环境变量：**

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export GLOO_SOCKET_IFNAME=enp189s0f0
export HCCL_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
```

环境变量说明：

- `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION`：当版本不兼容时使用。
- `GLOO_SOCKET_IFNAME`：GLOO后端端口，用于多机之间使用gloo通信时的网口名称。可通过`ifconfig`查找IP对应网卡的网卡名。
- `HCCL_SOCKET_IFNAME`：配置HCCL端口，用于多机之间使用HCCL通信时的网口名称。可通过`ifconfig`查找IP对应网卡的网卡名。
- `TP_SOCKET_IFNAME`：配置TP端口，用于多机之间使用TP通信时的网口名称。可通过`ifconfig`查找IP对应网卡的网卡名。

#### 启动服务

vLLM-MindSpore插件可使用OpenAI的API协议部署在线推理。以下是在线推理的启动流程：

```bash
# 启动配置参数说明
vllm-mindspore serve
 [模型标签：模型Config/权重路径]
 --trust-remote-code # 使用本地下载的model文件
 --max-num-seqs [最大Batch数]
 --max-model-len [模型上下文长度]
 --max-num-batched-tokens [单次迭代最大支持token数，推荐4096]
 --block-size [Block Size 大小，推荐128]
 --gpu-memory-utilization [显存利用率，推荐0.9]
 --tensor-parallel-size [TP 并行数]
 --headless # 仅从节点需要配置，表示不需要服务侧相关内容
 --data-parallel-size [DP 并行数]
 --data-parallel-size-local [当前服务节点中的DP数，所有节点求和等于data-parallel-size]
 --data-parallel-start-rank [当前服务节点中负责的首个DP的偏移量，当使用multiprocess启动方式时使用]
 --data-parallel-address [主节点的通讯IP，当使用multiprocess启动方式时使用]
 --data-parallel-rpc-port [主节点的通讯端口，当使用multiprocess启动方式时使用]
 --enable-expert-parallel # 使能专家并行
 --data-parallel-backend [ray，mp] # 指定 dp 部署方式为 Ray 或是 mp(即multiprocess)
 --addition-config # 并行功能与额外配置
```

- 用户可以通过指定模型保存的本地路径为模型标签；

- 用户可以通过`--addition-config`参数，配置并行与其他功能。其中并行可进行如下配置，对应的是DP4-EP4-TP4场景：

  ```bash
  --addition-config '{"data_parallel": 4, "model_parallel": 4, "expert_parallel": 4}'
  ```

以下分别为multiprocess与Ray两种启动方式的执行示例：

**multiprocess启动方式**

```bash
# 主节点：
vllm-mindspore serve MindSpore-Lab/DeepSeek-R1-0528-A8W8 --trust-remote-code --max-num-seqs=256 --max-model-len=32768 --max-num-batched-tokens=4096 --block-size=128 --gpu-memory-utilization=0.9 --tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-start-rank 0 --data-parallel-address 192.10.10.10 --data-parallel-rpc-port 12370 --enable-expert-parallel --addition-config '{"data_parallel": 4, "model_parallel": 4, "expert_parallel": 4}'

# 从节点：
vllm-mindspore serve MindSpore-Lab/DeepSeek-R1-0528-A8W8 --headless --trust-remote-code --max-num-seqs=256 --max-model-len=32768 --max-num-batched-tokens=4096 --block-size=128 --gpu-memory-utilization=0.9 --tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-start-rank 2 --data-parallel-address 192.10.10.10 --data-parallel-rpc-port 12370 --enable-expert-parallel --addition-config '{"data_parallel": 4, "model_parallel": 4, "expert_parallel": 4}'
```

**Ray启动方式**

```bash
# 主节点：
vllm-mindspore serve MindSpore-Lab/DeepSeek-R1-0528-A8W8 --trust-remote-code --max-num-seqs=256 --max-model-len=32768 --max-num-batched-tokens=4096 --block-size=128 --gpu-memory-utilization=0.9 --tensor-parallel-size 4 --data-parallel-size 4 --data-parallel-size-local 2 --enable-expert-parallel --addition-config '{"data_parallel": 4, "model_parallel": 4, "expert_parallel": 4}' --data-parallel-backend=ray
```

#### 发送请求

使用如下命令发送请求。其中`prompt`字段为模型输入：

```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "MindSpore-Lab/DeepSeek-R1-0528-A8W8", "prompt": "I am, "max_tokens": 120, "temperature": 0}'
```

用户需确认`"model"`字段与启动服务中的模型标签一致，请求才能成功匹配到模型。
