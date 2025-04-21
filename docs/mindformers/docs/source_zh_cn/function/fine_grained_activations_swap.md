# 细粒度激活值SWAP

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindformers/docs/source_zh_cn/function/fine_grained_activations_swap.md)

## 概述

在传统大模型训练任务中，计算卡的显存资源常常成为训练瓶颈，采用更大规模的模型并行（model parallel, mp）和流水线并行（pipeline parallel, pp）切分策略，虽然能一定程度上缓解单张计算卡的显存压力，但需要更大规模的集群资源，且引入过多的通信会极大地降低模型的MFU。在集群资源有限的情况下，重计算是另一个缓解内存压力的有效手段，其通过放弃存储正向传播阶段的激活值，并在梯度反向回传时重新计算所需激活值，来降低激活值的显存占用，由于重计算需引入额外的计算开销，因此该方法同样会显著降低模型训练的MFU（Model FLOPs Utilization）。

在此背景下，细粒度激活值SWAP技术可以提供第三种降低内存占用的有效手段，且拥有更大的性能优势。具体地，激活值SWAP技术在模型正向传播阶段，将需要长期存储的激活值卸载至host侧，并在反向传播阶段，使用该激活值时，提前将其预取回device侧。资源使用方面，激活值SWAP技术使用D2H/H2D带宽，可以在训练阶段与计算任务、D2D通信任务并发，实现对内存搬运开销的掩盖。

细粒度激活值SWAP技术具备较高的使用灵活度。大模型训练的正向传播阶段，将产生数据量大小不同的若干激活值，用户可按需选择特定的激活值进行SWAP，且选择激活值的粒度为算子级。当模型类型或规格改变时，用户可灵活调整对应的SWAP策略，以追求最低的内存开销和最优的性能。

## 使用说明

### 约束场景

- 仅支持静态图O0/O1模式
- 支持Llama系稠密模型，后续演进支持MoE稀疏模型
- Somas不支持异构，需在配置文件中设置

  ```yaml
  context:
    memory_optimize_level=O0
  ```

- 未开启流水线并行时，需使能lazy_inline场景，设置环境变量

  ```bash
  ENABLE_LAZY_INLINE_NO_PIPELINE=1
  ```

- 仅支持Ascend后端

### 接口说明

细粒度激活值SWAP特性通过YAML配置`swap_config`字段使能，包括`swap`、`default_prefetch`、`layer_swap`、`op_swap`四个功能接口，用户可通过此接口灵活选择特定层或特定层的特定算子使能激活值SWAP功能。

> 当前MindSpore框架将内存搬运与内存释放解耦。将激活值从device侧卸载至host侧时，即便数据已全部卸载，其在device侧占用的内存空间并未被立刻释放，而是需要再触发释放操作。内存释放操作触发前，会检测激活值卸载是否完成，若未完成，则进程会原地等待，直至激活值卸载完成。

| 配置项 | 类型 | 说明 |
|:--:|:--:|:---|
| swap | Bool | 默认值False。当为False时，本特性的四个功能接口全部不生效；当为True时，激活值SWAP功能开启，并检查`layer_swap`与`op_swap`是否为None，若均为None，则启用默认的SWAP策略，该策略将对所有层中的`flash_attention`算子使能SWAP。若`layer_swap`与`op_swap`存在非None值，则屏蔽默认策略并按照`layer_swap`与`op_swap`的配置使能SWAP功能。 |
| default_prefetch | Int | 默认值1。当swap=True、layer_None、op_swap=None时生效。`default_prefetch`用于调控默认SWAP策略的激活值内存释放时机和预取开始时机。当`default_prefetch`较大时，正向阶段释放内存时机较晚，激活值占用的device内存会在激活值卸载完成后被长期锁住，不被其他数据块复用，同时反向阶段开始将激活值从host侧拷贝至device侧的时机较早，申请相应内存空间的时间较早，内存压力未得到真正缓解；当`default_prefetch`较小时，正向阶段内存释放时机较早，存在等待激活值拷贝任务完成的空等时间，且反向阶段预取的开始时机较晚，若在使用激活值计算时仍未完成激活值预取，则也会引入等待时间，影响端到端性能。因此开放本接口，供用户调试内存释放时机与激活值预期时机，以达到最少的内存占用和最优的端到端性能。|
| layer_swap | List | 默认值None。当为None时，本接口不生效；当为List类型时，本接口包含若干Dict类型的列表元素，每个Dict类型元素包含`backward_prefetch`与`layers`两个键，提供使能SWAP的预取时机（即开始搬回操作的时机）和对应的层索引。 |
| op_swap | List | 默认值None。当为None时，本接口不生效；当为List类型时，本接口包含若干Dict类型的列表元素，每个Dict类型元素包含`op_name`、`backward_prefetch`与`layers`三个键，提供使能SWAP的预取时机和对应的算子名、层索引。 |

### 混合重计算

细粒度激活值SWAP与重计算存在耦合：

1. 任意算子在同时使能重计算与SWAP时，重计算将生效，SWAP不生效。
2. 对于任意使能了SWAP的算子，若使用其输出的算子使能了重计算，则该算子的SWAP不生效。
3. 重计算的YAML配置接口只支持从前至后选择特定数量的层使能重计算，而不支持选择特定层或特定层的特定算子使能重计算，这意味着同时使用SWAP与重计算时，SWAP只能使能靠后的层或靠后层中的算子，无法获取SWAP特性的最大收益。因此当且仅当`swap=True`时，重计算接口功能将按下表调整。

| 接口名称 | 原功能 | 开启SWAP后功能 |
|:--:|:---|:---|
| recompute | 确定各pipeline stage中使能重计算的层数 | 不感知pipeline stage，仅接受bool/list类型入参。当为bool类型时，所有层使能重计算；当为list类型时，列表元素为层索引，按索引选择特定层使能重计算 |
| select_recompute | 确定各pipeline stage中特定算子使能重计算的层数 | 不感知pipeline stage，对于每个算子的键值对，仅接受bool/list类型入参。当为bool类型时，所有层使能重计算；当为list类型时，列表元素为层索引，按索引选择特定层使能重计算 |
| select_comm_recompute | 确定各pipeline stage中通信算子使能重计算的层数 | 不感知pipeline stage，仅接受bool/list类型入参。当为bool类型时，所有层使能重计算；当为list类型时，列表元素为层索引，按索引选择特定层使能重计算 |

## 使用示例

本章节以 Llama2-7B 训练为例，演示细粒度激活值SWAP特性的使用。

### 环境准备

下载 MindSpore Transformers，并准备预训练数据集，如wikitext等。

### 示例一：默认SWAP策略

在YAML中修改补充重计算与SWAP配置，主要配置参数如下：

```yaml
context:
  memory_optimize_level: "O0"
model:
  model_config:
    num_layers: 4
recompute_config:
  recompute: False
  select_recompute: False
  select_comm_recompute: False
swap_config:
  swap: True
  default_prefetch: 10
```

执行以下脚本启动单机八卡训练，启动脚本所在路径为MindSpore Transformers代码根目录，执行脚本需用户指定YAML文件路径（其中，machine_ip需要填写本地环境IP）：

```bash
export GLOG_v=1
export MS_MEMORY_STATISTIC=1
export ENABLE_LAZY_INLINE_NO_PIPELINE=1
YAML_FILE=$1 # 用户指定YAML文件路径
ROOT_PATH=`pwd`

bash ./scripts/msrun_launcher.sh "run_mindformer.py \
    --config ${ROOT_PATH}/${YAML_FILE} \
    --run_mode train \
    --use_parallel True" \
    8 8 <machine_ip> 8118 0 output/msrun False 300
```

训练完毕后执行命令`cat output/msrun/worker_0.log | grep 'attention.flash_attention'`查看默认SWAP策略的执行情况：

```text
-INFO - Set op_swap at layer 0: attention.flash_attention, value=10
-INFO - Set op_swap at layer 1: attention.flash_attention, value=10
-INFO - Set op_swap at layer 2: attention.flash_attention, value=10
-INFO - Set op_swap at layer 3: attention.flash_attention, value=10
```

默认SWAP策略执行成功。

### 示例二：选择特定层使能SWAP

在YAML中修改补充重计算与SWAP配置，主要配置参数如下：

```yaml
context:
  memory_optimize_level: "O0"
model:
  model_config:
    num_layers: 4
recompute_config:
  recompute: False
  select_recompute: False
  select_comm_recompute: False
swap_config:
  swap: True
  layer_swap:
    - backward_prefetch: 20
      layers: [0,3]
```

执行以下脚本启动单机八卡训练，启动脚本所在路径为MindSpore Transformers代码根目录，执行脚本需用户指定YAML文件路径（其中，machine_ip需要填写本地环境IP）：

```bash
export GLOG_v=1
export MS_MEMORY_STATISTIC=1
export ENABLE_LAZY_INLINE_NO_PIPELINE=1
YAML_FILE=$1 # 用户指定YAML文件路径
ROOT_PATH=`pwd`

bash ./scripts/msrun_launcher.sh "run_mindformer.py \
    --config ${ROOT_PATH}/${YAML_FILE} \
    --run_mode train \
    --use_parallel True" \
    8 8 <machine_ip> 8118 0 output/msrun False 300
```

训练完毕后执行命令`cat output/msrun/worker_0.log | grep 'Set layer swap at'`查看默认SWAP策略的执行情况：

```text
-INFO - Set layer swap at layer 0 and value is: 20
-INFO - Set layer swap at layer 3 and value is: 20
```

选择特定层使能SWAP的策略执行成功。

### 示例三：选择特定层的特定算子使能SWAP

在YAML中修改补充重计算与SWAP配置，主要配置参数如下：

```yaml
context:
  memory_optimize_level: "O0"
model:
  model_config:
    num_layers: 4
recompute_config:
  recompute: False
  select_recompute: False
  select_comm_recompute: False
swap_config:
  swap: True
  op_swap:
    - op_name: 'attention'
      backward_prefetch: 20
      layers: [0,1,2]
    - op_name: 'attention'
      backward_prefetch: 10
      layers: [3]
    - op_name: 'feed_forward'
      backward_prefetch: 15
      layers: [1,2]
```

执行以下脚本启动单机八卡训练，启动脚本所在路径为MindSpore Transformers代码根目录，执行脚本需用户指定YAML文件路径（其中，machine_ip需要填写本地环境IP）：

```bash
export GLOG_v=1
export MS_MEMORY_STATISTIC=1
export ENABLE_LAZY_INLINE_NO_PIPELINE=1
YAML_FILE=$1 # 用户指定YAML文件路径
ROOT_PATH=`pwd`

bash ./scripts/msrun_launcher.sh "run_mindformer.py \
    --config ${ROOT_PATH}/${YAML_FILE} \
    --run_mode train \
    --use_parallel True" \
    8 8 <machine_ip> 8118 0 output/msrun False 300
```

训练完毕后执行命令`cat output/msrun/worker_0.log | grep 'Set op_swap at layer'`查看默认SWAP策略的执行情况：

```text
-INFO - Set op_swap at layer 0: .attention, value=20
-INFO - Set op_swap at layer 1: .attention, value=20, .feed_forward, value=15
-INFO - Set op_swap at layer 2: .attention, value=20, .feed_forward, value=15
-INFO - Set op_swap at layer 3: .attention, value=10
```

选择特定层的特定算子使能SWAP成功。

### 示例四：细粒度激活值SWAP与重计算混用

在YAML中修改补充重计算与SWAP配置，主要配置参数如下：

```yaml
context:
  memory_optimize_level: "O0"
model:
  model_config:
    num_layers: 4
recompute_config:
  recompute: False
  select_recompute:
    'feed_forward': [0,3]
  select_comm_recompute: False
swap_config:
  swap: True
  op_swap:
    - op_name: 'attention'
      backward_prefetch: 20
      layers: [0,1,2]
    - op_name: 'attention'
      backward_prefetch: 10
      layers: [3]
    - op_name: 'feed_forward'
      backward_prefetch: 15
      layers: [1,2]
```

执行以下脚本启动单机八卡训练，启动脚本所在路径为MindSpore Transformers代码根目录，执行脚本需用户指定YAML文件路径（其中，machine_ip需要填写本地环境IP）：

```bash
export GLOG_v=1
export MS_MEMORY_STATISTIC=1
export ENABLE_LAZY_INLINE_NO_PIPELINE=1
YAML_FILE=$1 # 用户指定YAML文件路径
ROOT_PATH=`pwd`

bash ./scripts/msrun_launcher.sh "run_mindformer.py \
    --config ${ROOT_PATH}/${YAML_FILE} \
    --run_mode train \
    --use_parallel True" \
    8 8 <machine_ip> 8118 0 output/msrun False 300
```

训练完毕后执行命令`cat output/msrun/worker_0.log | grep 'Set op_swap at layer' -C 1`查看默认SWAP策略的执行情况：

```text
-INFO - Set select recompute at layer 0: feed_forward
-INFO - Set op_swap at layer 0: .attention, value=20
-INFO - Set op_swap at layer 1: .attention, value=20, .feed_forward, value=15
-INFO - Set op_swap at layer 2: .attention, value=20, .feed_forward, value=15
-INFO - Set select recompute at layer 3: feed_forward
-INFO - Set op_swap at layer 3: .attention, value=10
```

细粒度激活值SWAP与重计算混用成功。