# Memory Optimization Features

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/memory_optimization.md)

## Recomputation

### Overview

Recomputation can significantly reduce activation memory usage during training but at the cost of additional computations. For more information about the principles of recalculation and framework measurement capabilities, please refer to [MindSpore Tutorial Document: Recompute](https://www.mindspore.cn/tutorials/en/master/parallel/recompute.html).

### Configuration and Usage

#### YAML Parameter Configuration

Users can enable recomputation by adding a `recompute_config` module to the YAML configuration file used for model training.

Taking the [DeepSeek-V3 pre-training's YAML file](https://gitee.com/mindspore/mindformers/blob/dev/research/deepseek3/deepseek3_671b/pretrain_deepseek3_671b.yaml#L113) as an example, it could be configured as follows:

```yaml
# recompute config
recompute_config:
  recompute: [3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 0]
  select_recompute: False
  parallel_optimizer_comm_recompute: True
  mp_comm_recompute: True
  recompute_slice_activation: True
```

For specific configurations targeting individual layers, a tuple-based approach can be used.

For instance, with a network having 48 layers, pp_interleave_num set to 2, pipeline_stage set to 5, and offset configured as [[0,1,1,1,1],[1,1,1,1,0]], the recomputation configuration would look like this:

```yaml
# recompute config
recompute_config:
  recompute: [[2,1,0,0,0],[1,0,0,0,0]]
  select_recompute:
    'feed_forward\.w1\.activation\.silu': True
    'feed_forward\.mul': True
    'feed_forward\.w1\.matmul': [[1,0,0,0,0],[2,1,0,0,0]]
    'feed_forward\.w3\.matmul': [2,1,0,0,0]
  select_comm_recompute: ['ffn_norm\.norm','attention_norm\.norm']
```

The log will print the recalculation strategy information after normalizing the input format:

```text
INFO - Formative layer_recompute: [[2, 1, 0, 0, 0], [1, 0, 0, 0, 0]]
INFO - Formative select_recompute: {'feed_forward\.w1\.activation\.silu': [[4, 5, 5, 5, 5], [5, 5, 5, 5, 4]], 'feed_forward\.mul': [[4, 5, 5, 5, 5], [5, 5, 5, 5, 4]], 'feed_forward\.w1\.matmul': [[1, 0, 0, 0, 0], [2, 1, 0, 0, 0]], 'feed_forward\.w3\.matmul': [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]]}
INFO - Formative select_comm_recompute: {'ffn_norm\.norm': [[4, 5, 5, 5, 5], [5, 5, 5, 5, 4]], 'attention_norm\.norm': [[4, 5, 5, 5, 5], [5, 5, 5, 5, 4]]}
```

Then the configuration of each layer recompute will be printed.

> 1. If both full recomputation and selective recomputation are configured for a layer, full recomputation takes effect.
> 2. Integers in a one-dimensional integer list or tuple can be replaced with True or False to enable or disable recomputation for all layers.

#### Key Parameters Introduction

The main parameters for recomputation configuration are listed in the following table:

| Parameter                         | Description                                                                                                            | ValueDescription                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|-----------------------------------|------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| recompute                         | (By layer) Full recompute.                                                                                             | Can be configured as bool, list or tuple of integers, or 2D list or tuple. <br>When configured as bool type, turn on or off full recompute for all layers; <br>When configured as list or tuple of integers, it indicates how many layers in each `pipline_stage` have full recompute enabled. When `pp_interleave_num > 1`, the number of recompute layers enabled will be evenly distributed to each interleave; <br>When configured as a 2D list or tuple of integers, it indicates how many layers in each mini stage have full recompute enabled.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| select_recompute                  | (By operator) Select recompute.                                                                                        | Can be configured as bool, list or tuple of integers, or two-dimensional list or tuple, list or tuple of strings, and dict. <br>The default selection recalculation operator is `['feed_forward\\.mul', 'feed_forward\\.w1\\.activation\\.silu']` . <br>When configured as bool type, it turns on or off the selection recalculation of the default operator for all layers; <br>When configured as an integer list or tuple, it represents how many layers in each `pipline_stage` turn on the selection recalculation of the default operator. When `pp_interleave_num > 1`, the number of selection recalculation layers turned on will be evenly distributed to each interleave; <br>When configured as an integer two-dimensional list or tuple, it represents how many layers in each mini stage turn on the selection recalculation of the default operator. <br>When configured as a string list or tuple, it indicates which operators are enabled for selective recomputation. The operator names are matched by regular expressions, and the hierarchical relationships are separated by `'\\.'`; <br>When configured as a dict, the key value corresponds to the operator name, and the value corresponds to the configuration method for selective recomputation. This method can fine-tune the recomputation strategy for each operator. |
| select_comm_recompute             | Select communication recomputation (by operator).                                                                      | The configuration method is the same as **select_recompute**. The default selection of communication recomputation operators is `['.*\\.norm']` . Generally, it is only configured for layer_norm or similar layers.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| parallel_optimizer_comm_recompute | Optimizer parallel communication recomputation. Whether to recompute AllGather communication in optimizer parallelism. | (bool, optional) - After enabling, in automatic parallelism or semi-automatic parallelism mode, specify whether AllGather communication introduced by optimizer parallelism in Cell is recomputed. Default value: `False`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| mp_comm_recompute                 | Model parallel communication recomputation, whether to recompute communication operators in model parallelism.         | (bool, optional) - After turning on, in automatic parallelism or semi-automatic parallelism mode, specify whether to recompute the communication operations introduced by model parallelism in the cell. Default value: `True`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| recompute_slice_activation        | Slice recomputation, whether to slice the cell output that will be kept in memory.                                     | (bool, optional) - Default value: `False`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

## Fine-Grained Activations SWAP

### Overview

In traditional large-scale model training tasks, the memory resources of computing cards often become a bottleneck. Although adopting larger-scale model parallel (mp) and pipeline parallel (pp) can alleviate the memory pressure on individual computing cards to some extent, it requires larger-scale cluster resources, and excessive communication can significantly reduce the model's Model FLOPs Utilization (MFU). Under limited cluster resources, recomputation is another effective method to mitigate memory pressure. It reduces the memory footprint of activations by discarding the storage of activation values during the forward propagation phase and recomputing the required activation values during gradient backpropagation. However, since recomputation introduces additional computational overhead, this method also significantly decreases the MFU of model training.

Against this backdrop, fine-grained activations SWAP can provide a third effective approach to reduce memory usage while offering greater end-to-end performance advantages. Specifically, SWAP offloads activations that need to be stored long-term to the host side during the forward propagation phase and prefetches them back to the device side in advance when they are needed during backpropagation. In terms of resource utilization, fine-grained activations SWAP leverages D2H/H2D bandwidth, which can overlap with computation tasks and D2D communication tasks during training, thereby masking the overhead of memory transfers.

The fine-grained activations SWAP technology offers high flexibility in usage. During the forward propagation phase of large model training, multiple activations of varying data sizes are generated, allowing users to swap specific activations at the granularity of the operator selectively. When the model type or configuration changes, users can flexibly adjust the corresponding SWAP strategy to minimize memory overhead and achieve optimal performance.

### Instrunction for Use

#### Constraint Scenarios

- Only support static graph O0/O1 mode
- Compatible with LLama-family dense models, MoE sparse models to be supported in future updates  
- Somas does not support heterogeneity and needs to be set in the configuration file:

  ```yaml
  context:
    memory_optimize_level=O0
  ```

- When pipeline parallelism is disabled, the lazy_inline scenario must be enabled by setting the environment variable:

  ```bash
  ENABLE_LAZY_INLINE_NO_PIPELINE=1
  ```

- Only support Ascend backend

#### Instruction for API

Fine-grained activations SWAP is enabled through the `swap_config` field in YAML configuration, which includes four functional interfaces: `swap`, `default_prefetch`, `layer_swap`, and `op_swap`. These interfaces allow users to flexibly enable SWAP for specific layers or specific operators within layers.  

> MindSpore framework currently decouples memory offloading and memory release. When activations are offloaded from the device side to the host side, the memory space occupied on the device side is not immediately released even after all data has been transferred. An explicit release operation is required instead. Before triggering the memory release, the system checks whether the activation offloading is complete. If not, the process will wait in place until the offloading finishes.

| Configuration Item | Type | Description |
|:--:|:--:|:---|
| swap | bool | Default False. When set to False, all four functional interfaces are disabled. When set to True, activations SWAP is enabled, and the system checks whether layer_swap and op_swap are None. If both are None, the default SWAP strategy is applied, which enables SWAP for the flash_attention operator across all layers. If either layer_swap or op_swap has a non-None value, the default policy is overridden, and SWAP is enabled according to the configurations in layer_swap and op_swap. |
| default_prefetch | int | Default 1 and only takes effect when swap=True, layer_swap=None, and op_swap=None. It controls the timing of releasing memory in forward phase and starting prefetch in backward phase of the default SWAP strategy. A larger `default_prefetch` delays memory release during the forward phase, keeping device memory occupied by activations locked for an extended period after offloading, preventing reuse by other data blocks. It also starts earlier prefetching from host to device during the backward phase, applying memory pressure prematurely. A smaller `default_prefetch` releases memory earlier in the forward phase but may introduce idle waiting for copy operations to complete. Additionally, delayed prefetch  in the backward phase may cause computation stalls if prefetching isn't finished before activation usage, impacting end-to-end performance. This interface allows users to fine-tune memory release and prefetch timing for optimal memory efficiency and performance.|
| layer_swap | list | Default None. When set to None, this interface is inactive. When the type is List, this interface contains several list elements of the Dict type. Each Dict element contains two keys: `backward_prefetch`, and `layers`, and provides the prefetch opportunity and layer index for enabling swap. |
| op_swap | list | Default None. When set to None, this interface is inactive. When the type is List, this interface contains several list elements of the Dict type. Each Dict element contains three keys: `op_name`, `backward_prefetch`, and `layers`, and provides the prefetch opportunity, operator name, and layer index for enabling swap. |

#### Used together with Recomputation

Fine-Grained Activations SWAP and Recomputation have coupling effects:

1. If any operator has both recomputation and SWAP enabled simultaneously, recomputation will take effect while SWAP will not.
2. For any operator with SWAP enabled, if its output is used by an operator with recomputation enabled, then SWAP for that operator will not take effect.
3. The YAML configuration interface for recomputation only supports enabling recomputation for a specific number of layers sequentially from front to back, rather than selecting specific layers or specific operators within layers. This means when using both SWAP and recomputation together, SWAP can only be enabled for later layers or operators within later layers, preventing full utilization of SWAP's benefits. Therefore, when and only when `swap=True`, the recomputation interface functionality will be adjusted as shown in the table below.

| Interface Name | Original Functionality | Functionality When Enabling SWAP |
|:--:|:---|:---|
| recompute | Determine the number of layers with recomputation enabled in each pipeline stage. | Pipeline stage-agnostic, only accepts bool/list type inputs. When bool type: enables recomputation for all layers; when list type: uses layer indices to enable recomputation for specific layers. |
| select_recompute | Determine the number of layers with recomputation enabled for specific operators in each pipeline stage. | Pipeline stage-agnostic, for each operator's key-value pair, only accepts bool/list type inputs. When bool type: enables recomputation for all layers; when list type: uses layer indices to enable recomputation for specific layers. |
| select_comm_recompute | Determine the number of layers with recomputation enabled for communication operators in each pipeline stage. | Pipeline stage-agnostic, only accepts bool/list type inputs. When bool type: enables recomputation for all layers; when list type: uses layer indices to enable recomputation for specific layers. |

### Cases of Fine-Grained Activations SWAP

This section demonstrates the usage of fine-grained activations SWAP using Llama2-7B training as an example.

#### Environmental Preparation

Download Mindformers, and prepare the pre-training dataset, such as wikitext.

#### Case 1: Default SWAP Strategy

Modify and supplement the recomputation and SWAP configurations in YAML as follows:

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

Execute the following script to launch single-node 8-NPU training, with the script's execution path being the root directory, requiring the user to specify the YAML file path(machine_ip needs to fill in the local environment IP address):

```bash
export GLOG_v=1
export MS_MEMORY_STATISTIC=1
export ENABLE_LAZY_INLINE_NO_PIPELINE=1
YAML_FILE=$1 # User specifies the YAML file path.
ROOT_PATH=`pwd`

bash ./scripts/msrun_launcher.sh "run_mindformer.py \
    --config ${ROOT_PATH}/${YAML_FILE} \
    --run_mode train \
    --use_parallel True" \
    8 8 <machine_ip> 8118 0 output/msrun False 300
```

After training completes, execute the command `cat output/msrun/worker_0.log | grep 'attention.flash_attention'` to check the execution status of the default SWAP strategy:

```text
-INFO - Set op_swap at layer 0: attention.flash_attention, value=10
-INFO - Set op_swap at layer 1: attention.flash_attention, value=10
-INFO - Set op_swap at layer 2: attention.flash_attention, value=10
-INFO - Set op_swap at layer 3: attention.flash_attention, value=10
```

The default SWAP strategy is executed successfully.

#### Case 2: Select Specific Layers to Enable SWAP

Modify and supplement the recomputation and SWAP configurations in YAML as follows:

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

Execute the following script to launch single-node 8-NPU training, with the script's execution path being the root directory, requiring the user to specify the YAML file path(machine_ip needs to fill in the local environment IP address):

```bash
export GLOG_v=1
export MS_MEMORY_STATISTIC=1
export ENABLE_LAZY_INLINE_NO_PIPELINE=1
YAML_FILE=$1 # User specifies the YAML file path.
ROOT_PATH=`pwd`

bash ./scripts/msrun_launcher.sh "run_mindformer.py \
    --config ${ROOT_PATH}/${YAML_FILE} \
    --run_mode train \
    --use_parallel True" \
    8 8 <machine_ip> 8118 0 output/msrun False 300
```

After training completes, execute the command `cat output/msrun/worker_0.log | grep 'Set layer swap at'` to check the execution status of the default SWAP strategy:

```text
-INFO - Set layer swap at layer 0 and value is: 20
-INFO - Set layer swap at layer 3 and value is: 20
```

The strategy of enabling SWAP for specific layers is executed successfully.

#### Case 3: Select Specific Operators within Layers to Enable SWAP

Modify and supplement the recomputation and SWAP configurations in YAML as follows:

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

Execute the following script to launch single-node 8-NPU training, with the script's execution path being the root directory, requiring the user to specify the YAML file path(machine_ip needs to fill in the local environment IP address):

```bash
export GLOG_v=1
export MS_MEMORY_STATISTIC=1
export ENABLE_LAZY_INLINE_NO_PIPELINE=1
YAML_FILE=$1 # User specifies the YAML file path.
ROOT_PATH=`pwd`

bash ./scripts/msrun_launcher.sh "run_mindformer.py \
    --config ${ROOT_PATH}/${YAML_FILE} \
    --run_mode train \
    --use_parallel True" \
    8 8 <machine_ip> 8118 0 output/msrun False 300
```

After training completes, execute the command `cat output/msrun/worker_0.log | grep 'Set op_swap at layer'` to check the execution status of the default SWAP strategy:

```text
-INFO - Set op_swap at layer 0: .attention, value=20
-INFO - Set op_swap at layer 1: .attention, value=20, .feed_forward, value=15
-INFO - Set op_swap at layer 2: .attention, value=20, .feed_forward, value=15
-INFO - Set op_swap at layer 3: .attention, value=10
```

The strategy of enabling SWAP for specific operators within layers is executed successfully.

#### Case 4: Use Fine-Grained Activations SWAP together with Recomputation

Modify and supplement the recomputation and SWAP configurations in YAML as follows:

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

Execute the following script to launch single-node 8-NPU training, with the script's execution path being the root directory, requiring the user to specify the YAML file path(machine_ip needs to fill in the local environment IP address):

```bash
export GLOG_v=1
export MS_MEMORY_STATISTIC=1
export ENABLE_LAZY_INLINE_NO_PIPELINE=1
YAML_FILE=$1 # User specifies the YAML file path.
ROOT_PATH=`pwd`

bash ./scripts/msrun_launcher.sh "run_mindformer.py \
    --config ${ROOT_PATH}/${YAML_FILE} \
    --run_mode train \
    --use_parallel True" \
    8 8 <machine_ip> 8118 0 output/msrun False 300
```

After training completes, execute the command `cat output/msrun/worker_0.log | grep 'Set op_swap at layer' -C 1` to check the execution status of the default SWAP strategy:

```text
-INFO - Set select recompute at layer 0: feed_forward
-INFO - Set op_swap at layer 0: .attention, value=20
-INFO - Set op_swap at layer 1: .attention, value=20, .feed_forward, value=15
-INFO - Set op_swap at layer 2: .attention, value=20, .feed_forward, value=15
-INFO - Set select recompute at layer 3: feed_forward
-INFO - Set op_swap at layer 3: .attention, value=10
```

The strategy of enabling fine-grained activations SWAP together with recomputation is executed successfully.
