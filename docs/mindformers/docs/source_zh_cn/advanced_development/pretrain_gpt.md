# 动态图并行

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/advanced_development/pretrain_gpt.md)

## 概述

本教程演示如何使用MindSpore Transformers动态图并行框架训练GPT模型，此框架支持张量并行、流水线并行、序列并行等并行场景，还有支持使用分布式优化器动态学习率等场景，帮助开发者快速、便捷地构建和训练基于动态图并行框架的GPT预训练模型。

## 操作实践

下面基于Ascend平台，进行GPT模型训练。

### 样例代码参考

目录结构如下：

```text
└─ gpt
    ├─ pretrain_gpt.py
    ├─ pretrain_gpt.sh
    └─ pretrain_gpt_7B.yaml
    ...
```

其中，`pretrain_gpt.py`是环境配置、模型对象创建及训练的脚本。`pretrain_gpt.sh`是启动执行脚本。`pretrain_gpt_7B.yaml`是配置项。

### 模型结构

GPT以`Transformer`模型为主要架构，网络结构主要围绕`Transformer`的基本构建块构建。

在模型中，初始化五个参数，`config`是模型配置项（在yaml文件的`model_config`中），`num_tokentypes`指定embedding的类型，`parallel_output`用来确认是否输出每一个并行Tensor的输出，`pre_process`和`post_process`分别指定是否为第一阶段和最后一阶段。

调用的`get_language_model`是一个基于`Transformer`模型的接口，详情请看`get_language_model`的api文档。

注意：数据集返回值要与模型定义的前向过程所需要的参数相对应。

```python
from mindformers.experimental.parallel_core.pynative.transformer.module import Module
from mindformers.experimental.parallel_core.pynative.transformer.language_model import get_language_model
from mindformers.experimental.parallel_core.pynative.transformer import ParallelLMLogits
from mindformers.experimental.parallel_core.pynative.training.loss_func import VocabParallelCrossEntropy


class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2
    no_mask = 3
    padding_causal = 4


attn_mask_type_mapping = {
    "padding": AttnMaskType.padding,
    "causal": AttnMaskType.causal,
}


class GPTModel(Module):
    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True):
        super().__init__(config=config,\
                         share_embeddings_and_output_weights=not config.untie_embeddings_and_output_weights)

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.untie_embeddings_and_output_weights = config.untie_embeddings_and_output_weights
        self.fp16_lm_cross_entropy = config.fp16_lm_cross_entropy

        self.set_model_key()
        encoder_attn_mask_type = None
        if config.encoder_attn_mask_type is not None:
            encoder_attn_mask_type = attn_mask_type_mapping.get(config.encoder_attn_mask_type)
            if encoder_attn_mask_type is None:
                raise ValueError(f"encoder_attn_mask_type must be one of {attn_mask_type_mapping.keys()}, but got"
                                 f"{config.encoder_attn_mask_type}")

        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=encoder_attn_mask_type,
            pre_process=self.pre_process,
            post_process=self.post_process)

        if self.post_process:
            self.parallel_lm_logits = ParallelLMLogits(config=config,
                                                       bias=False,
                                                       compute_dtype=config.compute_dtype)
            self.loss = VocabParallelCrossEntropy()

        if not config.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """ set input_tensor to model """
        self.language_model.set_input_tensor(input_tensor)

    def set_model_key(self):
        """ set model key for differentiate PipelineCell process """
        self.model_key = "gpt3"

    def construct(self, input_ids, position_ids, attention_mask, loss_mask,
                  retriever_input_ids=None,
                  retriever_position_ids=None,
                  retriever_attn_mask=None,
                  labels=None, tokentype_ids=None, inference_params=None):
        """ gpt model forward """
        # use RoPE
        position_ids = None
        retriever_input_ids = None
        retriever_position_ids = None
        retriever_attn_mask = None
        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            retriever_input_ids=retriever_input_ids,
            retriever_position_ids=retriever_position_ids,
            retriever_attn_mask=retriever_attn_mask,
            inference_params=inference_params)
        if self.post_process:
            return post_language_model_processing(
                self.parallel_lm_logits, self.loss,
                lm_output, labels,
                self.language_model.output_layer.weight if\
                    self.untie_embeddings_and_output_weights else self.shared_embedding_or_output_weight(),
                self.parallel_output,
                self.fp16_lm_cross_entropy,
                loss_mask)
        else:
            return lm_output
```

当`post_process`为`True`时，需要对语言模型的输出`lm_output`进行后处理，输出损失和预测结果。

```python
import mindspore.common.dtype as mstype

def post_language_model_processing(parallel_lm_logits, loss_fn, lm_output, labels, logit_weights,
                                   parallel_output, fp16_lm_cross_entropy, loss_mask):
    """ gpt model post process forward """
    output = parallel_lm_logits(lm_output, logit_weights, parallel_output)

    if labels is None:
        return output

    labels = labels
    loss_mask = loss_mask.reshape(-1)

    if fp16_lm_cross_entropy:
        if output.dtype != mstype.float16:
            raise ValueError(f"When fp16_lm_cross_entropy=True, output should be float16, but got {output.dtype}")
        loss = loss_fn(output, labels, loss_mask)
    else:
        loss = loss_fn(output.astype(mstype.float32), labels)
    token_nums = loss_mask.sum()
    loss_mask = loss_mask.astype(mstype.float32)
    loss = ops.sum(loss * loss_mask.float()) / loss_mask.sum()
    return loss, output, token_nums
```

### 动态图并行训练配置

动态图并行的配置项通过yaml文件来读取，并分为不同种类，包括训练配置、并行配置、模型配置等，接下来简单介绍一下大模型训练需要的基本配置。

#### 配置训练参数（training_config）

```yaml
training_config:
  seed: 42                                        # 固定随机性用的种子
  output_dir: './output'                          # 输出目录，用于储存checkpoints和日志等
  training_iters: 10                              # 训练迭代次数
  log_interval: 1                                 # 日志打印的频率
  save_interval: null                             # 储存checkpoints的频率
  loss_scale: 4096                                # loss scale的初始值
  grad_clip_kwargs:
    grad_clip_type: "ClipGlobalNorm"              # 梯度裁剪的方法，可选："ClipGlobalNorm"或者"GradClipByValue"
    clip_value: 1.0
  loss_reduction: "mean"                          # loss reduction的方法，可选："mean"或者"sum"
  loss_func_kwargs:
    loss_func_type: "VocabParallelCrossEntropy"   # 损失函数，可选: "VocabParallelCrossEntropy"或者"CrossEntropyLoss"
  use_distributed_optimizer: True                 # 是否使用分布式优化器
```

#### 配置并行模式（parallel_config）

```yaml
parallel_config:
  tensor_model_parallel_size: 1                    # 张量并行
  pipeline_model_parallel_size: 1                  # 流水线并行
  expert_model_parallel_size: 1                    # 专家并行
  virtual_pipeline_model_parallel_size: null       # 虚拟流水线并行
  sequence_parallel: False                         # 序列并行
```

#### 配置模型参数（gpt_config）

```yaml
model_config:
  params_dtype: "float32"                          # 参数初始化类型
  compute_dtype: "bfloat16"                        # 计算时使用的类型
  position_embedding_type: 'rope'                  # 位置编码的类型，可选："rope"或者"absolute"
  untie_embeddings_and_output_weights: True        # embedding层和head层是否不共享权重
  # 配置GPT 7B模型
  num_layers: 6                                    # Transformer层数
  hidden_size: 4096                                # 隐藏层的大小
  ffn_hidden_size: 11008                           # 前馈神经网络隐藏层大小
  num_attention_heads: 32                          # 注意力头的数量
```

GPT模型当前有三种不同规格的配置：7B、13B和70B。

```yaml
7B:
  num_layers: 32
  hidden_size: 4096
  ffn_hidden_size: 11008
  num_attention_heads: 32
13B:
  num_layers: 40
  hidden_size: 5120
  ffn_hidden_size: 13824
  num_attention_heads: 40
70B:
  num_layers: 80
  hidden_size: 8192
  ffn_hidden_size: 28672
  num_attention_heads: 64
  group_query_attention: True
  num_query_groups: 8
```

#### 数据集配置（dataset_config）

```yaml
dataset_config:
  batch_size: 1                                    # 一次迭代从数据集中取出的数据大小
  micro_batch_num: 2                               # 微批次个数
  dataset_dir: './dataset'                         # 数据集所在目录
  shuffle: False                                   # 是否打乱顺序
```

#### 优化器配置（optimizer_config）

```yaml
optimizer_config:
  optimizer_type: "AdamW"                          # 优化器类型，可选："AdamW", "Adam", "SGD", "Came", "mint.AdamW"及"SpeedAdamW"
  betas:                                           # 优化器输入参数
    - 0.9
    - 0.95
  eps: 1.e-8
  learning_rate: 1.25e-6                           # 初始学习率
  weight_decay: 1.e-1                              # 权重衰减系数
  learning_rate_scheduler_kwargs:                  # 学习率调整策略
    warmup_steps: 200
    decay_steps: 2000
    use_cosine: True
    end_learning_rate: 1.25e-7
```

### 模型训练配置解析

在pretrain_gpt.py里对传入的yaml配置文件进行解析，可以得到训练配置、模型配置、优化器配置、并行策略配置以及数据集配置。

```python
import argparse
from mindformers.experimental.parallel_core.pynative.config import (
    init_configs_from_yaml
)

def get_arg_parser():
    """get argument parser"""
    parser = argparse.ArgumentParser(description="Train gpt model")
    parser.add_argument("--config_path", type=str, default="pretrain_gpt.yaml", help="The path to the config file.")
    parser.add_argument("--run_cmd", type=str, default="", help="running cmd.")
    parser.add_argument("--model_type", type=str, default="gpt_config", help="Input model config.")
    return parser
parser = get_arg_parser()
args = parser.parse_args()

all_config = init_configs_from_yaml(args.config_path)

training_config = all_config.training_config
model_config = all_config.model_config
optimizer_config = all_config.optimizer_config
parallel_config = all_config.parallel_config
dataset_config = all_config.dataset_config
```

### 通信配置

通过set_context接口可以指定运行模式、运行设备、运行卡号等。并行脚本还需指定并行模式`parallel_mode`为数据并行模式，并通过init根据不同的设备需求初始化HCCL、NCCL或者MCCL通信。指定平台：设置`device_target`为`Ascend`。调试阶段可以使用`set_context(pynative_synchronize=True)`开启同步模式，更准确地定位报错位置。

```python
import mindspore as ms


def set_parallel_context(parallel_config):
    init()
    initialize_model_parallel(
        tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=parallel_config.virtual_pipeline_model_parallel_size,
    )
    logger.info(
        f"dp {get_data_parallel_world_size()} | "
        f"pp {parallel_config.pipeline_model_parallel_size} | "
        f"tp {parallel_config.tensor_model_parallel_size} | "
        f"sp {parallel_config.sequence_parallel} | "
        f"vpp {parallel_config.virtual_pipeline_model_parallel_size}"
    )


def set_seed(seed):
    # set global seed, np seed, and dataset seed
    ms.set_seed(seed)
    # set rng seed
    ms.manual_seed(seed)


ms.set_context(mode=ms.PYNATIVE_MODE)
ms.set_device(device_target="Ascend")
set_parallel_context(parallel_config)
set_seed(training_config.seed)
```

### 创建网络对象

从模型库获取GPT模型，根据配置文件创建网络模型对象。通过`set_weight_decay`来为不同参数设置不同的权重衰减系数，这个函数会将参数分为两组，一组应用特定的权重衰减值，另一组权重衰减为`0`，然后返回一个包含参数分组信息的列表，赋值给`group_params`变量。调用`get_optimizer`函数，传入`optimizer_config`（优化器配置）、`training_config`（训练配置）、`group_params`（前面得到的参数分组信息）、`network_with_loss`（包含模型和损失的对象）以及一个梯度归约操作（从`training_config.loss_reduction`获取），返回一个优化器对象，并赋值给`optimizer`变量。
创建一个`TrainOneStepCell`对象，它通常用于在训练过程中执行一步优化。传入`network_with_loss`、`optimizer`及配置作为参数，并将其赋值给train_one_step_cell变量。

完整的创建网络对象代码：

```python
from mindformers.experimental.parallel_core.pynative.optimizer import get_optimizer
from mindformers.experimental.parallel_core.pynative.training import get_model
from mindformers.experimental.parallel_core.pynative.training import TrainOneStepCell
from mindformers.experimental.parallel_core.models import GPTModel


def decay_filter(x):
    return "norm" not in x.name.lower() and "bias" not in x.name.lower()


def set_weight_decay(params, weight_decay=1e-1):
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = []
    if decay_params:
        group_params.append({"params": decay_params, "weight_decay": weight_decay})
    if other_params:
        group_params.append({"params": other_params, "weight_decay": 0.0})
    return group_params


def model_provider_func(pre_process=True, post_process=True):
    network_with_loss = GPTModel(
        model_config, pre_process=pre_process, post_process=post_process
    )
    return network_with_loss

network_with_loss = get_model(model_provider_func, training_config)

group_params = set_weight_decay(network_with_loss.trainable_params(), optimizer_config.weight_decay)
optimizer = get_optimizer(
    optimizer_config,
    training_config,
    group_params,
    network_with_loss,
    grad_allreduce_op=training_config.loss_reduction
)

train_one_step_cell = TrainOneStepCell(network_with_loss, optimizer, None, training_config, model_config)
```

### 加载数据集及执行训练

```python
from dataset import get_dataset
from mindformers.experimental.parallel_core.pynative.training import train

train_dataset_iterator, val_dataset_iterator = get_dataset(dataset_config)
train(
    train_one_step_cell,
    train_dataset_iterator,
    training_config,
    val_dataset_iterator,
    metrics,
    evaluation,
)
```

### 运行训练脚本

```bash
bash pretrain_gpt.sh xx.yaml
```

若不指定xx.yaml，则默认为pretrain_gpt_7B.yaml。

训练脚本`pretrain_gpt.sh`详细解析如下：

#### 设置环境变量

`HCCL_BUFFSIZE=200`设置两个NPU之间共享数据的缓存区大小为200M；`HCCL_EXEC_TIMEOUT=600`设置设备间执行时同步的等待时间为10分钟。`ASCEND_RT_VISIBLE_DEVICES`指定了可见的设备编号，这里设置为设备`0`号卡。

```bash
export HCCL_BUFFSIZE=200
export HCCL_EXEC_TIMEOUT=600
export ASCEND_RT_VISIBLE_DEVICES='0'
```

#### 设置端口号

```bash
port=8828
```

如果之前的配置异常退出，可以使用如下代码进行清理。

```bash
PIDS=$(sudo lsof -i :$port | awk 'NR>1 {print $2}')
if [ -n "$PIDS" ]; then
    for pid in $PIDS; do
        kill -9 $pid
        echo "Killed process $pid"
    done
else
    echo "No processes found listening on port $port."
fi
```

#### 设置日志存储路径

获取当前脚本所在的目录路径并存储在`project_dir`变量中，同时设置日志路径变量`log_path="msrun_log"`。先删除名为`msrun_log`的目录（如果存在），然后重新创建这个目录。

```bash
project_dir=$(cd "$(dirname "$0")" || exit; pwd)
log_path="msrun_log"

rm -rf "${log_path}"
mkdir "${log_path}"
```

#### 设置可用设备数量

```bash
# 计算设备数量
IFS=',' read -r -a devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
work_num=${#devices[@]}
```

#### 获取配置文件

尝试从命令行参数中获取配置文件路径，如果没有提供命令行参数，则使用默认的配置文件 "pretrain_gpt_7B.yaml"。

```bash
config_path=$1
if [ -z "$config_path" ]; then
    config_path="pretrain_gpt_7B.yaml"
fi
```

#### 以msrun模式执行训练脚本

```bash
msrun --worker_num "$work_num" --local_worker_num="$work_num" --master_port=$port --log_dir="$log_path" --join=True --cluster_time_out=300 pretrain_gpt.py --config_path="${config_path}"
```

#### 运行结果

接下来通过命令调用对应的脚本。

```bash
bash pretrain_gpt.sh
```

执行完后，日志文件保存到`output`目录下，其中部分文件目录结构如下：

```text
└─ output
    └─ log
        ├─ rank_0
        |   ├─ info.log
        |   └─ error.log
        ├─ rank_1
        |   ├─ info.log
        |   └─ error.log
    ...
```

关于Loss部分结果保存在`output/log/rank_*/info.log`中，示例如下：

```text
train: Epoch:0, Step:5, Loss: 10.341485, Finite_grads: True, Loss_scale: 4096.0, Learning_rate: (1.250000e-06,1.250000e-06,), Time: 1403.24 ms
train: Epoch:0, Step:6, Loss: 10.38118, Finite_grads: True, Loss_scale: 4096.0, Learning_rate: (1.250000e-06,1.250000e-06,), Time: 1378.19 ms
train: Epoch:0, Step:7, Loss: 10.165115, Finite_grads: True, Loss_scale: 4096.0, Learning_rate: (1.250000e-06,1.250000e-06,), Time: 1370.32 ms
train: Epoch:0, Step:8, Loss: 10.039211, Finite_grads: True, Loss_scale: 4096.0, Learning_rate: (1.250000e-06,1.250000e-06,), Time: 1386.89 ms
train: Epoch:0, Step:9, Loss: 10.040031, Finite_grads: True, Loss_scale: 4096.0, Learning_rate: (1.250000e-06,1.250000e-06,), Time: 1475.95 ms
...
```
