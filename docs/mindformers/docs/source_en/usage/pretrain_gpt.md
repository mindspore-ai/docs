# Dynamic Graph Parallelism

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindformers/docs/source_en/usage/pretrain_gpt.md)

## Overview

This tutorial demonstrates how to use MindSpore Transformers dynamic graph parallel framework to train GPT models. This framework supports tensor parallel, pipeline parallel, sequence parallel and other parallel scenarios, as well as support for the use of distributed optimizer dynamic learning rate and other scenarios, to help developers quickly and easily build and train GPT pre-training models based on dynamic graph parallel framework.

## Operating Practice

The following GPT model training is based on Ascend platform.

### Sample Code Reference

The directory structure is as follows:

```text
└─ gpt
    ├─ pretrain_gpt.py
    ├─ pretrain_gpt.sh
    └─ pretrain_gpt_7B.yaml
    ...
```

Among them, `pretrain_gpt.py` is the script for environment configuration, model object creation and training. `pretrain_gpt.sh` is the startup execution script. `pretrain_gpt_7B.yaml` is the configuration item.

### Model Structure

GPT uses the `Transformer` model as its main architecture, and the network structure is mainly built around the basic building blocks of the `Transformer`.

In the model, five parameters are initialized, `config` is the model configuration item (in the `model_config` of the yaml file), `num_tokentypes` specifies the type of embedding, `parallel_output` is used to confirm whether to output the output of each parallel Tensor, `pre_ process` and `post_process` specify whether it is the first and last stage, respectively.

The called `get_language_model` is an interface based on the `Transformer` model, see the api documentation for `get_language_model` for details.

Note: The dataset return values are to correspond to the parameters required by the forward process defined by the model.

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

When `post_process` is set to `True`, the output `lm_output` of the language model needs to be post-processed to output losses and predictions.

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

### Dynamic Graph Parallel Training Configuration

Configuration items for dynamic graph parallel are read through a yaml file and are categorized into different types, including training configuration, parallel configuration, and model configuration. The next section briefly describes the basic configurations needed for large model training.

#### training_config

```yaml
training_config:
  seed: 42                                        # Seeds for fixed randomness
  output_dir: './output'                          # Output directory for storing checkpoints, logs, etc.
  training_iters: 10                              # The number of training iterations
  log_interval: 1                                 # Frequency of log prints
  save_interval: null                             # Frequency of storing checkpoints
  loss_scale: 4096                                # Initial value of loss scale
  grad_clip_kwargs:
    grad_clip_type: "ClipGlobalNorm"              # Gradient cropping methods, optional: "ClipGlobalNorm" or  "GradClipByValue"
    clip_value: 1.0
  loss_reduction: "mean"                          # loss reduction methods, optional: "mean" or "sum"
  loss_func_kwargs:
    loss_func_type: "VocabParallelCrossEntropy"   # Loss function, optional: "VocabParallelCrossEntropy" or "CrossEntropyLoss"
  use_distributed_optimizer: True                 # Whether to use a distributed optimizer
```

#### parallel_config

```yaml
parallel_config:
  tensor_model_parallel_size: 1                    # Tensor parallel
  pipeline_model_parallel_size: 1                  # Pipeline parallel
  expert_model_parallel_size: 1                    # Expert parallel
  virtual_pipeline_model_parallel_size: null       # Virtual pipeline parallel
  sequence_parallel: False                         # Sequence parallel
```

#### gpt_config

```yaml
model_config:
  params_dtype: "float32"                          # Parameter initialization type
  compute_dtype: "bfloat16"                        # Types used in calculations
  position_embedding_type: 'rope'                  # Type of location code, optional: "rope" or "absolute"
  untie_embeddings_and_output_weights: True        # Whether the embedding layer and the head layer do not share weights
  # Configure the GPT 7B model
  num_layers: 6                                    # The number of Transformer layers
  hidden_size: 4096                                # Size of the hidden layer
  ffn_hidden_size: 11008                           # Size of feedforward neural network hidden layer
  num_attention_heads: 32                          # Number of attention heads
```

The GPT model is currently available in three different sizes of configurations: 7B, 13B and 70B.

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

#### dataset_config

```yaml
dataset_config:
  batch_size: 1                                    # Size of data removed from the dataset in one iteration
  micro_batch_num: 2                               # Number of micro batches
  dataset_dir: './dataset'                         # Catalog where the dataset is located
  shuffle: False                                   # Whether to break the order
```

#### optimizer_config

```yaml
optimizer_config:
  optimizer_type: "AdamW"                          # Optimizer types, optional: "AdamW", "Adam", "SGD", "Came", "mint.AdamW" and "SpeedAdamW"
  betas:                                           # Optimizer input parameters
    - 0.9
    - 0.95
  eps: 1.e-8
  learning_rate: 1.25e-6                           # Initial learning rate
  weight_decay: 1.e-1                              # Weight decay factor
  learning_rate_scheduler_kwargs:                  # Learning rate adjustment strategy
    warmup_steps: 200
    decay_steps: 2000
    use_cosine: True
    end_learning_rate: 1.25e-7
```

### Model Training Configuration Parsing

The passing yaml configuration file is parsed in pretrain_gpt.py to get the training configuration, model configuration, optimizer configuration, parallel strategy configuration, and dataset configuration.

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

### Communication Configuration

The set_context interface allows you to specify the run mode, run device, and run card number. The parallel script also needs to specify the parallel mode `parallel_mode` as the data parallel mode and initialize the HCCL, NCCL or MCCL communication through init depending on the different device requirements. Specify platform: set `device_target` to `Ascend`. You can use `set_context(pynative_synchronize=True)` in debugging phase to enable synchronization mode and locate the error report location more accurately.

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

### Creating Network Objects

Get the GPT model from the model library and create a network model object based on the configuration file. Set different weight decay coefficients for different parameters via `set_weight_decay`, a function that divides the parameters into two groups, one with a specific weight decay value applied and the other with a weight decay of `0`, and returns a list containing information about the grouping of the parameters assigned to the `group_params` variable. The `get_optimizer` function is called, passing in `optimizer_config` (optimizer configuration), `training_config` (training configuration), `group_params` (information about the grouping of parameters obtained earlier), `network_with_loss` (an object containing the model and loss ), and a gradient reduction operation (obtained from `training_config.loss_reduction`) that returns an optimizer object and assigns it to the `optimizer` variable.
Create a `TrainOneStepCell` object, which is typically used to perform one-step optimization during training. Pass `network_with_loss`, `optimizer` and configuration as parameters and assign them to the train_one_step_cell variable.

Complete code for creating network objects:

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

### Loading the Dataset and Performing Training

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

### Running the Training Script

```bash
bash pretrain_gpt.sh xx.yaml
```

If xx.yaml is not specified, it defaults to pretrain_gpt_7B.yaml.

The training script `pretrain_gpt.sh` is parsed in detail below:

#### Setting Environment Variables

`HCCL_BUFFSIZE=200` sets the size of the buffer for sharing data between the two NPUs to 200M; `HCCL_EXEC_TIMEOUT=600` sets the wait time for synchronization during execution between the devices to 10 minutes. `ASCEND_RT_VISIBLE_DEVICES` specifies the visible device number, here set to device `0` card.

```bash
export HCCL_BUFFSIZE=200
export HCCL_EXEC_TIMEOUT=600
export ASCEND_RT_VISIBLE_DEVICES='0'
```

#### Setting Port Number

```bash
port=8828
```

If the previous configuration exits abnormally, you can use the following code to clean it up.

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

#### Setting Log Storage Path

Get the path to the directory where the current script is located and store it in the `project_dir` variable, and set the log path variable `log_path=“msrun_log”`. Delete the directory named `msrun_log` (if it exists) and recreate it.

```bash
project_dir=$(cd "$(dirname "$0")" || exit; pwd)
log_path="msrun_log"

rm -rf "${log_path}"
mkdir "${log_path}"
```

#### Setting the Number of Available Devices

```bash
# Calculate the number of devices
IFS=',' read -r -a devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
work_num=${#devices[@]}
```

#### Getting the Configuration File

Try to get the configuration file path from the command line arguments, if no command line arguments are provided, the default configuration file “pretrain_gpt_7B.yaml” is used.

```bash
config_path=$1
if [ -z "$config_path" ]; then
    config_path="pretrain_gpt_7B.yaml"
fi
```

#### Executing Training Scripts in msrun Mode

```bash
msrun --worker_num "$work_num" --local_worker_num="$work_num" --master_port=$port --log_dir="$log_path" --join=True --cluster_time_out=300 pretrain_gpt.py --config_path="${config_path}"
```

#### Running Results

Next, the corresponding script is invoked by command.

```bash
bash pretrain_gpt.sh
```

After execution, the log files are saved to the `output` directory, where some of the files have the following directory structure:

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

The results on the Loss section are saved in `output/log/rank_*/info.log`, example below:

```text
train: Epoch:0, Step:5, Loss: 10.341485, Finite_grads: True, Loss_scale: 4096.0, Learning_rate: (1.250000e-06,1.250000e-06,), Time: 1403.24 ms
train: Epoch:0, Step:6, Loss: 10.38118, Finite_grads: True, Loss_scale: 4096.0, Learning_rate: (1.250000e-06,1.250000e-06,), Time: 1378.19 ms
train: Epoch:0, Step:7, Loss: 10.165115, Finite_grads: True, Loss_scale: 4096.0, Learning_rate: (1.250000e-06,1.250000e-06,), Time: 1370.32 ms
train: Epoch:0, Step:8, Loss: 10.039211, Finite_grads: True, Loss_scale: 4096.0, Learning_rate: (1.250000e-06,1.250000e-06,), Time: 1386.89 ms
train: Epoch:0, Step:9, Loss: 10.040031, Finite_grads: True, Loss_scale: 4096.0, Learning_rate: (1.250000e-06,1.250000e-06,), Time: 1475.95 ms
...
```
