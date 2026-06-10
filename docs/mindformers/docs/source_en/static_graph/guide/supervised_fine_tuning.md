<!-- __SG_DEPRECATED_BANNER__ -->

```{admonition} Deprecated
:class: warning

This page belongs to the **Static Graph (GRAPH_MODE) Implementation** section and has been marked as deprecated. New features are primarily being developed in the "r2.0.0 Dynamic Graph Implementation" section. Please refer to the dynamic graph documentation first.
```

# Supervised Fine-Tuning (SFT)

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/static_graph/guide/supervised_fine_tuning.md)

## Overview

SFT (Supervised Fine-Tuning) adopts the concept of supervised learning, referring to the process of adjusting some or all parameters of a pre-trained model to better adapt it to specific tasks or datasets.

MindSpore Transformers supports two SFT fine-tuning methods: full-parameter fine-tuning and LoRA fine-tuning. Full-parameter fine-tuning involves updating all parameters during training, suitable for large-scale data refinement, offering optimal task adaptability but requiring significant computational resources. LoRA fine-tuning updates only a subset of parameters, consuming less memory and training faster than full-parameter fine-tuning, though its performance may be inferior in certain tasks.

## Basic Process of SFT Fine-Tuning

Combining practical operations, SFT fine-tuning can be broken down into the following steps:

### 1. Weight Preparation

Before fine-tuning, the weight files of the pre-trained model need to be prepared. MindSpore Transformers supports loading [safetensors weights](https://www.mindspore.cn/mindformers/docs/en/master/feature/safetensors.html), enabling direct loading of model weights downloaded from the Hugging Face model hub.

### 2. Dataset Preparation

MindSpore Transformers currently supports datasets in [Hugging Face format](https://www.mindspore.cn/mindformers/docs/en/master/feature/dataset.html#hugging-face-dataset) and [MindRecord format](https://www.mindspore.cn/mindformers/docs/en/master/feature/dataset.html#mindrecord-dataset) for the fine-tuning phase. Users can prepare data according to task requirements.

### 3. Configuration File Preparation

Fine-tuning tasks are uniformly controlled through [configuration files](https://www.mindspore.cn/mindformers/docs/en/master/feature/configuration.html), allowing users to flexibly adjust [model training hyperparameters](https://www.mindspore.cn/mindformers/docs/en/master/feature/training_hyperparameters.html). Additionally, fine-tuning performance can be optimized using [distributed parallel training](https://www.mindspore.cn/mindformers/docs/en/master/feature/parallel_training.html), [memory optimization features](https://www.mindspore.cn/mindformers/docs/en/master/feature/memory_optimization.html), and [other training features](https://www.mindspore.cn/mindformers/docs/en/master/feature/other_training_features.html).

### 4. Launching the Training Task

MindSpore Transformers provides a [one-click startup script](https://www.mindspore.cn/mindformers/docs/en/master/feature/start_tasks.html) to initiate fine-tuning tasks. During training, [logs](https://www.mindspore.cn/mindformers/docs/en/master/feature/logging.html) and [visualization tools](https://www.mindspore.cn/mindformers/docs/en/master/feature/monitor.html) can be used to monitor the training process.

### 5. Model Saving

Checkpoints are saved during training, or model weights are saved to a specified path upon completion. Currently, weights can be saved in [Safetensors format](https://www.mindspore.cn/mindformers/docs/en/master/feature/safetensors.html) or [Ckpt format](https://www.mindspore.cn/mindformers/docs/en/master/feature/ckpt.html), which can be used for resumed training or further fine-tuning.

### 6. Fault Recovery

To handle exceptions such as training interruptions, MindSpore Transformers offers [training high availability](https://www.mindspore.cn/mindformers/docs/en/master/feature/high_availability.html) like last-state saving and automatic recovery, as well as [checkpoint-based resumed training](https://www.mindspore.cn/mindformers/docs/en/master/feature/resume_training.html), enhancing training stability.

## Full-Parameter Fine-Tuning with MindSpore Transformers

### Selecting a Pre-Trained Model

MindSpore Transformers currently supports mainstream large-scale models in the industry. This guide uses the Qwen3-8B model as an example.

If you need to use other models or learn recommended configurations for different scenarios, refer to the [Model Library](https://www.mindspore.cn/mindformers/docs/en/master/introduction/models.html).

### Downloading Model Weights

MindSpore Transformers supports loading Hugging Face model weights, enabling direct loading of weights downloaded from the Hugging Face model hub. For details, refer to [MindSpore Transformers-Safetensors Weights](https://www.mindspore.cn/mindformers/docs/en/master/feature/safetensors.html).

| Model Name | Hugging Face Weight Download Link                     |
|:-----------| :---------------------------------------------------: |
| Qwen3-8B   | [Link](https://huggingface.co/Qwen/Qwen3-8B)        |

### Dataset Preparation

MindSpore Transformers supports online loading of Hugging Face datasets. For details, refer to [MindSpore Transformers-Dataset-Hugging Face Dataset](https://www.mindspore.cn/mindformers/docs/en/master/feature/dataset.html#hugging-face-dataset).

This guide uses [llm-wizard/alpaca-gpt4-data](https://huggingface.co/datasets/llm-wizard/alpaca-gpt4-data) as the fine-tuning dataset.

| Dataset Name                | Applicable Phase | Download Link                                                      |
| :-------------------------- | :--------------: | :----------------------------------------------------------------: |
| llm-wizard/alpaca-gpt4-data | Fine-Tuning      | [Link](https://huggingface.co/datasets/llm-wizard/alpaca-gpt4-data) |

### Executing the Fine-Tuning Task

#### Single-NPU Training

First, prepare the configuration file. This guide provides a fine-tuning configuration file for the Qwen3-8B model, `finetune_qwen3.yaml`, available for download from the [AtomGit repository](https://atomgit.com/mindspore/mindformers/blob/master/configs/qwen3/finetune_qwen3.yaml).

> Due to limited single-NPU memory, the `num_layers` in the configuration file is set to 4, used as an example only.

Then, modify the parameters in the configuration file based on actual conditions, mainly including:

```yaml
pretrained_model_dir: '/path/to/Qwen3-8B'
...
train_dataset: &train_dataset
  ...
  data_loader:
    type: HFDataLoader
    path: "llm-wizard/alpaca-gpt4-data-zh" # An Alpaca-style dataset. Ensure the network can access Hugging Face for automatic dataset download.
    # path: "json"  # If using a local JSON file for offline dataset loading, uncomment the next two lines and comment out the line above
    # data_files: '/path/to/alpaca_gpt4_data_zh.json'
    ...
    handler:
      - type: take # Invoke the `take` method from the datasets library to fetch the first n samples for demonstration
        n: 2000    # Take the first 2000 samples for demonstration. Remove this line and the one above during actual use.

model:
  model_config:
    num_hidden_layers: 4
    ...
parallel_config:
  data_parallel: 1
  model_parallel: 1
  pipeline_stage: 1
  use_seq_parallel: False
  micro_batch_num: 1
```

Run `run_mindformer.py` to start the single-NPU fine-tuning task. The command is as follows:

```shell
python run_mindformer.py \
 --config configs/qwen3/finetune_qwen3.yaml \
 --use_parallel False \
 --run_mode finetune
```

Parameter descriptions:

```text
config:            Model configuration file
use_parallel:      Whether to enable parallel training
run_mode:          Running mode, train: training, finetune: fine-tuning, predict: inference
```

#### Single-Node Training

First, prepare the configuration file. This guide provides a fine-tuning configuration file for the Qwen3-8B model, `finetune_qwen3.yaml`, available for download from the [AtomGit repository](https://atomgit.com/mindspore/mindformers/blob/master/configs/qwen3/finetune_qwen3.yaml).

Then, modify the parameters in the configuration file based on actual conditions, mainly including:

```yaml
pretrained_model_dir: '/path/to/Qwen3-8B'
...
train_dataset: &train_dataset
  ...
  data_loader:
    type: HFDataLoader
    path: "llm-wizard/alpaca-gpt4-data-zh" # An Alpaca-style dataset. Ensure the network can access Hugging Face for automatic dataset download.
    # path: "json"  # If using a local JSON file for offline dataset loading, uncomment the next two lines and comment out the line above
    # data_files: '/path/to/alpaca_gpt4_data_zh.json'
    ...
    handler:
      - type: take # Invoke the `take` method from the datasets library to fetch the first n samples for demonstration
        n: 2000    # Take the first 2000 samples for demonstration. Remove this line and the one above during actual use.
parallel_config:
  data_parallel: 1
  model_parallel: 4
  pipeline_stage: 2
  micro_batch_num: 2
```

Run the following msrun startup script for 8-NPU distributed training:

```bash
total_rank_num=8
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config configs/qwen3/finetune_qwen3.yaml \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode finetune" \
$total_rank_num
```

Parameter descriptions:

```text
config:            Model configuration file
auto_trans_ckpt:   Whether to automatically convert the weight file format
use_parallel:      Whether to enable parallel training
run_mode:          Running mode, train: training, finetune: fine-tuning, predict: inference
```

After task completion, a checkpoint folder will be generated in the mindformers/output directory, and the model files will be saved in this folder.

#### Multi-Node Training

Multi-Node, multi-NPU fine-tuning tasks are similar to launching pre-training. Refer to [Multi-Node, Multi-NPU pre-training commands](https://www.mindspore.cn/mindformers/docs/en/master/guide/pre_training.html#multi-node-training).

First, modify the configuration file, adjusting settings based on the number of nodes:

```yaml
parallel_config:
  data_parallel: ...
  model_parallel: ...
  pipeline_stage: ...
  context_parallel: ...
```

Modify the command as follows:

1. Add the startup script parameter `--config configs/qwen3/finetune_qwen3.yaml` to load pre-trained weights.
2. Set `--run_mode finetune` in the startup script, where run_mode indicates the running mode: train (training), finetune (fine-tuning), or predict (inference).

After task completion, a checkpoint folder will be generated in the mindformers/output directory, and the model files will be saved in this folder.

## LoRA Fine-Tuning with MindSpore Transformers

MindSpore Transformers supports configuration-driven LoRA fine-tuning, eliminating the need for code adaptations for each model. By modifying the model configuration in the full-parameter fine-tuning YAML file and adding the `pet_config` parameter-efficient fine-tuning configuration, LoRA fine-tuning tasks can be performed. Below is an example of the model configuration section in a YAML file for LoRA fine-tuning of the Qwen3 model, with detailed explanations of the `pet_config` parameters.

### Introduction to LoRA Principles

LoRA significantly reduces the number of parameters by decomposing the original model’s weight matrix into two low-rank matrices. For example, suppose a weight matrix W has dimensions $m \times n$. With LoRA, it is decomposed into two low-rank matrices A and B, where A has dimensions $m \times r$ and B has dimensions $r \times n$ ($r$ is much smaller than $m$ and $n$). During fine-tuning, only these two low-rank matrices are updated, leaving the rest of the original model unchanged.

This approach not only drastically reduces the computational cost of fine-tuning but also preserves the model’s original performance, making it particularly suitable for model optimization in environments with limited data or computational resources. For detailed principles, refer to the paper [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).

### Modifying the Configuration File

Based on the full-parameter fine-tuning configuration file, add LoRA-related parameters to the model configuration and rename it to `finetune_qwen3_8b_lora.yaml`. Below is an example configuration snippet showing how to add LoRA fine-tuning parameters for the Qwen3-8B model:

```yaml
# model config
model:
  model_config:
    ...
    # Add `pet_config` under the `model_config` level.
    pet_config:
      pet_type: lora
      lora_rank: 8
      lora_alpha: 16
      lora_dropout: 0.1
      lora_a_init: 'normal'
      lora_b_init: 'zeros'
      target_modules: '.*word_embeddings|.*linear_qkv|.*linear_proj|.*linear_fc1|.*linear_fc2'
      freeze_include: ['*']
      freeze_exclude: ['*lora*']
```

### Detailed Explanation of pet_config Parameters

In the `model_config`, `pet_config` is the core configuration section for LoRA fine-tuning, used to specify LoRA-related parameters. The parameters are explained as follows:

- **pet_type:** Specifies the type of Parameter-Efficient Tuning (PET) as LoRA. This means LoRA modules will be inserted into key layers of the model to reduce the number of parameters required for fine-tuning.
- **lora_rank:** Defines the rank of the low-rank matrices. A smaller rank results in fewer parameters to update, reducing computational resource usage. Setting it to 8 is a common balance point, significantly reducing the parameter count while maintaining model performance.
- **lora_alpha:** Controls the scaling factor for weight updates in the LoRA module. This value determines the magnitude and impact of weight updates during fine-tuning. Setting it to 16 indicates a moderate scaling factor, helping to stabilize the training process.
- **lora_dropout:** Sets the dropout probability in the LoRA module. Dropout is a regularization technique used to reduce the risk of overfitting. A value of 0.1 means there is a 10% chance of randomly “disabling” certain neural connections during training, which is particularly important when data is limited.
- **lora_a_init:** Specifies the initialization method for the LoRA A matrix. Common choices include 'normal' and 'zeros'.
- **lora_b_init:** Specifies the initialization method for the LoRA B matrix. Common choices include 'normal' and 'zeros'.
- **target_modules:** Apply LoRA to modules, with the above configuration applying LoRA to the weight matrices of word_embeddings, attention, and mlp.

### LoRA Fine-Tuning Example for Qwen3-8B

The dataset used for LoRA fine-tuning can be prepared as described in the [Dataset Preparation](#dataset-preparation) section of the full-parameter fine-tuning process.

For the Qwen3-8B model, the following msrun startup command can be executed for 8-NPU distributed fine-tuning:

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config /path/to/finetune_qwen3_8b_lora.yaml \
 --use_parallel True \
 --run_mode finetune" 8
```
