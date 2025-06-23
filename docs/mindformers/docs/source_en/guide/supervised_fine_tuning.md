# Supervised Fine-Tuning (SFT)

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/guide/supervised_fine_tuning.md)

## Overview

SFT (Supervised Fine-Tuning) adopts the concept of supervised learning, referring to the process of adjusting some or all parameters of a pre-trained model to better adapt it to specific tasks or datasets.

MindSpore Transformers supports two SFT fine-tuning methods: full-parameter fine-tuning and LoRA fine-tuning. Full-parameter fine-tuning involves updating all parameters during training, suitable for large-scale data refinement, offering optimal task adaptability but requiring significant computational resources. LoRA fine-tuning updates only a subset of parameters, consuming less memory and training faster than full-parameter fine-tuning, though its performance may be inferior in certain tasks.

## Basic Process of SFT Fine-Tuning

Combining practical operations, SFT fine-tuning can be broken down into the following steps:

### 1. Weight Preparation

Before fine-tuning, the weight files of the pre-trained model need to be prepared. MindSpore Transformers supports loading [safetensors weights](https://www.mindspore.cn/mindformers/docs/en/dev/feature/safetensors.html), enabling direct loading of model weights downloaded from the Hugging Face model hub.

### 2. Dataset Preparation

MindSpore Transformers currently supports datasets in [Hugging Face format](https://www.mindspore.cn/mindformers/docs/en/dev/feature/dataset.html#huggingface-datasets) and [MindRecord format](https://www.mindspore.cn/mindformers/docs/en/dev/feature/dataset.html#mindrecord-dataset) for the fine-tuning phase. Users can prepare data according to task requirements.

### 3. Configuration File Preparation

Fine-tuning tasks are uniformly controlled through [configuration files](https://www.mindspore.cn/mindformers/docs/en/dev/feature/configuration.html), allowing users to flexibly adjust [model training hyperparameters](https://www.mindspore.cn/mindformers/docs/en/dev/feature/training_hyperparameters.html). Additionally, fine-tuning performance can be optimized using [distributed parallel training](https://www.mindspore.cn/mindformers/docs/en/dev/feature/parallel_training.html), [memory optimization features](https://www.mindspore.cn/mindformers/docs/en/dev/feature/memory_optimization.html), and [other training features](https://www.mindspore.cn/mindformers/docs/en/dev/feature/other_training_features.html).

### 4. Launching the Training Task

MindSpore Transformers provides a [one-click startup script](https://www.mindspore.cn/mindformers/docs/en/dev/feature/start_tasks.html) to initiate fine-tuning tasks. During training, [logs](https://www.mindspore.cn/mindformers/docs/en/dev/feature/logging.html) and [visualization tools](https://www.mindspore.cn/mindformers/docs/en/dev/feature/monitor.html) can be used to monitor the training process.

### 5. Model Saving

Checkpoints are saved during training, or model weights are saved to a specified path upon completion. Currently, weights can be saved in [Safetensors format](https://www.mindspore.cn/mindformers/docs/en/dev/feature/safetensors.html) or [Ckpt format](https://www.mindspore.cn/mindformers/docs/en/dev/feature/ckpt.html), which can be used for resumed training or further fine-tuning.

### 6. Fault Recovery

To handle exceptions such as training interruptions, MindSpore Transformers offers [high-availability features](https://www.mindspore.cn/mindformers/docs/en/dev/feature/high_availability.html) like last-state saving and automatic recovery, as well as [checkpoint-based resumed training](https://www.mindspore.cn/mindformers/docs/en/dev/feature/resume_training.html), enhancing training stability.

## Full-Parameter Fine-Tuning with MindSpore Transformers

### Selecting a Pre-Trained Model

MindSpore Transformers currently supports mainstream large-scale models in the industry. This guide uses the Qwen2.5-7B model as an example.

### Downloading Model Weights

MindSpore Transformers supports loading Hugging Face model weights, enabling direct loading of weights downloaded from the Hugging Face model hub. For details, refer to [MindSpore Transformers-Safetensors Weights](https://www.mindspore.cn/mindformers/docs/en/dev/feature/safetensors.html).

| Model Name  | Hugging Face Weight Download Link                     |
| :---------- | :---------------------------------------------------: |
| Qwen2.5-7B  | [Link](https://huggingface.co/Qwen/Qwen2.5-7B)        |

### Dataset Preparation

MindSpore Transformers supports online loading of Hugging Face datasets. For details, refer to [MindSpore Transformers-Dataset-Hugging Face Dataset](https://www.mindspore.cn/mindformers/docs/en/dev/feature/dataset.html#huggingface-datasets).

This guide uses [llm-wizard/alpaca-gpt4-data](https://huggingface.co/datasets/llm-wizard/alpaca-gpt4-data) as the fine-tuning dataset.

| Dataset Name                | Applicable Phase | Download Link                                                      |
| :-------------------------- | :--------------: | :----------------------------------------------------------------: |
| llm-wizard/alpaca-gpt4-data | Fine-Tuning      | [Link](https://huggingface.co/datasets/llm-wizard/alpaca-gpt4-data) |

### Executing the Fine-Tuning Task

#### Single-NPU Training

First, prepare the configuration file. This guide provides a fine-tuning configuration file for the Qwen2.5-7B model, `finetune_qwen2_5_7b_8k_1p.yaml`, available for download from the [Gitee repository](https://gitee.com/mindspore/docs/tree/master/docs/mindformers/docs/source_zh_cn/example/supervised_fine_tuning/finetune_qwen2_5_7b_8k_1p.yaml).

> Due to limited single-NPU memory, the `num_layers` in the configuration file is set to 4, used as an example only.

Then, modify the parameters in the configuration file based on actual conditions, mainly including:

```yaml
load_checkpoint: '/path/to/Qwen2.5-7B/'                   # Path to the pre-trained model weight folder
...
train_dataset: &train_dataset
  ...
  data_loader:
    ...
    handler:
      - type: AlpacaInstructDataHandler
        tokenizer:
          vocab_file: "/path/to/Qwen2.5-7B/vocab.json"    # Path to the vocabulary file
          merges_file: "/path/to/Qwen2.5-7B/merges.txt"   # Path to the merges file
```

Run `run_mindformer.py` to start the single-NPU fine-tuning task. The command is as follows:

```shell
python run_mindformer.py \
 --config /path/to/finetune_qwen2_5_7b_8k_1p.yaml \
 --register_path research/qwen2_5 \
 --use_parallel False \
 --run_mode finetune
```

Parameter descriptions:

```commandline
config:            Model configuration file
use_parallel:      Whether to enable parallel training
run_mode:          Running mode, train: training, finetune: fine-tuning, predict: inference
```

#### Single-Node Training

First, prepare the configuration file. This guide provides a fine-tuning configuration file for the Qwen2.5-7B model, `finetune_qwen2_5_7b_8k.yaml`, available for download from the [Gitee repository](https://gitee.com/mindspore/docs/tree/master/docs/mindformers/docs/source_zh_cn/example/supervised_fine_tuning/finetune_qwen2_5_7b_8k.yaml).

Then, modify the parameters in the configuration file based on actual conditions, mainly including:

```yaml
load_checkpoint: '/path/to/Qwen2.5-7B/'                   # Path to the pre-trained model weight folder
...
train_dataset: &train_dataset
  ...
  data_loader:
    ...
    handler:
      - type: AlpacaInstructDataHandler
        tokenizer:
          vocab_file: "/path/to/Qwen2.5-7B/vocab.json"    # Path to the vocabulary file
          merges_file: "/path/to/Qwen2.5-7B/merges.txt"   # Path to the merges file
```

Run the following msrun startup script for 8-NPU distributed training:

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/qwen2_5 \
 --config /path/to/finetune_qwen2_5_7b_8k.yaml \
 --use_parallel True \
 --run_mode finetune" 8
```

Parameter descriptions:

```commandline
config:            Model configuration file
use_parallel:      Whether to enable parallel training
run_mode:          Running mode, train: training, finetune: fine-tuning, predict: inference
```

After task completion, a checkpoint folder will be generated in the mindformers/output directory, and the model files will be saved in this folder.

#### Multi-Node Training

Multi-Node, multi-NPU fine-tuning tasks are similar to launching pre-training. Refer to [multi-node, multi-NPU pre-training commands](https://www.mindspore.cn/mindformers/docs/en/dev/guide/pre_training.html#multi-node-training).

First, modify the configuration file, adjusting settings based on the number of nodes:

```yaml
parallel_config:
  data_parallel: ...
  model_parallel: ...
  pipeline_stage: ...
  context_parallel: ...
```

Modify the command as follows:

1. Add the startup script parameter `--config /path/to/finetune_qwen2_5_7b_8k.yaml` to load pre-trained weights.
2. Set `--run_mode finetune` in the startup script, where run_mode indicates the running mode: train (training), finetune (fine-tuning), or predict (inference).

After task completion, a checkpoint folder will be generated in the mindformers/output directory, and the model files will be saved in this folder.

## LoRA Fine-Tuning with MindSpore Transformers

MindSpore Transformers supports configuration-driven LoRA fine-tuning, eliminating the need for code adaptations for each model. By modifying the model configuration in the full-parameter fine-tuning YAML file and adding the `pet_config` parameter-efficient fine-tuning configuration, LoRA fine-tuning tasks can be performed. Below is an example of the model configuration section in a YAML file for LoRA fine-tuning of the Llama2 model, with detailed explanations of the `pet_config` parameters.

### Introduction to LoRA Principles

LoRA significantly reduces the number of parameters by decomposing the original model’s weight matrix into two low-rank matrices. For example, suppose a weight matrix W has dimensions $m \times n$. With LoRA, it is decomposed into two low-rank matrices A and B, where A has dimensions $m \times r$ and B has dimensions $r \times n$ ($r$ is much smaller than $m$ and $n$). During fine-tuning, only these two low-rank matrices are updated, leaving the rest of the original model unchanged.

This approach not only drastically reduces the computational cost of fine-tuning but also preserves the model’s original performance, making it particularly suitable for model optimization in environments with limited data or computational resources. For detailed principles, refer to the paper [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).

### Modifying the Configuration File

Based on the full-parameter fine-tuning configuration file, add LoRA-related parameters to the model configuration and rename it to `fine_tune_qwen2_5_7b_8k_lora.yaml`. Below is an example configuration snippet showing how to add LoRA fine-tuning parameters for the Qwen2.5-7B model:

```yaml
# model config
model:
  model_config:
    ...
    pet_config:
      pet_type: lora
      lora_rank: 16
      lora_alpha: 16
      lora_dropout: 0.05
      target_modules: '.*wq|.*wk|.*wv|.*wo'
```

### Detailed Explanation of pet_config Parameters

In the `model_config`, `pet_config` is the core configuration section for LoRA fine-tuning, used to specify LoRA-related parameters. The parameters are explained as follows:

- **pet_type:** Specifies the type of Parameter-Efficient Tuning (PET) as LoRA. This means LoRA modules will be inserted into key layers of the model to reduce the number of parameters required for fine-tuning.
- **lora_rank:** Defines the rank of the low-rank matrices. A smaller rank results in fewer parameters to update, reducing computational resource usage. Setting it to 16 is a common balance point, significantly reducing the parameter count while maintaining model performance.
- **lora_alpha:** Controls the scaling factor for weight updates in the LoRA module. This value determines the magnitude and impact of weight updates during fine-tuning. Setting it to 16 indicates a moderate scaling factor, helping to stabilize the training process.
- **lora_dropout:** Sets the dropout probability in the LoRA module. Dropout is a regularization technique used to reduce the risk of overfitting. A value of 0.05 means there is a 5% chance of randomly “disabling” certain neural connections during training, which is particularly important when data is limited.
- **target_modules:** Specifies which weight matrices in the model LoRA will be applied to, using regular expressions. In Llama, this configuration applies LoRA to the Query (wq), Key (wk), Value (wv), and Output (wo) matrices in the self-attention mechanism. These matrices play critical roles in the Transformer architecture, and applying LoRA to them maintains model performance while reducing the parameter count.

### LoRA Fine-Tuning Example for Qwen2.5-7B

The dataset used for LoRA fine-tuning can be prepared as described in the [Dataset Preparation](#dataset-preparation) section of the full-parameter fine-tuning process.

For the Qwen2.5-7B model, the following msrun startup command can be executed for 8-NPU distributed fine-tuning:

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/qwen2_5 \
 --config /path/to/finetune_qwen2_5_7b_8k_lora.yaml \
 --use_parallel True \
 --run_mode finetune" 8
```
