# Training Configuration Template Instruction

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/advanced_development/training_template_instruction.md)

## Overview

MindSpore Transformers provides a universal configuration file template for training, which can be used in two main scenarios:

1. User-developed models can be adapted by writing training configurations based on templates.
2. For the existing models of MindSpore Transformers, when users wish to use specific specification models that are not currently configured, they can use configuration templates and combine them with HuggingFace or ModelScope model configurations to initiate training tasks.

MindSpore Transformers provides corresponding configuration templates for different training scenarios, as follows:

When pre-training DENSE model, please use [llm_pretrain_dense_template.yaml](https://gitee.com/mindspore/mindformers/blob/master/configs/general/llm_pretrain_dense_template.yaml).

When pre-training MOE model, please use [llm_pretrain_moe_template.yaml](https://gitee.com/mindspore/mindformers/blob/master/configs/general/llm_pretrain_moe_template.yaml).

When fine-tuning DENSE model training, please use [llm_finetune_dense_template.yaml](https://gitee.com/mindspore/mindformers/blob/master/configs/general/llm_finetune_dense_template.yaml).

When fine-tuning MOE model training, please use [llm_finetune_moe_template.yaml](https://gitee.com/mindspore/mindformers/blob/master/configs/general/llm_finetune_moe_template.yaml).

## Instructions for Use

### Module Description

The template mainly covers the configuration of the following nine functional modules, and detailed parameter configuration instructions can be referred to [Profile Description](https://www.mindspore.cn/mindformers/docs/en/master/feature/configuration.html).

| Module Name                             | Module Usage                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|-----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Basic Configuration                     | The basic configuration is mainly used to specify MindSpore random seeds and related settings for loading weights.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Dataset Configuration                   | Dataset configuration is mainly used for dataset-related settings during MindSpore model training. For details, please refer to the [Dataset](https://www.mindspore.cn/mindformers/docs/en/master/feature/dataset.html).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Model Configuration                     | There are differences in the configuration parameters of different models, and the parameters in the template are universal configurations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Model Optimization Configuration        | MindSpore Transformers provides configuration related to recalculation to reduce the memory usage of the model during training. For details, please refer to [Recalculation](https://www.mindspore.cn/mindformers/docs/en/master/advanced_development/performance_optimization.html#recomputation).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Model Training Configuration            | When starting model training, the configuration module for relevant parameters is mainly included in the template, which includes parameters for the required training modules such as trainer, runner_config, runner_wrapper, learning rate (lr_schedule), and optimizer.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Parallel Configuration                  | In order to improve the performance of the model, it is usually necessary to configure parallel strategies for the model in large-scale cluster usage scenarios. For details, please refer to [Distributed Parallel](https://www.mindspore.cn/mindformers/docs/en/master/feature/parallel_training.html).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Callback Function Configuration         | MindSpore Transformers provides encapsulated callbacks function classes, which mainly implement operations such as returning the training state of the model and outputting, saving the model weight file, etc. during the model training process. Currently, the following callback function classes are supported. <br> 1.MFLossMonitor <br> This callback function class is mainly used to print information such as training progress, model loss, and learning rate during the training process. <br> 2.SummaryMonitor <br> This callback function class is mainly used to collect Summary data. For details, please refer to [mindspore.SummaryCollector](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.SummaryCollector.html). <br> 3.CheckpointMonitor<br>This callback function class is mainly used to save the model weight file during the model training process. |
| Context configuration                   | Context configuration is mainly used to specify the related parameters in [mindspore.set_context](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_context.html).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Performance Analysis Tool Configuration | MindSpore Transformers provides Profile as the main tool for model performance tuning. For details, please refer to the [Performance Tuning Guide](https://www.mindspore.cn/mindformers/docs/en/master/advanced_development/performance_optimization.html).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

## Basic Configuration Modification

When using a configuration template for training, modify the following basic configurations to quickly start.

The default configuration template uses 8 cards.

### Dataset Configuration Modification

1. The pre-training scenario uses the Megatron dataset. For details, please refer to the [Megatron Dataset](https://www.mindspore.cn/mindformers/docs/en/master/feature/dataset.html#megatron-dataset).
2. Fine-tune the scenario using the HuggingFace dataset. Please refer to [HuggingFace dataset](https://www.mindspore.cn/mindformers/docs/en/master/feature/dataset.html#hugging-face-dataset) for details.

### Model Configuration Modification

1. When modifying the model configuration, you can choose to download the HuggingFace model and directly modify the pretrained-model-dir in the YAML configuration to read the model configuration (this feature does not currently support pretraining). During model training, a tokenizer and model_config will be automatically generated, and the model list is supported:

   | Model Name |
   |------------|
   | Deepseek3  |
   | Qwen3      |
   | Qwen2_5    |

2. The generated model configuration shall be based on the YAML configuration first, and if no parameters are configured, the parameters in the config.json file under the pretrained-model-dir path shall be taken as the values. If you want to modify the custom model configuration, you only need to add the relevant configuration in mode_config.
3. For general configuration details, please refer to [Model Configuration](https://www.mindspore.cn/mindformers/docs/en/master/feature/configuration.html#basic-configuration).

## Advanced Configuration Modification

Further modifications can be made in the following way to customize the training.

### Basic Configuration Modification

When conducting pre-training, the generated weight format can be modified through load_ckpt_format, which supports safetensors and ckpt. It is recommended to use safetensors. The path for generating logs, weights, and policy files during the training process can be specified through output_dir.

### Training Parameter Modification

1. Configuration modifications related to recompute_config, optimizer, and lr_schedule can affect the accuracy of model training results.
2. If insufficient memory occurs, preventing the model from starting training, we can consider enabling recomputation to reduce the model memory usage during training.
3. By modifying the learning rate configuration, the learning effect during model training can be achieved.
4. Modifying optimizer configuration can modify the gradient during model training.
5. Configurations related to parallel (model parallelism) and context can affect the performance of model training.
6. During model training, the performance can be improved by enabling use_parallel=True, and the expected performance can be achieved by debugging and configuring parallel strategies. Please refer to [Parallel Configuration](https://www.mindspore.cn/mindformers/docs/en/master/feature/configuration.html#parallel-configuration) for detailed parameter configuration.
7. For specific configurations, refer to [Model Training Configuration](https://www.mindspore.cn/mindformers/docs/en/master/feature/configuration.html#model-training-configuration).

### Callback Function Configuration Modification

1. The template provides callback functions related to saving weights: save_checkpoint_steps can modify the interval for saving weights; keep_checkpoint_max can set the maximum number of weights to be saved, effectively controlling the disk space for weight saving.
2. Please refer to [callback function configuration](https://www.mindspore.cn/mindformers/docs/en/master/feature/configuration.html#callbacks-configuration) for other callback function applications.

### Resume Training

When performing resumable training after breakpoint, it is necessary to modify the load_checkpoint to the weight directory saved in the previous training task based on the YAML configuration file used in the previous training, that is, the checkpoint directory under the directory specified by the output_dir parameter, and set the resume_training to True. For details, please refer to [Resume training](https://www.mindspore.cn/mindformers/docs/en/master/feature/resume_training.html).