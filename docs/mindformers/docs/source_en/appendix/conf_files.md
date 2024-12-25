# Configuration File Descriptions

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/appendix/conf_files.md)

## Overview

Different parameters usually need to be configured during the training and inference process of a model. MindFormers supports the use of `YAML` files to centrally manage and adjust the configurable items, which makes the configuration of the model more structured and improves its maintainability at the same time.

## Description of the YAML File Contents

The `YAML` file provided by MindFormers contains configuration items for different functions, which are described below according to their contents.

### Basic Configuration

The basic configuration is mainly used to specify MindSpore random seeds and related settings for loading weights.

| Parameters           | Descriptions                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Types |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| seed            | Set the global seed. For details, refer to [mindspore.set_seed](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_seed.html)                                                                                                                                                                                                                                                                                                                 | int   |
| run_mode        | Set the running mode of the model: `train`, `finetune`, `eval` or `predict`.                                                                                                                                                                                                                                                                                                                                                                                          | str   |
| output_dir      | Set the path where log, checkpoint, strategy, etc. files are saved.                                                                                                                                                                                                                                                                                                                                                                                                   | str   |
| load_checkpoint | File or folder paths for loading weights. Currently there are 3 application scenarios<br/>1. Support for passing in full weight file paths<br/>2. Support for passing in offline sliced weight folder paths<br/>3. Support for passing in folder paths containing lora weights and base weights<br/>Refer to [Weight Conversion Function](https://www.mindspore.cn/mindformers/docs/en/dev/function/weight_conversion.html) for the ways of obtaining various weights. | str   |
| auto_trans_ckpt | Enable online weight automatic conversion. Refer to [Weight Conversion Function](https://www.mindspore.cn/mindformers/docs/en/dev/function/weight_conversion.html)                                                                                                                                                                                                                                                                                                    | bool  |
| resume_training | Enable resumable training after breakpoint. For details, refer to [Resumable Training After Breakpoint](https://www.mindspore.cn/mindformers/docs/en/dev/function/resume_training.html#resumable-training)                                                                                                                                                                                                                                                            | bool  |
| load_ckpt_format| The format of loading checkpoint, either `ckpt` or `safetensors`                                                                                                                                                                                                                                                                                                                                                                                                      | str   |
| remove_redundancy  | Whether the checkpoint has removed redundancy while loading checkpoint. The default value is `False`                                                                                                                                                                                                                                                                                                                                                                  | bool  |

### Context Configuration

Context configuration is mainly used to specify the [mindspore.set_context](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_context.html) in the related parameters.

| Parameters              | Descriptions                                                                                                                                                                                                                      | Types   |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| context.mode                | Set the backend execution mode, `0` means GRAPH_MODE. MindFormers currently only supports running in GRAPH_MODE mode.                                                                                          | int      |
| context.device_target       | Set the backend execution device. MindFormers is only supported on `Ascend` devices.                                                                                                              | str      |
| context.device_id           | Set the execution device ID. The value must be within the range of available devices, and the default value is `0`.                                                                                                                      | int      |
| context.enable_graph_kernel | Enable graph fusion to optimize network execution performance, defaults to `False`. See [graph fusion](https://www.mindspore.cn/docs/en/master/model_train/optimize/graph_fusion_engine.html) for details.                    | bool     |
| context.max_call_depth      | Set the maximum depth of a function call. The value must be a positive integer, and the default value is `1000`.                                                                                                                    | int      |
| context.max_device_memory   | Set the maximum memory available to the device in the format “xxGB”, and the default value is `1024GB`.                                                                                                                 | str      |
| context.mempool_block_size | Set the size of the memory pool block for devices. The format is "xxGB". Default value is `"1GB"` | str |
| context.save_graphs         | Save the compilation graph during execution.<br/>1. `False` or `0` indicates that the intermediate compilation map is not saved.<br/>2. `1` means outputting some of the intermediate files generated during the compilation of the diagram.<br/>3. `True` or `2` indicates the generation of more backend-process-related IR files. <br/>4. `3` indicates the generation of visualized computational diagrams and more detailed front-end IR diagrams. | bool/int |
| context.save_graphs_path | Path for saving the compilation diagram. | str |

### Model Configuration

Since the configuration will vary from model to model, only the generic configuration of models in MindFormers is described here.

| Parameters              | Descriptions                                                                                                                                                                                                                      | Types   |
|--------------------------------------------|--------------------------------------------------------------------------------------------------|------|
| model.arch.type                            | Set the model class to instantiate the model according to the model class when constructing the model                                                                       | str  |
| model.model_config.type                    | Set the model configuration class, the model configuration class needs to match the model class to be used, i.e. the model configuration class should contain all the parameters used by the model class                                                     | str  |
| model.model_config.num_layers              | Set the number of model layers, usually the number of layers in the model Decoder Layer.                                                                     | int  |
| model.model_config.seq_length              | Set the model sequence length, this parameter indicates the maximum sequence length supported by the model                                                                       | int  |
| model.model_config.hidden_size             | Set the dimension of the model hidden state                                                                                      | int  |
| model.model_config.vocab_size              | Set the model word list size                                                                                         | int  |
| model.model_config.top_k                   | Sample from the `top_k` tokens with the highest probability during inference                                                                     | int  |
| model.model_config.top_p                   | Sample from tokens that have the highest probability and whose probability accumulation does not exceed `top_p` during inference                                                              | int  |
| model.model_config.use_past                | Turn on model incremental inference, when turned on you can use Paged Attention to improve inference performance, must be set to `False` during model training                                          | bool |
| model.model_config.max_decode_length       | Set the maximum length of the generated text, including the input length                                                                               | int  |
| model.model_config.max_length              | The descriptions are same as `max_decode_length`. When set together with `max_decode_length`, `max_length` takes effect.                                    | int  |
| model.model_config.max_new_tokens          | Set the maximum length of the generated new text, excluding the input length, when set together with `max_length`, `max_new_tokens` takes effect.                                       | int  |
| model.model_config.min_length              | Set the minimum length of the generated text, including the input length                                                                               | int  |
| model.model_config.min_new_tokens          | Set the minimum length of the new text to be generated, excluding the input length; when set together with `min_length`, `min_new_tokens` takes effect.                                       | int  |
| model.model_config.repetition_penalty      | Set the penalty factor for generating duplicate text, `repetition_penalty` is not less than 1. When it equals to 1, duplicate outputs will not be penalized.                                            | int  |
| model.model_config.block_size              | Set the size of the block in Paged Attention, only works if `use_past=True`.                                                   | int  |
| model.model_config.num_blocks              | Set the total number of blocks in Paged Attention, effective only if `use_past=True`. `batch_size×seq_length<=block_size×num_blocks` should be satisfied. | int  |
| model.model_config.return_dict_in_generate | Set to return the inference results of the `generate` interface as a dictionary, defaults to `False`.                                                            | bool |
| model.model_config.output_scores           | Set to include score before the input softmax for each forward generation when returning the result as a dictionary, defaults to `False`                                                  | bool |
| model.model_config.output_logits           | Set to include the logits output by the model at each forward generation when returning results as a dictionary, defaults to `False`.                                                     | bool |

### MoE Configuration

In addition to the basic configuration of the model above, the MoE model needs to be configured separately with some superparameters of the moe module, and since the parameters used will vary from model to model, only the generic configuration will be explained:

| Parameters                                         | Descriptions                                                                                               | Types   |
|--------------------------------------------|--------------------------------------------------------------------------------------------------|------|
| moe_config.expert_num                    | Set the number of routing experts                                                    | int  |
| moe_config.shared_expert_num                    | Set the number of sharing experts                                                     | int  |
| moe_config.moe_intermediate_size                    | Set the size of the intermediate dimension of the expert layer                                                     | int  |
| moe_config.capacity_factor              | Set the expert capacity factor                                                                     | int  |
| moe_config.num_experts_chosen             | Set the number of experts to select per token                                                                                      | int  |
| moe_config.enable_sdrop              | Set whether to enable token drop policy `sdrop`, since MindFormers's MoE is a static shape implementation so it can't retain all tokens                                                                       | bool  |
| moe_config.aux_loss_factor              | Set the weights of the equilibrium loss                                                                       | list[float]  |
| moe_config.first_k_dense_replace              | Set the enable block of the moe layer, generally set to 1 to indicate that moe is not enabled in the first block                                                                       | int  |
| moe_config.balance_via_topk_bias              | Set whether to enable `aux_loss_free` load balancing algorithm                                                                                         | bool  |
| moe_config.topk_bias_update_rate                   | Set `aux_loss_free` load balancing algorithm `bias` update step size                                                                     | float  |
| moe_config.comp_comm_parallel                   | Set whether to enable computational communication parallelism for ffn. Default value: False                                                              | bool  |
| moe_config.comp_comm_parallel_degree                   | Set ffn to compute the number of communication splits. The higher the number, the more overlap there is, but it will consume more memory. This parameter is only valid when comp_com_parallel is enabled                               | int  |
| moe_config.moe_shared_expert_overlap                   | Set whether to enable computational communication parallelism for shared experts and routing experts. Default value: False                                                              | bool  |

### Model Training Configuration

When starting model training, in addition to model-related parameters, you also need to set the parameters of trainer, runner_config, learning rate, and optimizer and other modules required for training, MindFormers provides the following configuration items.

| Parameters              | Descriptions                                                                                                                                                                                                                      | Types |
|---------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| trainer.type                                | Set the trainer class, usually different models for different application scenarios will set different trainer classes                                                                                                                                | str   |
| trainer.model_name                          | Set the model name in the format '{name}_xxb', indicating a certain specification of the model                                                                                                                                    | str   |
| runner_config.epochs                        | Set the number of rounds for model training                                                                                                                                                           | int   |
| runner_config.batch_size                    | Set the sample size of the batch data, which overrides the `batch_size` in the dataset configuration.                                                                                                                               | int   |
| runner_config.sink_mode                     | Enable data sink mode, see [Sink Mode](https://www.mindspore.cn/docs/en/master/model_train/train_process/optimize/sink_mode.html) for details                                                     | bool  |
| runner_config.sink_size                     | Set the number of iterations to be sent down from Host to Device per iteration, effective only when `sink_mode=True`                                                                                                                        | int   |
| runner_config.gradient_accumulation_steps   | Set the number of gradient accumulation steps, the default value is 1, which means that gradient accumulation is not enabled.                                                                                                                                            | int   |
| runner_wrapper.type                         | Set the wrapper class, generally set 'MFTrainOneStepCell'                                                                                                                              | str   |
| runner_wrapper.scale_sense.type             | Set the gradient scaling class, generally just set 'DynamicLossScaleUpdateCell'                                                                                                                          | str   |
| runner_wrapper.scale_sense.use_clip_grad    | Turn on gradient clipping. Turning on to avoid cases where the inverse gradient is too large and training fails to converge                                                                                                                                       | bool  |
| runner_wrapper.scale_sense.loss_scale_value | Set the loss dynamic scale factor, the model loss can change dynamically according to the configuration of this parameter                                                                                                                                    | int   |
| lr_schedule.type                            | Set the lr_schedule class, lr_schedule is mainly used to adjust the learning rate in model training                                                                                                                           | str   |
| lr_schedule.learning_rate                   | Set the initialized learning rate size                                                                                                                                                          | float |
| lr_scale                                    | Whether to enable learning rate scaling                                                                                                                                                           | bool  |
| lr_scale_factor                             | Set the learning rate scaling factor                                                                                                                                                           | int   |
| layer_scale                                 | Whether to turn on layer attenuation                                                                                                                                                             | bool  |
| layer_decay                                 | Set the layer attenuation factor                                                                                                                                                            | float |
| optimizer.type                              | Set the optimizer class, the optimizer is mainly used to calculate the gradient for model training                                                                                                                                             | str   |
| optimizer.weight_decay                      | Set the optimizer weight decay factor                                                                                                                                                         | float |
| train_dataset.batch_size                    | The description is same as that of `runner_config.batch_size`                                                                                                                                         | int   |
| train_dataset.input_columns                 | Set the input data columns for the training dataset                                                                                                                                                       | list  |
| train_dataset.output_columns                | Set the output data columns for the training dataset                                                                                                                                                       | list  |
| train_dataset.column_order                  | Set the order of the output data columns of the training dataset                                                                                                                                                     | list  |
| train_dataset.num_parallel_workers          | Set the number of processes that read the training dataset                                                                                                                                                       | int   |
| train_dataset.python_multiprocessing        | Enabling Python multi-process mode to improve data processing performance                                                                                                                                               | bool  |
| train_dataset.drop_remainder                | Whether to discard the last batch of data if it contains fewer samples than batch_size.                                                                                                                             | bool  |
| train_dataset.repeat                        | Set the number of dataset duplicates                                                                                                                                                         | int   |
| train_dataset.numa_enable                   | Set the default state of NUMA to data read startup state                                                                                                                                                | bool  |
| train_dataset.prefetch_size                 | Set the amount of pre-read data                                                                                                                                                            | int   |
| train_dataset.data_loader.type              | Set the data loading class                                                                                                                                                             | str   |
| train_dataset.data_loader.dataset_dir       | Set the path for loading data                                                                                                                                                           | str   |
| train_dataset.data_loader.shuffle           | Whether to randomly sort the data when reading the dataset                                                                                                                                                  | bool  |
| train_dataset.transforms                    | Set options related to data enhancement                                                                                                                                                          | -     |
| train_dataset_task.type                     | Set up the dataset class, which is used to encapsulate the data loading class and other related configurations                                                                                                                                   | str   |
| train_dataset_task.dataset_config           | Typically set as a reference to `train_dataset`, containing all configuration entries for `train_dataset`.                                                                                                                     | -     |
| auto_tune                                   | Enable auto-tuning of data processing parameters, see [set_enable_autotune](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.config.set_enable_autotune.html) for details            | bool  |
| filepath_prefix                             | Set the save path for parameter configurations after data optimization                                                                                                                                                   | str   |
| autotune_per_step                           | Set the configuration tuning step interval for automatic data acceleration, for details see [set_autotune_interval](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.config.set_autotune_interval.html) | int   |

### Parallel Configuration

In order to improve the performance of the model, it is usually necessary to configure the parallelism strategy for the model in large-scale cluster usage scenarios. For details, please refer to [Distributed Parallelism](https://www.mindspore.cn/mindformers/docs/en/dev/function/distributed_parallel.html), the parallel configuration in MindFormers is as follows.

| Parameters              | Descriptions                                                                                                                                                                                                                      | Types   |
|-----------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| use_parallel                                                    | Enable parallel mode                                                                                                                                                                                           | bool |
| parallel_config.data_parallel                                   | Set the number of data parallel                                                                                                                                                                                          | int  |
| parallel_config.model_parallel                                  | Set the number of model parallel                                                                                                                                                                                          | int  |
| parallel_config.context_parallel                                | Set the number of sequence parallel                                                                                                                                                                                         | int  |
| parallel_config.pipeline_stage                                  | Set the number of pipeline parallel                                                                                                                                                                                         | int  |
| parallel_config.micro_batch_num                                 | Set the pipeline parallel microbatch size, which should satisfy `parallel_config.micro_batch_num` >= `parallel_config.pipeline_stage` when `parallel_config.pipeline_stage` is greater than 1                                                                     | int  |
| parallel_config.seq_split_num                                   | Set the sequence split number in sequence pipeline parallel, which should be a divisor of sequence length                                                                    | int  |
| parallel_config.gradient_aggregation_group                      | Set the size of the gradient communication operator fusion group                                                                                                                                                                                   | int  |
| parallel_config.context_parallel_algo                      | Set the long sequence parallel scheme, optionally `colossalai_cp`, `ulysses_cp` and `hybird_cp`, effective only if the number of `context_parallel` slices is greater than 1    | str  |
| parallel_config.ulysses_degree_in_cp                      | Setting the Ulysses sequence parallel dimension, configured in parallel with the `hybird_cp` long sequence parallel scheme, requires ensuring that `context_parallel` is divisible by this parameter and greater than 1, and that `ulysses_degree_in_cp` is divisible by the number of attention heads.  | int  |
| micro_batch_interleave_num                                      | Set the number of multicopy parallel, enable multicopy parallelism if it is greater than 1. Usually enabled when using model parallel, mainly used to optimize the communication loss generated by model parallel, and not recommended to be enabled when only using streaming parallel. For details, please refer to [MicroBatchInterleaved](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.MicroBatchInterleaved.html) | int  |
| parallel.parallel_mode                                          | Set parallel mode, `0` means data parallel mode, `1` means semi-automatic parallel mode, `2` means automatic parallel mode, `3` means mixed parallel mode, usually set to semi-automatic parallel mode.                                                                                                                          | int  |
| parallel.gradients_mean                                         | Whether to execute the averaging operator after the gradient AllReduce. Typically set to `False` in semi-automatic parallel mode and `True` in data parallel mode                                                                                                                                        | bool |
| parallel.enable_alltoall                                        | Enables generation of the AllToAll communication operator during communication. Typically set to `True` only in MOE scenarios, default value is `False`                                                                                                                                             | bool |
| parallel.full_batch                                             | Set the dataset to load the full batch in parallel mode, set to `True` in auto-parallel mode and semi-auto-parallel mode, and `False` in data-parallel mode                                                                                                                                 | bool |
| parallel.search_mode                                            | Set fully-automatic parallel strategy search mode, options are `recursive_programming`, `dynamic_programming` and `sharding_propagation`, only works in fully-automatic parallel mode, experimental interface                                                                                         | str  |
| parallel.strategy_ckpt_save_file                                | Set the save path for the parallel slicing strategy file                                                                                                                                                                                  | str  |
| parallel.strategy_ckpt_config.only_trainable_params             | Whether to save (or load) information about the slicing strategy for trainable parameters only, default is True, set this parameter to `False` when there are frozen parameters in the network but need to be sliced                                                                                                                                   | bool |
| parallel.enable_parallel_optimizer                              | Turn on optimizer parallel.<br/> 1. slice model weight parameters by number of devices in data parallel mode <br/>2. slice model weight parameters by `parallel_config.data_parallel` in semi-automatic parallel mode                                                                                          | bool |
| parallel.parallel_optimizer_config.gradient_accumulation_shard  | Set whether the cumulative gradient variable is sliced on the data-parallel dimension, only effective if `enable_parallel_optimizer=True`                                                                                                                                    | bool |
| parallel.parallel_optimizer_config.parallel_optimizer_threshold | Set the threshold for the optimizer weight parameter cut, effective only if `enable_parallel_optimizer=True`.                                                                                                                                             | int  |
| parallel.parallel_optimizer_config.optimizer_weight_shard_size  | Set the size of the optimizer weight parameter to slice the communication domain, requiring the value to be integrable by `parallel_config.data_parallel`, effective only if `enable_parallel_optimizer=True`.                                                                                                  | int  |

> Configure the parallel strategy to satisfy device_num = data_parallel × model_parallel × context_parallel × pipeline_stage

### Model Optimization Configuration

MindFormers provides recomputation-related configurations to reduce the memory footprint of the model during training, see [Recomputation](https://www.mindspore.cn/mindformers/docs/en/dev/perf_optimize/perf_optimize.html#recomputation) for details.

| Parameters              | Descriptions                                                                                                                                                                                                                      | Types   |
|----------------------------------------------------|-------------------------------|-----------|
| recompute_config.recompute                         | Enable recompute                         | bool      |
| recompute_config.select_recompute                  | Turn on recomputation to recompute only for the operators in the attention layer | bool/list |
| recompute_config.parallel_optimizer_comm_recompute | Whether to recompute AllGather communication introduced in parallel by the optimizer  | bool/list |
| recompute_config.mp_comm_recompute                 | Whether to recompute communications introduced by model parallel           | bool      |
| recompute_config.recompute_slice_activation        | Whether to output slices for Cells kept in memory            | bool      |

### Callbacks Configuration

MindFormers provides encapsulated Callbacks function class, mainly to achieve to return to the model training state and output in the model training process, save the model weight file and other operations. Currently the following Callbacks function class is supported.

1. MFLossMonitor

   This callback function class is mainly used to print information such as training progress, model Loss, and learning rate during the training process and has several configurable items as follows:

   | Parameters                     | Descriptions                                                                                                                                                                                                                                                                                                | Types |
   |--------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
   | learning_rate                  | Set the initial learning rate in `MFLossMonitor`. The default value is `None`                                                                                                                                                                                                                               | float |
   | per_print_times                | Set the interval for printing log information in `MFLossMonitor`. The default value is `1`, that is, the log information is printed every step                                                                                                                                                              | int   |
   | micro_batch_num                | Set the size of the micro batch data in each step in the training, which is used to calculate the actual loss value. If this parameter is not set, the value of this parameter is the same as that of `parallel_config.micro_batch_num` in [Parallel Configuration](#parallel-configuration)                | int   |
   | micro_batch_interleave_num     | Set the size of the interleave micro batch data in each step of the training. This parameter is used to calculate the actual loss value. If this parameter is not set, the value of this parameter is the same as that of `micro_batch_interleave_num` in [Parallel Configuration](#parallel-configuration) | int   |
   | origin_epochs                  | Set the initial number of training epochs in `MFLossMonitor`. If this parameter is not set, the value of this parameter is the same as that of `runner_config.epochs` in [Model Training Configuration](#model-training-configuration)                                                                      | int   |
   | dataset_size                   | Set initial size of the dataset in `MFLossMonitor`. If this parameter is not set, the size of the initialized dataset is the same as the size of the actual dataset used for training                                                                                                                       | int   |
   | initial_epoch                  | Set start epoch number of training in `MFLossMonitor`. The default value is `0`                                                                                                                                                                                                                             | int   |
   | initial_step                   | Set start step number of training in `MFLossMonitor`. The default value is `0`                                                                                                                                                                                                                              | int   |
   | global_batch_size              | Set the number of global batch data samples in `MFLossMonitor`. If this parameter is not set, the system automatically calculates the number of global batch data samples based on the dataset size and parallel strategy                                                                                   | int   |
   | gradient_accumulation_steps    | Set the number of gradient accumulation steps in `MFLossMonitor`. If this parameter is not set, the value of this parameter is the same as that of `gradient_accumulation_steps` in [Model Training Configuration](#model-training-configuration)                                                           | int   |
   | check_for_nan_in_loss_and_grad | Whether to enable overflow detection in `MFLossMonitor`. After overflow detection is enabled, the training exits if overflow occurs during model training. The default value is `False`                                                                                                                     | bool  |

2. SummaryMonitor

   This callback function class is mainly used to collect Summary data, see [mindspore.SummaryCollector](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.SummaryCollector.html) for details.

3. CheckpointMonitor

   This callback function class is mainly used to save the model weights file during the model training process and has several configurable items as follows:

   | Parameters                   | Descriptions                                                                                                                                                                                                                                                                                                                                                                                                           | Types |
   |------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
   | prefix                       | Set the prefix for saving file names                                                                                                                                                                                                                                                                                                                                                                                   | str   |
   | directory                    | Set the directory for saving file names                                                                                                                                                                                                                                                                                                                                                                                | str   |
   | save_checkpoint_seconds      | Set the number of seconds between saving model weights                                                                                                                                                                                                                                                                                                                                                                 | int   |
   | save_checkpoint_steps        | Set the number of interval steps for saving model weights                                                                                                                                                                                                                                                                                                                                                              | int   |
   | keep_checkpoint_max          | Set the maximum number of model weight files to be saved, if there are more model weight files in the save path, they will be deleted starting from the earliest file created to ensure that the total number of files does not exceed `keep_checkpoint_max`.                                                                                                                                                          | int   |
   | keep_checkpoint_per_n_minutes | Set the number of minutes between saving model weights                                                                                                                                                                                                                                                                                                                                                                 | int   |
   | integrated_save              | Turn on aggregation to save the weights file.<br/>1. When set to True, it means that the weights of all devices are aggregated when the weight file is saved, i.e., the weights of all devices are the same.<br/>2. False means that all devices save their own weights<br/>When using semi-automatic parallel mode, it is usually necessary to set it to False to avoid memory problems when saving the weights file. | bool  |
   | save_network_params          | Set to save only model weights, default value is `False`.                                                                                                                                                                                                                                                                                                                                                              | bool  |
   | save_trainable_params        | Set the additional saving of trainable parameter weights, i.e. the parameter weights of the model when partially fine-tuned, default to `False`.                                                                                                                                                                                                                                                                       | bool  |
   | async_save                   | Set an asynchronous execution to save the model weights file                                                                                                                                                                                                                                                                                                                                                           | bool  |
   | remove_redundancy             | Whether to remove the redundancy for the checkpoint, default value is `False`.                                                                                                                                                                                                                                                                                                                                         | bool  |
   | checkpoint_format           | The format of the checkpoint while saving the checkpoint, default value is `ckpt`. Either `ckpt` or `safetensors`                                                                                                                                                                                                                                                                                                      | str   |

Multiple Callbacks function classes can be configured at the same time under the `callbacks` field. The following is an example of `callbacks` configuration.

```yaml
callbacks:
  - type: MFLossMonitor
  - type: CheckpointMonitor
    prefix: "name_xxb"
    save_checkpoint_steps: 1000
    integrated_save: False
    async_save: False
```

### Processor Configuration

Processor is mainly used to preprocess the inference data of the input model. Since the Processor configuration items are not fixed, only the generic configuration items of Processor in MindFormers are explained here.

| Parameters              | Descriptions                                                                                                                                                                                                                      | Types   |
|--------------------------------|--------------------------------------|-----|
| processor.type                 | Set the data processing class                              | str |
| processor.return_tensors       | Set the type of tensor returned by the data processing class, typically use 'ms'              | str |
| processor.image_processor.type | Set the image data processing class                            | str |
| processor.tokenizer.type       | Set the text tokenizer class                       | str |
| processor.tokenizer.vocab_file | Set the path of the file to be read by the text tokenizer, which needs to correspond to the tokenizer class | str |

### Model Evaluation Configuration

MindFormers provides model evaluation function, and also supports model evaluation while training. The following is the configuration related to model evaluation.

| Parameters              | Descriptions                                                                                                                                                                                                                      | Types   |
|---------------------|-------------------------------------------------------------|------|
| eval_dataset        | Used in the same way as `train_dataset`                                      | -    |
| eval_dataset_task   | Used in the same way as `eval_dataset_task`                                  | -    |
| metric.type         | Used in the same way as `callbacks`                                          | -    |
| do_eval             | Enable evaluation while training                                                | bool |
| eval_step_interval  | Set evaluation step interval, default value is 100. The value less than 0 means disable evaluation according to step interval.                  | int  |
| eval_epoch_interval | Set the epoch interval for evaluation, the default value is -1. The value less than 0 means disable the function of evaluating according to epoch interval, it is not recommended to use this configuration in data sinking mode. | int  |
| metric.type         | Set the type of evaluation                                                     | str  |

### Profile Configuration

MindFormers provides Profile as the main tool for model performance tuning, please refer to [Performance Tuning Guide](https://www.mindspore.cn/mindformers/docs/en/dev/perf_optimize/perf_optimize.html) for more details. The following is the Profile related configuration.

| Parameters              | Descriptions                                                                                                                                                                                                                      | Types   |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------------|------|
| profile               | Enable the performance capture tool, see [mindspore.Profiler](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Profiler.html) for details | bool |
| profile_start_step    | Set the number of steps to start collecting performance data                                                                                                            | int  |
| profile_stop_step     | Set the number of steps to stop collecting performance data                                                                                                            | int  |
| profile_communication | Set whether communication performance data is collected in multi-device training, this parameter is invalid when using single card training. Default: `False`                                                                               | bool |
| profile_memory        | Set whether to collect Tensor memory data                                                                                                            | bool |
| init_start_profile    | Set whether to turn on collecting performance data when the Profiler is initialized; this parameter does not take effect when `profile_start_step` is set. This parameter needs to be set to `True` when `profile_memory` is turned on.                                  | bool |

### TensorBoard Configuration

The TensorBoard configuration is primarily used to configure parameters related to TensorBoard during training, allowing for real-time monitoring and visualization of training metrics. Below is a description of the common TensorBoard configuration options in MindFormers:

| Parameters                                | Descriptions                                                                                                                         | Types  |
|-------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|--------|
| tensorboard.tensorboard_dir               | Set the TensorBoard log directory, specifying the path where TensorBoard saves the log files.                                        | str    |
| tensorboard.tensorboard_queue_size        | Sets the maximum queue size for TensorBoard, controlling the speed of log writing.                                                   | int    |
| tensorboard.log_loss_scale_to_tensorboard | Whether to log loss scale information to TensorBoard.                                                                                | bool   |
| tensorboard.log_timers_to_tensorboard     | Whether to log timer information to TensorBoard, including the duration and throughput of the current training step (or iteration).  | bool   |

The actual path for saving event files (events.*) is `tensorboard.tensorboard_dir/rank_id`. You can start the TensorBoard Web visualization service with the following command:

```bash
tensorboard --logdir=/path/events.* --host=0.0.0.0 --port=6006

# parameter description
logdir: The path to the directory where TensorBoard event files are saved.
host:   Default is 127.0.0.1, meaning it only allows access from the local machine. Set it to 0.0.0.0 to allow access from external devices. Be mindful of security.
port:   The port that the service listens on. Default is 6006.
```
