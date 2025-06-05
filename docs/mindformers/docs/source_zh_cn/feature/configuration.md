# 配置文件说明

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/configuration.md)

## 概述

在模型的训练和推理过程中通常需要配置不同的参数，MindSpore Transformers支持使用`YAML`文件集中管理和调整可配置项，使模型的配置更加结构化，同时提高了其可维护性。

## YAML文件内容说明

MindSpore Transformers提供的`YAML`文件中包含对于不同功能的配置项，下面按照配置项的内容对其进行说明。

### 基础配置

基础配置主要用于指定MindSpore随机种子以及加载权重的相关设置。

| 参数                   | 说明                                                                                                                                                                                                                      | 类型             |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| seed                 | 设置全局种子，详情可参考[mindspore.set_seed](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_seed.html)。                                                                                              | int            |
| run_mode             | 设置模型的运行模式，可选`train`、`finetune`、`eval`或`predict`。                                                                                                                                                                        | str            |
| output_dir           | 设置保存log、checkpoint、strategy等文件的路径。                                                                                                                                                                                      | str            |
| load_checkpoint      | 加载权重的文件或文件夹路径，目前有3个应用场景：<br/>1. 支持传入完整权重文件路径。<br/>2. 支持传入离线切分后的权重文件夹路径。<br/>3. 支持传入包含lora权重和base权重的文件夹路径。<br/>各种权重的获取途径可参考[权重转换功能](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/ckpt.html)。 | str            |
| auto_trans_ckpt      | 是否开启分布式权重自动切分与合并功能，详情可参考[分布式权重切分与合并](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/ckpt.html)。                                                                                                            | bool           |
| resume_training      | 是否开启断点续训功能，详情可参考[断点续训功能](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/resume_training.html#%E6%96%AD%E7%82%B9%E7%BB%AD%E8%AE%AD)。                                                                        | bool           |
| load_ckpt_format     | 加载的模型权重的格式，可选`ckpt`、`safetensors`。                                                                                                                                                                                      | str            |
| remove_redundancy    | 加载的模型权重是否去除了冗余。默认值为`False`。                                                                                                                                                                                             | bool           |
| train_precision_sync | 训练确定性计算开关。默认值为`None` 。                                                                                                                                                                                                  | Optional[bool] |
| infer_precision_sync | 推理确定性计算开关。默认值为`None`。                                                                                                                                                                                                   | Optional[bool] |

### Context配置

Context配置主要用于指定[mindspore.set_context](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_context.html)中的相关参数。

| 参数                        | 说明                                                         | 类型     |
| --------------------------- | ------------------------------------------------------------ | -------- |
| context.mode                | 设置后端执行模式，`0`表示GRAPH_MODE，MindSpore Transformers目前仅支持在GRAPH_MODE模式下运行。 | int      |
| context.device_target       | 设置后端执行设备，MindSpore Transformers仅支持在`Ascend`设备上运行。      | str      |
| context.device_id           | 设置执行设备ID，其值必须在可用设备范围内，默认值为`0`。        | int      |
| context.enable_graph_kernel | 是否开启图算融合去优化网络执行性能，默认值为`False`。 | bool     |
| context.max_call_depth      | 设置函数调用的最大深度，其值必须为正整数，默认值为`1000`。     | int      |
| context.max_device_memory   | 设置设备可用的最大内存，格式为"xxGB"，默认值为`1024GB`。       | str      |
| context.mempool_block_size  | 设置内存块大小，格式为"xxGB"，默认值为`1GB`。                  | str      |
| context.save_graphs         | 在执行过程中保存编译图。<br/>1. `False`或`0`表示不保存中间编译图。<br/>2. `1`表示运行时会输出图编译过程中生成的一些中间文件。<br/>3. `True`或`2`表示生成更多后端流程相关的IR文件。<br/>4. `3`表示生成可视化计算图和更多详细的前端IR图。 | bool/int |
| context.save_graphs_path    | 保存编译图的路径。                                             | str      |
| context.affinity_cpu_list   | 可选配置项，用于实现用户自定义绑核策略。不配置时，默认绑核。`None`表示关闭绑核。默认值为`{}`，如想使能自定义绑核策略，需传入`dict`，详情可参考[mindspore.runtime.set_cpu_affinity](https://www.mindspore.cn/docs/zh-CN/master/api_python/runtime/mindspore.runtime.set_cpu_affinity.html#mindspore.runtime.set_cpu_affinity)。 | dict/str      |

### 模型配置

由于不同的模型配置会有差异，这里仅对MindSpore Transformers中模型的通用配置进行说明。

| 参数                                         | 说明                                                                                               | 类型   |
|--------------------------------------------|--------------------------------------------------------------------------------------------------|------|
| model.arch.type                            | 设置模型类，构建模型时可以根据模型类对模型进行实例化。                                                                       | str  |
| model.model_config.type                    | 设置模型配置类，模型配置类需要与模型类匹配使用，即模型配置类中应包含所有模型类使用的参数。                                                     | str  |
| model.model_config.num_layers              | 设置模型层数，通常指模型Decoder Layer的层数。                                                                     | int  |
| model.model_config.seq_length              | 设置模型序列长度，该参数表示模型所支持的最大序列长度。                                                                       | int  |
| model.model_config.hidden_size             | 设置模型隐藏状态的维数。                                                                                      | int  |
| model.model_config.vocab_size              | 设置模型词表大小。                                                                                         | int  |
| model.model_config.top_k                   | 设置推理时从概率最大的`top_k`个tokens中采样。                                                                     | int  |
| model.model_config.top_p                   | 设置推理时从概率最大且概率累计不超过`top_p`的tokens中采样。                                                              | int  |
| model.model_config.use_past                | 是否开启模型增量推理，开启后可使用Paged Attention提升推理性能，在模型训练时必须设置为`False`。                                        | bool |
| model.model_config.max_decode_length       | 设置生成文本的最大长度，包括输入长度。                                                                               | int  |
| model.model_config.max_length              | 同`max_decode_length`，与`max_decode_length`同时设置时，`max_length`生效。                                    | int  |
| model.model_config.max_new_tokens          | 设置生成新文本的最大长度，不包括输入长度，与`max_length`同时设置时，`max_new_tokens`生效。                                       | int  |
| model.model_config.min_length              | 设置生成文本的最小长度，包括输入长度。                                                                               | int  |
| model.model_config.min_new_tokens          | 设置生成新文本的最小长度，不包括输入长度，与`min_length`同时设置时，`min_new_tokens`生效。                                       | int  |
| model.model_config.repetition_penalty      | 设置生成重复文本的惩罚系数，`repetition_penalty`不小于1，等于1时不对重复输出进行惩罚。                                            | int  |
| model.model_config.block_size              | 设置Paged Attention中block的大小，仅`use_past=True`时生效。                                                  | int  |
| model.model_config.num_blocks              | 设置Paged Attention中block的总数，仅`use_past=True`时生效，应满足`batch_size×seq_length<=block_size×num_blocks`。 | int  |
| model.model_config.return_dict_in_generate | 是否以字典形式返回`generate`接口的推理结果，默认为`False`。                                                            | bool |
| model.model_config.output_scores           | 是否以字典形式返回结果时，包含每次前向生成时的输入softmax前的分数，默认为`False`。                                                  | bool |
| model.model_config.output_logits           | 是否以字典形式返回结果时，包含每次前向生成时模型输出的logits，默认为`False`。                                                     | bool |
| model.model_config.layers_per_stage        | 设置开启pipeline stage时，每个stage分配到的transformer层数，默认为`None`，表示每个stage平均分配。设置的值为一个长度为pipeline stage数量的整数列表，第i位表示第i个stage被分配到的transformer层数。                                                | list |

### MoE配置

除了上述模型的基本配置，MoE模型需要单独配置一些moe模块的超参，由于不同模型使用的参数会有不同，仅对通用配置进行说明：

| 参数                                         | 说明                                                                                               | 类型   |
|--------------------------------------------|--------------------------------------------------------------------------------------------------|------|
| moe_config.expert_num                    | 设置路由专家数量。                                                     | int  |
| moe_config.shared_expert_num                    | 设置共享专家数量。                                                     | int  |
| moe_config.moe_intermediate_size                    | 设置专家层中间维度大小。                                                     | int  |
| moe_config.capacity_factor              | 设置专家容量因子。                                                                     | int  |
| moe_config.num_experts_chosen             | 设置每个token选择专家数目。                                                                                      | int  |
| moe_config.enable_sdrop              | 设置是否使能token丢弃策略`sdrop`，由于MindSpore Transformers的MoE是静态shape实现所以不能保留所有token。                                                                       | bool  |
| moe_config.aux_loss_factor              | 设置均衡性loss的权重。                                                                       | list[float]  |
| moe_config.first_k_dense_replace              | 设置moe层的使能block，一般设置为1，表示第一个block不使能moe。                                                                       | int  |
| moe_config.balance_via_topk_bias              | 设置是否使能`aux_loss_free`负载均衡算法。                                                                                         | bool  |
| moe_config.topk_bias_update_rate                   | 设置`aux_loss_free`负载均衡算法`bias`更新步长。                                                                     | float  |
| moe_config.comp_comm_parallel                   | 设置是否开启ffn的计算通信并行。默认值：False。                                                             | bool  |
| moe_config.comp_comm_parallel_degree                   | 设置ffn计算通信的分割数。数字越大，重叠越多，但会消耗更多内存。此参数仅在comp_com_parallel启用时有效。                                                              | int  |
| moe_config.moe_shared_expert_overlap                   | 设置是否开启共享专家和路由专家的计算通信并行。默认值：False。                                                              | bool  |

### 模型训练配置

启动模型训练时，除了模型相关参数，还需要设置trainer、runner_config、学习率以及优化器等训练所需模块的参数，MindSpore Transformers提供了如下配置项。

| 参数                                          | 说明                                                                                                                                                                  | 类型    |
|---------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| trainer.type                                | 设置trainer类，通常不同应用场景的模型会设置不同的trainer类。                                                                                                                                | str   |
| trainer.model_name                          | 设置模型名称，格式为'{name}_xxb'，表示模型的某一规格。                                                                                                                                    | str   |
| runner_config.epochs                        | 设置模型训练的轮数。                                                                                                                                                           | int   |
| runner_config.batch_size                    | 设置批处理数据的样本数，该配置会覆盖数据集配置中的`batch_size`。                                                                                                                               | int   |
| runner_config.sink_mode                     | 是否开启数据下沉模式。                                                                                                                                                          | bool  |
| runner_config.sink_size                     | 设置每次从Host下发到Device的迭代数量，仅`sink_mode=True`时生效，此参数将在后续版本中废弃。                                                                                                           | int   |
| runner_config.gradient_accumulation_steps   | 设置梯度累积步数，默认值为1，表示不开启梯度累积。                                                                                                                                            | int   |
| runner_wrapper.type                         | 设置wrapper类，一般设置'MFTrainOneStepCell'即可。                                                                                                                               | str   |
| runner_wrapper.scale_sense.type             | 设置梯度缩放类，一般设置'DynamicLossScaleUpdateCell'即可。                                                                                                                          | str   |
| runner_wrapper.scale_sense.use_clip_grad    | 是否开启梯度剪裁，开启可避免反向梯度过大导致训练无法收敛的情况。                                                                                                                                     | bool  |
| runner_wrapper.scale_sense.loss_scale_value | 设置loss动态尺度系数，模型loss可以根据该参数配置动态变化。                                                                                                                                    | int   |
| lr_schedule.type                            | 设置lr_schedule类，lr_schedule主要用于调整模型训练中的学习率。                                                                                                                           | str   |
| lr_schedule.learning_rate                   | 设置初始化学习率大小。                                                                                                                                                          | float |
| lr_scale                                    | 是否开启学习率缩放。                                                                                                                                                           | bool  |
| lr_scale_factor                             | 设置学习率缩放系数。                                                                                                                                                           | int   |
| layer_scale                                 | 是否开启层衰减。                                                                                                                                                             | bool  |
| layer_decay                                 | 设置层衰减系数。                                                                                                                                                             | float |
| optimizer.type                              | 设置优化器类，优化器主要用于计算模型训练的梯度。                                                                                                                                             | str   |
| optimizer.weight_decay                      | 设置优化器权重衰减系数。                                                                                                                                                         | float |
| train_dataset.batch_size                    | 同`runner_config.batch_size`。                                                                                                                                         | int   |
| train_dataset.input_columns                 | 设置训练数据集输入的数据列。                                                                                                                                                       | list  |
| train_dataset.output_columns                | 设置训练数据集输出的数据列。                                                                                                                                                       | list  |
| train_dataset.construct_args_key            | 设置模型`construct`输入的数据集部分`keys`, 按照字典序传入模型，当模型的传参顺序和数据集输入的顺序不一致时使用该功能。                    | list |
| train_dataset.column_order                  | 设置训练数据集输出数据列的顺序。                                                                                                                                                     | list  |
| train_dataset.num_parallel_workers          | 设置读取训练数据集的进程数。                                                                                                                                                       | int   |
| train_dataset.python_multiprocessing        | 是否开启Python多进程模式提升数据处理性能。                                                                                                                                             | bool  |
| train_dataset.drop_remainder                | 是否在最后一个批处理数据包含样本数小于batch_size时，丢弃该批处理数据。                                                                                                                             | bool  |
| train_dataset.repeat                        | 设置数据集重复数据次数。                                                                                                                                                         | int   |
| train_dataset.numa_enable                   | 设置NUMA的默认状态为数据读取启动状态。                                                                                                                                                | bool  |
| train_dataset.prefetch_size                 | 设置预读取数据量。                                                                                                                                                            | int   |
| train_dataset.data_loader.type              | 设置数据加载类。                                                                                                                                                            | str   |
| train_dataset.data_loader.dataset_dir       | 设置加载数据的路径。                                                                                                                                                           | str   |
| train_dataset.data_loader.shuffle           | 是否在读取数据集时对数据进行随机排序。                                                                                                                                                  | bool  |
| train_dataset.transforms                    | 设置数据增强相关选项。                                                                                                                                                          | -     |
| train_dataset_task.type                     | 设置dataset类，该类用于对数据加载类以及其他相关配置进行封装。                                                                                                                                   | str   |
| train_dataset_task.dataset_config           | 通常设置为`train_dataset`的引用，包含`train_dataset`的所有配置项。                                                                                                                     | -     |
| auto_tune                                   | 是否开启数据处理参数自动调优，详情可参考[set_enable_autotune](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.config.set_enable_autotune.html)。          | bool  |
| filepath_prefix                             | 设置数据优化后的参数配置的保存路径。                                                                                                                                                   | str   |
| autotune_per_step                           | 设置自动数据加速的配置调整step间隔，详情可参考[set_autotune_interval](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.config.set_autotune_interval.html)。 | int   |

### 并行配置

为了提升模型的性能，在大规模集群的使用场景中通常需要为模型配置并行策略，详情可参考[分布式并行](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/parallel_training.html)，MindSpore Transformers中的并行配置如下。

| 参数                                                              | 说明                                                                                                                                                                                               | 类型   |
|-----------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| use_parallel                                                    | 是否开启并行模式。                                                                                                                                                                                         | bool |
| parallel_config.data_parallel                                   | 设置数据并行数。                                                                                                                                                                                          | int  |
| parallel_config.model_parallel                                  | 设置模型并行数。                                                                                                                                                                                          | int  |
| parallel_config.context_parallel                                | 设置序列并行数。                                                                                                                                                                                          | int  |
| parallel_config.pipeline_stage                                  | 设置流水线并行数。                                                                                                                                                                                         | int  |
| parallel_config.micro_batch_num                                 | 设置流水线并行的微批次大小，在`parallel_config.pipeline_stage`大于1时，应满足`parallel_config.micro_batch_num` >= `parallel_config.pipeline_stage`。                                                                     | int  |
| parallel_config.seq_split_num                                   | 在序列流水线并行中设置序列分割数，该数应为序列长度的除数。                                                                     | int  |
| parallel_config.gradient_aggregation_group                      | 设置梯度通信算子融合组的大小。                                                                                                                                                                                   | int  |
| parallel_config.context_parallel_algo                      | 设置长序列并行方案，可选`colossalai_cp`、`ulysses_cp`和`hybrid_cp`，仅在`context_parallel`切分数大于1时生效。                                                                                                                                                                                   | str  |
| parallel_config.ulysses_degree_in_cp                      | 设置Ulysses序列并行维度，与`hybrid_cp`长序列并行方案同步配置，需要确保`context_parallel`可以被该参数整除且大于1，同时确保`ulysses_degree_in_cp`可以被attention head数整除。                                                                                                                                                                      | int  |
| micro_batch_interleave_num                                      | 设置多副本并行数，大于1时开启多副本并行。通常在使用模型并行时开启，主要用于优化模型并行产生的通信损耗，仅使用流水并行时不建议开启。详情可参考[MicroBatchInterleaved](https://www.mindspore.cn/docs/zh-CN/master/api_python/parallel/mindspore.parallel.nn.MicroBatchInterleaved.html)。 | int  |
| parallel.parallel_mode                                          | 设置并行模式，`0`表示数据并行模式, `1`表示半自动并行模式, `2`表示自动并行模式, `3`表示混合并行模式，一般设置为半自动并行模式。                                                                                                                          | int  |
| parallel.gradients_mean                                         | 是否在梯度AllReduce后执行平均算子。通常半自动并行模式下设为`False`，数据并行模式下设为`True`。                                                                                                                                        | bool |
| parallel.enable_alltoall                                        | 是否在通信期间生成AllToAll通信算子。通常仅在MOE场景下设为`True`，默认值为`False`。                                                                                                                                             | bool |
| parallel.full_batch                                             | 是否在并行模式下从数据集中读取加载完整的批数据，设置为`True`表示所有rank都读取完整的批数据，设置为`False`表示每个rank仅加载对应的批数据，设置为`False`时必须设置对应的`dataset_strategy`。                                                                                                                                                                  | bool |
| parallel.dataset_strategy                                       | 仅支持`List of List`类型且仅在`full_batch=False`时生效，列表中子列表的个数需要等于`train_dataset.input_columns`的长度，并且列表中的每个子列表需要和数据集返回的数据的shape保持一致。一般在数据的第1维进行数据并行切分，所以子列表的第1位数配置与`data_parallel`相同，其他位配置为`1`。具体原理可以参考[数据集切分](https://www.mindspore.cn/tutorials/zh-CN/master/parallel/dataset_slice.html)。 | list |
| parallel.search_mode                                            | 设置全自动并行策略搜索模式，可选`recursive_programming`、`dynamic_programming`和`sharding_propagation`，仅在全自动并行模式下生效，实验性接口。                                                                                         | str  |
| parallel.strategy_ckpt_save_file                                | 设置并行切分策略文件的保存路径。                                                                                                                                                                                  | str  |
| parallel.strategy_ckpt_config.only_trainable_params             | 是否仅保存（或加载）可训练参数的切分策略信息，默认为`True`，当网络中存在冻结的参数但又需要切分时将该参数设为`False`。                                                                                                                                 | bool |
| parallel.enable_parallel_optimizer                              | 是否开启优化器并行。<br/>1. 在数据并行模式下将模型权重参数按device数进行切分。<br/>2. 在半自动并行模式下将模型权重参数按`parallel_config.data_parallel`进行切分。                                                                                        | bool |
| parallel.parallel_optimizer_config.gradient_accumulation_shard  | 设置累计的梯度变量是否在数据并行的维度上进行切分，仅`enable_parallel_optimizer=True`时生效。                                                                                                                                    | bool |
| parallel.parallel_optimizer_config.parallel_optimizer_threshold | 设置优化器权重参数切分的阈值，仅`enable_parallel_optimizer=True`时生效。                                                                                                                                             | int  |
| parallel.parallel_optimizer_config.optimizer_weight_shard_size  | 设置优化器权重参数切分通信域的大小，要求该值可以整除`parallel_config.data_parallel`，仅`enable_parallel_optimizer=True`时生效。                                                                                                  | int  |
| parallel.pipeline_config.pipeline_interleave  | 使能interleave，使用Seq-Pipe流水线并行时需设置为`true`。                                                                                                  | bool  |
| parallel.pipeline_config.pipeline_scheduler  | Seq-Pipe的流水线调度策略，目前只支持`"seqpipe"`。                                                                                                  | str  |

> 配置并行策略时应满足device_num = data_parallel × model_parallel × context_parallel × pipeline_stage。

### 模型优化配置

1. MindSpore Transformers提供重计算相关配置，以降低模型在训练时的内存占用，详情可参考[重计算](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/advanced_development/performance_optimization.html#重计算)。

   | 参数                                                 | 说明                            | 类型              |
   |----------------------------------------------------|-------------------------------|-----------------|
   | recompute_config.recompute                         | 是否开启重计算。                       | bool/list/tuple |
   | recompute_config.select_recompute                  | 开启选择重计算，只针对attention层的算子进行重计算。 | bool/list       |
   | recompute_config.parallel_optimizer_comm_recompute | 是否对由优化器并行引入的AllGather通信进行重计算。  | bool/list       |
   | recompute_config.mp_comm_recompute                 | 是否对由模型并行引入的通信进行重计算。            | bool            |
   | recompute_config.recompute_slice_activation        | 是否对保留在内存中的Cell输出切片。            | bool            |
   | recompute_config.select_recompute_exclude          | 关闭指定算子的重计算，只对Primitive算子有效。    | bool/list       |
   | recompute_config.select_comm_recompute_exclude     | 关闭指定算子的通讯重计算，只对Primitive算子有效。  | bool/list       |

2. MindSpore Transformers提供细粒度激活值SWAP相关配置，以降低模型在训练时的内存占用，详情可参考[细粒度激活值SWAP](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/memory_optimization.html#%E7%BB%86%E7%B2%92%E5%BA%A6%E6%BF%80%E6%B4%BB%E5%80%BCswap)。

   | 参数 | 说明 | 类型 |
   |------|-----|-----|
   | swap_config.swap | 是否开启激活值SWAP。 | bool |
   | swap_config.default_prefetch | 设置激活值卸载至host时的内存释放时机与开始取回device的时机，仅在开启激活值SWAP且未设置layer_swap与op_swap时生效。 | int |
   | swap_config.layer_swap | 选择特定的层使能激活值SWAP。 | list |
   | swap_config.op_swap | 选择特定层中的特定算子使能激活值SWAP。 | list |

### Callbacks配置

MindSpore Transformers提供封装后的Callbacks函数类，主要实现在模型训练过程中返回模型的训练状态并输出、保存模型权重文件等一些操作，目前支持以下几个Callbacks函数类。

1. MFLossMonitor

   该回调函数类主要用于在训练过程中对训练进度、模型Loss、学习率等信息进行打印，有如下几个可配置项：

   | 参数                             | 说明                                                                                      | 类型    |
   |--------------------------------|-----------------------------------------------------------------------------------------|-------|
   | learning_rate                  | 设置`MFLossMonitor`中初始化学习率，默认值为`None`。                                                     | float |
   | per_print_times                | 设置`MFLossMonitor`中日志信息打印频率，默认值为`1`，即每一步打印一次日志信息。                                         | int   |
   | micro_batch_num                | 设置训练中每一步的批数据大小，用于计算实际的loss值，若不配置该参数，则与[并行配置](#并行配置)中`parallel_config.micro_batch_num`一致。 | int   |
   | micro_batch_interleave_num     | 设置训练中每一步的多副本批数据大小，用于计算实际的loss值，若不配置该参数，则与[并行配置](#并行配置)中`micro_batch_interleave_num`一致。   | int   |
   | origin_epochs                  | 设置`MFLossMonitor`中训练的轮数，若不配置该参数，则与[模型训练配置](#模型训练配置)中`runner_config.epochs`一致。            | int   |
   | dataset_size                   | 设置`MFLossMonitor`中初始化数据集大小，若不配置该参数，则与实际训练使用的数据集大小一致。                                     | int   |
   | initial_epoch                  | 设置`MFLossMonitor`中训练起始轮数，默认值为`0`。                                                        | int   |
   | initial_step                   | 设置`MFLossMonitor`中训练起始步数，默认值为`0`。                                                        | int   |
   | global_batch_size              | 设置`MFLossMonitor`中全局批数据样本数，若不配置该参数，则会根据数据集大小以及并行策略自动计算。                                  | int   |
   | gradient_accumulation_steps    | 设置`MFLossMonitor`中梯度累计步数，若不配置该参数，则与[模型训练配置](#模型训练配置)中`gradient_accumulation_steps`一致。    | int   |
   | check_for_nan_in_loss_and_grad | 设置是否在`MFLossMonitor`中开启溢出检测，开启后在模型训练过程中出现溢出则退出训练，默认值为`False`。                            | bool  |

2. SummaryMonitor

   该回调函数类主要用于收集Summary数据，详情可参考[mindspore.SummaryCollector](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.SummaryCollector.html)。

3. CheckpointMonitor

   该回调函数类主要用于在模型训练过程中保存模型权重文件，有如下几个可配置项：

   | 参数                            | 说明                                                                                                                                             | 类型   |
   |-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|------|
   | prefix                        | 设置保存文件名称的前缀。                                                                                                                                    | str  |
   | directory                     | 设置保存文件名称的目录。                                                                                                                                    | str  |
   | save_checkpoint_seconds       | 设置保存模型权重的间隔秒数。                                                                                                                                  | int  |
   | save_checkpoint_steps         | 设置保存模型权重的间隔steps数。                                                                                                                              | int  |
   | keep_checkpoint_max           | 设置保存模型权重文件的最大数量，如果保存路径内存在超出数量的模型权重文件，会从创建时间最早的文件开始删除，以保证文件总数不超过`keep_checkpoint_max`。                                                           | int  |
   | keep_checkpoint_per_n_minutes | 设置保存模型权重的间隔分钟数。                                                                                                                                 | int  |
   | integrated_save               | 开启聚合保存权重文件。<br/>1. 设为True时表示在保存权重文件时聚合所有device的权重，即所有device权重一致。<br/>2. 设为False时表示所有device各自保存自己的权重。<br/>使用半自动并行模式时通常需要设置为False，以避免保存权重文件时出现内存问题。 | bool |
   | save_network_params           | 是否仅保存模型权重，默认值为`False`。                                                                                                                          | bool |
   | save_trainable_params         | 是否额外保存可训练的参数权重，即部分微调时模型的参数权重，默认为`False`。                                                                                                       | bool |
   | async_save                    | 是否异步执行保存模型权重文件。                                                                                                                                 | bool |
   | remove_redundancy             | 是否去除模型权重的冗余，默认值为`False`。                                                                                                                        | bool |                                                                                                                                            |      |
   | checkpoint_format             | 保存的模型权重的格式，默认值为`ckpt`。可选`ckpt`，`safetensors`。                                                                                                   | str  |

在`callbacks`字段下可同时配置多个Callbacks函数类，以下是`callbacks`配置示例。

```yaml
callbacks:
  - type: MFLossMonitor
  - type: CheckpointMonitor
    prefix: "name_xxb"
    save_checkpoint_steps: 1000
    integrated_save: False
    async_save: False
```

### Processor配置

Processor主要用于对输入模型的推理数据进行预处理，由于Processor配置项不固定，这里仅对MindSpore Transformers中的Processor通用配置项进行说明。

| 参数                             | 说明                                   | 类型  |
|--------------------------------|--------------------------------------|-----|
| processor.type                 | 设置数据处理类。                              | str |
| processor.return_tensors       | 设置数据处理类返回的张量类型，一般使用'ms'。              | str |
| processor.image_processor.type | 设置图像数据处理类。                            | str |
| processor.tokenizer.type       | 设置文本tokenizer类。                       | str |
| processor.tokenizer.vocab_file | 设置文本tokenizer读取文件路径，需要与tokenizer类相对应。 | str |

### 模型评估配置

MindSpore Transformers提供模型评估功能，同时支持模型边训练边评估功能，以下是模型评估相关配置。

| 参数                  | 说明                                                         | 类型   |
|---------------------|------------------------------------------------------------|------|
| eval_dataset        | 使用方式与`train_dataset`相同。                                     | -    |
| eval_dataset_task   | 使用方式与`eval_dataset_task`相同。                                | -    |
| metric.type         | 使用方式与`callbacks`相同。                                        | -    |
| do_eval             | 是否开启边训练边评估功能 。                                              | bool |
| eval_step_interval  | 设置评估的step间隔，默认值为100，设置小于0表示关闭根据step间隔评估功能。                  | int  |
| eval_epoch_interval | 设置评估的epoch间隔，默认值为-1，设置小于0表示关闭根据epoch间隔评估功能，不建议在数据下沉模式使用该配置。 | int  |
| metric.type         | 设置评估的类型。                                                    | str  |

### Profile配置

MindSpore Transformers提供Profile作为模型性能调优的主要工具，详情可参考[性能调优指南](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/advanced_development/performance_optimization.html)。以下是Profile相关配置。

| 参数                    | 说明                                                                                                                                        | 类型   |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------|------|
| profile               | 是否开启性能采集工具，默认值为`False`，详情可参考[mindspore.Profiler](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Profiler.html)。 | bool |
| profile_start_step    | 设置开始采集性能数据的step数，默认值为`1`。                                                                                                                  | int  |
| profile_stop_step     | 设置停止采集性能数据的step数，默认值为`10`。                                                                                                                 | int  |
| profile_communication | 设置是否在多设备训练中收集通信性能数据，使用单卡训练时，该参数无效，默认值为`False`。                                                                                             | bool |
| profile_memory        | 设置是否收集Tensor内存数据，默认值为`True`。                                                                                                               | bool |
| profile_rank_ids      | 设置开启性能采集的rank ids，默认值为`None`，表示所有rank id均开启性能采集。                                                                                           | list |
| profile_pipeline      | 设置是否按流水线并行每个stage的其中一张卡开启性能采集，默认值为`False`。                                                                                                 | bool |
| profile_output        | 设置保存性能采集生成文件的文件夹路径。                                                                                                                        | str  |
| profile_level         | 设置采集数据的级别，可选值为(0, 1, 2)，默认值为`1`。                                                                                                           | int  |
| with_stack            | 设置是否收集Python侧的调用栈数据，默认值为`False`。                                                                                                           | bool |
| data_simplification   | 设置是否开启数据精简，开启后将在导出性能采集数据后删除FRAMEWORK目录以及其他多余数据，默认为`False`。                                                                                 | int  |
| init_start_profile    | 设置是否在Profiler初始化时开启采集性能数据，设置`profile_start_step`时该参数不生效，开启`profile_memory`时需要将该参数设为`True`。                                                 | bool |
| mstx                  | 设置是否收集mstx时间戳记录，包括训练step、HCCL通信算子等，默认值为`False`。                                                                                                            | bool |

### 指标监控配置

指标监控配置主要用于配置训练过程中各指标的记录方式，详情可参考[训练指标监控](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/monitor.html)。以下是MindSpore Transformers中通用的指标监控配置项说明：

| 参数名称                                    | 说明                                                                                                                         | 类型            |
|-----------------------------------------|----------------------------------------------------------------------------------------------------------------------------|---------------|
| monitor_config.monitor_on               | 设置是否开启监控。默认为`False`，此时以下所有参数不生效。                                                                                            | bool          |
| monitor_config.dump_path                | 设置训练过程中`local_norm`、`device_local_norm`、`local_loss`指标文件的保存路径。未设置或设置为`null`时取默认值'./dump'。                                   | str           |
| monitor_config.target                   | 设置指标`优化器状态`和`local_norm`所监控的的目标参数的名称（片段），可为正则表达式。未设置或设置为`null`时取默认值['.*']，即指定所有参数。                                          | list[str]     |
| monitor_config.invert                   | 设置反选`monitor_config.target`所指定的参数，默认为`False`。                                                                               | bool          |
| monitor_config.step_interval            | 设置记录指标的频率。默认为1，即每个step记录一次。                                                                                                 | int           |
| monitor_config.local_loss_format        | 设置指标`local_loss`的记录形式，可选值为字符串'tensorboard'和'log'（分别表示写入 Tensorboard 和写入日志），或由两者组成的列表，或`null`。未设置时默认为`null`，表示不监控该指标。        | str或list[str] |
| monitor_config.device_local_loss_format | 设置指标`device_local_loss`的记录形式，可选值为字符串'tensorboard'和'log'（分别表示写入 Tensorboard 和写入日志），或由两者组成的列表，或`null`。未设置时默认为`null`，表示不监控该指标。 | str或list[str] |
| monitor_config.local_norm_format        | 设置指标`local_norm`的记录形式，可选值为字符串'tensorboard'和'log'（分别表示写入 Tensorboard 和写入日志），或由两者组成的列表，或`null`。未设置时默认为`null`，表示不监控该指标。        | str或list[str] |
| monitor_config.device_local_norm_format | 设置指标`device_local_norm`的记录形式，可选值为字符串'tensorboard'和'log'（分别表示写入 Tensorboard 和写入日志），或由两者组成的列表，或`null`。未设置时默认为`null`，表示不监控该指标。 | str或list[str] |
| monitor_config.optimizer_state_format   | 设置指标`优化器状态`的记录形式，可选值为字符串'tensorboard'和'log'（分别表示写入 Tensorboard 和写入日志），或由两者组成的列表，或`null`。未设置时默认为`null`，表示不监控该指标。             | str或list[str] |
| monitor_config.weight_state_format      | 设置指标`权重L2-norm`的记录形式，可选值为字符串'tensorboard'和'log'（分别表示写入 Tensorboard 和写入日志），或由两者组成的列表，或`null`。未设置时默认为`null`，表示不监控该指标。         | str或list[str] |
| monitor_config.throughput_baseline      | 设置指标`吞吐量线性度`的基线值，需要为正数。未设置时默认为`null`，表示不监控该指标。                                                                              | int或float     |
| monitor_config.print_struct             | 设置是否打印模型的全部可训练参数名。若为`True`，则会在第一个step开始时打印所有可训练参数的名称，并在step结束后退出训练。默认为`False`。                                              | bool          |

### TensorBoard配置

TensorBoard配置主要用于配置训练过程中与TensorBoard相关的参数，便于在训练过程中实时查看和监控训练信息，详情可参考[训练指标监控](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/monitor.html)。以下是MindSpore Transformers中通用的TensorBoard配置项说明：

| 参数名称                                      | 说明                                                      | 类型   |
|-------------------------------------------|---------------------------------------------------------|------|
| tensorboard.tensorboard_dir               | 设置 TensorBoard 事件文件的保存路径。                                | str  |
| tensorboard.tensorboard_queue_size        | 设置采集队列的最大缓存值，超过该值便会写入事件文件，默认值为10。                        | int  |
| tensorboard.log_loss_scale_to_tensorboard | 设置是否将 loss scale 信息记录到事件文件，默认为`False`。                   | bool |
| tensorboard.log_timers_to_tensorboard     | 设置是否将计时器信息记录到事件文件，计时器信息包含当前训练步骤（或迭代）的时长以及吞吐量，默认为`False`。 | bool |