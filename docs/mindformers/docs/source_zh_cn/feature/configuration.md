# 配置文件说明

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/configuration.md)

## 概述

在模型的训练和推理过程中，通常需要配置不同的参数。MindSpore Transformers支持使用`YAML`文件集中管理和调整可配置项，使模型配置更加结构化，同时提高了可维护性。

## YAML文件内容说明

MindSpore Transformers提供的`YAML`文件中包含不同功能的配置项，下面按照配置项内容对其进行说明。

### 基础配置

基础配置主要用于指定MindSpore随机种子以及加载权重的相关设置。

| 参数名称                          | 数据类型  | 是否可选 | 默认值    | 取值说明                                                                                                                                                                                                                                 |
|-------------------------------|-------|------|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| seed                          | int   | 可选   | 0      | 设置全局随机种子，用于保证实验可复现性。详情可参考 [mindspore.set_seed](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_seed.html)。                                                                                             |
| run_mode                      | str   | 必选   | 无      | 设置模型的运行模式，可选：`train`、`finetune`、`eval` 或 `predict`。                                                                                                                                                                                  |
| output_dir                    | str   | 可选   | 无      | 设置保存日志（log）、权重（checkpoint）、并行策略（strategy）等文件的输出路径。若路径不存在，会尝试自动创建。                                                                                                                                                                     |
| load_checkpoint               | str   | 可选   | 无      | 加载权重的文件或文件夹路径，支持以下三种场景：<br/>1. 完整权重文件路径；<br/>2. 离线切分后的分布式权重文件夹路径；<br/>3. 包含 LoRA 增量权重和 base 模型权重的文件夹路径。<br/>各种权重的获取方式详见 [权重转换功能](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/ckpt.html)                           |
| auto_trans_ckpt               | bool  | 可选   | False  | 是否开启分布式权重自动切分与合并功能。开启后可在单卡加载多卡切分权重，或多卡加载单卡权重。详情见 [分布式权重切分与合并](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/ckpt.html)                                                                                              |
| resume_training               | bool  | 可选   | False  | 是否开启断点续训功能。开启后将从 `load_checkpoint` 指定的路径恢复优化器状态、学习率调度器状态等，继续训练。详情见 [断点续训功能](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/resume_training.html#%E6%96%AD%E7%82%B9%E7%BB%AD%E8%AE%AD)                                |
| load_ckpt_format              | str   | 可选   | "ckpt" | 加载的模型权重的格式，可选 `"ckpt"` 和 `"safetensors"`。                                                                                                                                                                                            |
| remove_redundancy             | bool  | 可选   | False  | 加载的模型权重是否已去除冗余。详情可参考[权重去冗余保存与加载](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/safetensors.html#%E5%8E%BB%E5%86%97%E4%BD%99%E4%BF%9D%E5%AD%98%E5%8F%8A%E5%8A%A0%E8%BD%BD)                                           |
| train_precision_sync          | bool  | 可选   | None   | 训练确定性计算开关。设置为 `True`，则开启训练同步计算，可以提升计算的确定性，一般可用于确保实验的可复现性；设置为 `False`，则不开启。                                                                                                                                                           |
| infer_precision_sync          | bool  | 可选   | None   | 推理确定性计算开关。设置为 `True`，则开启推理同步计算，可以提升计算的确定性，一般可用于确保实验的可复现性；设置为 `False`，则不开启。                                                                                                                                                           |
| use_skip_data_by_global_norm  | bool  | 可选   | False  | 是否启用基于全局梯度范数的数据跳过功能。当某批次数据导致梯度爆炸时，自动跳过该批次以提升训练稳定性。详情可见 [数据跳过](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/skip_data_and_ckpt_health_monitor.html)。                                                                |
| use_checkpoint_health_monitor | bool  | 可选   | False  | 是否启用权重健康监测功能。开启后会在保存 checkpoint 时校验其完整性与可用性，防止保存损坏的权重文件。详情可见 [权重健康监测](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/skip_data_and_ckpt_health_monitor.html#%E6%9D%83%E9%87%8D%E5%81%A5%E5%BA%B7%E7%9B%91%E6%B5%8B)。 |

### Context配置

Context配置主要用于指定[mindspore.set_context](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_context.html)中的相关参数。

| 参数名称                        | 数据类型          | 是否可选 | 默认值       | 取值说明                                                                                                                                                                                                                                        |
|-----------------------------|---------------|------|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| context.mode                | int           | 必选   | 无         | 设置后端执行模式，`0` 表示 GRAPH_MODE。MindSpore Transformers 目前仅支持在 GRAPH_MODE 模式下运行。                                                                                                                                                                  |
| context.device_target       | string        | 必选   | 无         | 设置后端执行设备，MindSpore Transformers 仅支持在 `Ascend` 设备上运行。                                                                                                                                                                                        |
| context.device_id           | int           | 可选   | 0         | 设置执行设备 ID，其值必须在可用设备范围内，默认值为 `0`。                                                                                                                                                                                                            |
| context.enable_graph_kernel | bool          | 可选   | False     | 是否开启图算融合去优化网络执行性能，默认值为 `False`。                                                                                                                                                                                                             |
| context.max_call_depth      | int           | 可选   | 1000      | 设置函数调用的最大深度，其值必须为正整数，默认值为 `1000`。                                                                                                                                                                                                           |
| context.max_device_memory   | string        | 可选   | "1024GB"  | 设置设备可用的最大内存，格式为 `"xxGB"`。默认值为 `"1024GB"`。                                                                                                                                                                                                   |
| context.mempool_block_size  | string        | 可选   | "1GB"     | 设置内存块大小，格式为 `"xxGB"`，默认值为 `"1GB"`。                                                                                                                                                                                                          |
| context.save_graphs         | bool / int    | 可选   | False     | 在执行过程中保存编译图：<br/>• `False` 或 `0` ：不保存中间编译图<br/>• `1`：输出图编译过程中的部分中间文件<br/>• `True`或`2`：生成更多后端流程相关的IR文件<br/>• `3`：生成可视化计算图和更详细的前端IR图                                                                                                          |
| context.save_graphs_path    | string        | 可选   | './graph' | 保存编译图的路径。若未设置且 `save_graphs != False`，则使用默认临时路径 `'./graph'`。                                                                                                                                                                                |
| context.affinity_cpu_list   | dict / string | 可选   | None      | 可选配置项，用于实现用户自定义绑核策略。<br/>• 不配置时：默认自动绑核<br/>• `None` 或未设置：关闭绑核<br/>• 传入 `dict`：自定义CPU核心绑定策略，详情参考 [mindspore.runtime.set_cpu_affinity](https://www.mindspore.cn/docs/zh-CN/master/api_python/runtime/mindspore.runtime.set_cpu_affinity.html) |

### Legacy 模型配置

如果使用 MindSpore Transformers 拉起 legacy 模型的任务，需要在 yaml 文件中进行相关超参的配置。注意，此板块介绍的配置仅适用于 legacy 模型，不可与 mcore 模型配置进行混用，请注意[版本配套关系](https://gitee.com/mindspore/mindformers/blob/master/README_CN.md#%E6%A8%A1%E5%9E%8B%E5%88%97%E8%A1%A8)。

由于不同的模型配置会有差异，这里仅对MindSpore Transformers中模型的通用配置进行说明。

| 参数名称                                       | 数据类型      | 是否可选 | 默认值   | 取值说明                                                                                                                                                     |
|--------------------------------------------|-----------|------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| model.arch.type                            | string    | 必选   | 无     | 设置模型类，构建模型时可以根据模型类对模型进行实例化。                                                                                                                              |
| model.model_config.type                    | string    | 必选   | 无     | 设置模型配置类，模型配置类需要与模型类匹配使用，即模型配置类中应包含所有模型类使用的参数。                                                                                                            |
| model.model_config.num_layers              | int       | 必选   | 无     | 设置模型层数，通常指模型 Decoder Layer 的层数。                                                                                                                          |
| model.model_config.seq_length              | int       | 必选   | 无     | 设置模型序列长度，该参数表示模型所支持的最大序列长度。                                                                                                                              |
| model.model_config.hidden_size             | int       | 必选   | 无     | 设置模型隐藏状态的维数。                                                                                                                                             |
| model.model_config.vocab_size              | int       | 必选   | 无     | 设置模型词表大小。                                                                                                                                                |
| model.model_config.top_k                   | int       | 可选   | 无     | 设置推理时从概率最大的 `top_k` 个 tokens 中采样。                                                                                                                        |
| model.model_config.top_p                   | float     | 可选   | 无     | 设置推理时从概率最大且概率累计不超过 `top_p` 的 tokens 中采样，取值范围通常为 `(0,1]`。                                                                                                 |
| model.model_config.use_past                | bool      | 可选   | False | 是否开启模型增量推理，开启后可使用 Paged Attention 提升推理性能，在模型训练时必须设置为 `False`。                                                                                            |
| model.model_config.max_decode_length       | int       | 可选   | 无     | 设置生成文本的最大长度，包括输入长度。                                                                                                                                      |
| model.model_config.max_length              | int       | 可选   | 无     | 同 `max_decode_length`，与 `max_decode_length` 同时设置时，仅 `max_length` 生效。                                                                                     |
| model.model_config.max_new_tokens          | int       | 可选   | 无     | 设置生成新文本的最大长度，不包括输入长度，与`max_length`同时设置时，仅 `max_new_tokens` 生效。                                                                                           |
| model.model_config.min_length              | int       | 可选   | 无     | 设置生成文本的最小长度，包括输入长度。                                                                                                                                      |
| model.model_config.min_new_tokens          | int       | 可选   | 无     | 设置生成新文本的最小长度，不包括输入长度，与 `min_length` 同时设置时，仅 `min_new_tokens`生效。                                                                                          |
| model.model_config.repetition_penalty      | float     | 可选   | 1.0   | 设置生成重复文本的惩罚系数，`repetition_penalty` 不小于 1；等于 1 时，不对重复输出进行惩罚。                                                                                              |
| model.model_config.block_size              | int       | 可选   | 无     | 设置 Paged Attention中block 的大小，仅 `use_past=True` 时生效。                                                                                                      |
| model.model_config.num_blocks              | int       | 可选   | 无     | 设置 Paged Attention中block 的总数，仅 `use_past=True` 时生效，应满足 `batch_size×seq_length <= block_size×num_blocks`。                                                 |
| model.model_config.return_dict_in_generate | bool      | 可选   | False | 是否以字典形式返回 `generate` 接口的推理结果，默认为 `False`。                                                                                                                |
| model.model_config.output_scores           | bool      | 可选   | False | 是否以字典形式返回结果时，包含每次前向生成时的输入softmax前的分数，默认为 `False`。                                                                                                        |
| model.model_config.output_logits           | bool      | 可选   | False | 是否以字典形式返回结果时，包含每次前向生成时模型输出的logits，默认为 `False`。                                                                                                           |
| model.model_config.layers_per_stage        | list(int) | 可选   | None  | 设置开启 pipeline stage 时，每个 stage 分配到的 transformer 层数。默认为 `None`，表示每个 stage 平均分配。设置的值为一个长度为 pipeline stage 数量的整数列表，第 i 位表示第 i 个 stage 被分配到的 transformer 层数。 |
| model.model_config.bias_swiglu_fusion      | bool      | 可选   | False | 是否使用 swiglu 融合算子，默认为 `False`。                                                                                                                            |
| model.model_config.apply_rope_fusion       | bool      | 可选   | False | 是否使用 RoPE 融合算子，默认为 `False`。                                                                                                                              |

除了上述模型的基本配置，MoE模型需要单独配置一些MoE模块的超参，由于不同模型使用的参数会有不同，仅对通用配置进行说明：

| 参数名称                                  | 数据类型        | 是否可选 | 默认值    | 取值说明                                                                                        |
|---------------------------------------|-------------|------|--------|---------------------------------------------------------------------------------------------|
| moe_config.expert_num                 | int         | 必选   | 无      | 设置路由专家数量。                                                                                   |
| moe_config.shared_expert_num          | int         | 必选   | 无      | 设置共享专家数量。                                                                                   |
| moe_config.moe_intermediate_size      | int         | 必选   | 无      | 设置专家层中间维度大小。                                                                                |
| moe_config.capacity_factor            | int         | 必选   | 无      | 设置专家容量因子。                                                                                   |
| moe_config.num_experts_chosen         | int         | 必选   | 无      | 设置每个 token 选择专家数目。                                                                          |
| moe_config.enable_sdrop               | bool        | 可选   | False  | 设置是否使能 token 丢弃策略`sdrop`，由于 MindSpore Transformers 的 MoE 是静态 shape 实现，所以不能保留所有 token。       |
| moe_config.aux_loss_factor            | list(float) | 可选   | 无      | 设置均衡性 loss 的权重。                                                                             |
| moe_config.first_k_dense_replace      | int         | 可选   | 1      | 设置 moe 层的使能 block，一般设置为 `1`，表示第一个 block 不使能 moe。                                            |
| moe_config.balance_via_topk_bias      | bool        | 可选   | False  | 设置是否使能 `aux_loss_free` 负载均衡算法。                                                              |
| moe_config.topk_bias_update_rate      | float       | 可选   | 无      | 设置`aux_loss_free`负载均衡算法`bias`更新步长。                                                          |
| moe_config.comp_comm_parallel         | bool        | 可选   | False  | 设置是否开启 ffn 的计算通信并行。                                                                         |
| moe_config.comp_comm_parallel_degree  | int         | 可选   | 无      | 设置 ffn 计算通信的分割数。数字越大，重叠越多，但会消耗更多内存。此参数仅在 `comp_comm_parallel=True` 时有效。                      |
| moe_config.moe_shared_expert_overlap  | bool        | 可选   | False  | 设置是否开启共享专家和路由专家的计算通信并行。                                                                     |
| moe_config.use_gating_sigmoid         | bool        | 可选   | False  | 设置 MoE 中 gating 的结果使用 sigmoid 函数进行激活。                                                       |
| moe_config.use_gmm                    | bool        | 可选   | False  | 设置 MoE 专家计算是否使用 GroupedMatmul。                                                              |
| moe_config.use_fused_ops_permute      | bool        | 可选   | False  | 设置是否 MoE 使用 permute、unpermute 融合算子进行性能加速，仅在 `use_gmm=True` 时生效。                               |
| moe_config.enable_deredundency        | bool        | 可选   | False  | 设置是否开启去冗余通信，要求专家并行数是每个节点中NPU卡数量的整数倍，默认值：`False`，当 `use_gmm=True` 时生效。                       |
| moe_config.npu_nums_per_device        | int         | 可选   | 8      | 设置每个节点中 NPU 卡的数量，默认值：`8`，当 `enable_deredundency=True` 时生效。                                  |
| moe_config.enable_gmm_safe_tokens     | bool        | 可选   | False  | 保证每个专家至少分配 1 个 tokens，避免极度负载不均衡情况下，GroupedMatmul 计算失败，默认值为 `False`。当 `use_gmm=True` 时，建议开启。 |

### Mcore 模型配置

使用 MindSpore Transformers 拉起 mcore 模型的任务时，需要在 `model_config` 下对相关超参进行配置，包括模型选择、模型参数、计算类型、MoE 参数等。

由于不同的模型配置会有差异，这里介绍 MindSpore Transformers 中模型常用配置。

| 参数                                                        | 数据类型                  | 是否可选 | 默认值        | 取值说明                                                                                                                                                                                                                                   |
|-----------------------------------------------------------|-----------------------|------|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model.model_config.model_type                             | string                | 必选   | None       | 设置模型配置类，模型配置类需要与模型类匹配使用，即模型配置类中应包含所有模型类使用的参数。例如 `qwen3`、`deepseek_v3` 等。                                                                                                                                                               |
| model.model_config.architectures                          | string / list(string) | 必选   | None       | 设置模型类，构建模型时可以根据模型类对模型进行实例化。例如可设置为 `["Qwen3ForCausalLM"]`、`["DeepseekV3ForCausalLM"]`、`"Qwen3MoeForCausalLM"` 等。                                                                                                                        |
| model.model_config.offset                                 | int / list(int)       | 可选   | 0          | 开启 pp 并行时，需要根据模型层数设置偏移量，构建流水线并行。                                                                                                                                                                                                       |
| model.model_config.vocab_size                             | int                   | 可选   | 128000     | 模型的词表大小。                                                                                                                                                                                                                               |
| model.model_config.hidden_size                            | int                   | 必选   | 0          | Transformer 隐藏层大小。                                                                                                                                                                                                                     |
| model.model_config.ffn_hidden_size                        | int                   | 可选   | None       | Transformer 前馈层大小，对应 HuggingFace 中的 `intermediate_size` 。若不配置，默认设置为 `4 * hidden_size`。                                                                                                                                                 |
| model.model_config.num_layers                             | int                   | 必选   | 0          | Transformer 层数，对应 HuggingFace 中的 `num_hidden_layers`。                                                                                                                                                                                  |
| model.model_config.max_position_embeddings                | int                   | 可选   | 4096       | 模型可以处理的最大序列长度。                                                                                                                                                                                                                         |
| model.model_config.hidden_act                             | string                | 可选   | 'gelu'     | 用于 MLP 中的非线性的激活函数。可选配：`'gelu'`、`'silu'`、`'swiglu'`。                                                                                                                                                                                    |
| model.model_config.num_attention_heads                    | int                   | 必选   | 0          | Transformer 注意力头数。                                                                                                                                                                                                                     |
| model.model_config.num_query_groups                       | int                   | 可选   | None       | 组查询注意力机制的查询组数量，对应 HuggingFace 中的 `num_key_value_heads` 。若不配置，则使用普通注意力机制。                                                                                                                                                               |
| model.model_config.kv_channels                            | int                   | 可选   | None       | 多头注意力机制中的投影权重维度，对应 HuggingFace 中的 `head_dim`。若不配置，则默认为 `hidden_size // num_attention_heads`。                                                                                                                                           |
| model.model_config.layernorm_epsilon                      | float                 | 可选   | 1e-5       | 任何 LayerNorm 操作的 Epsilon 值。                                                                                                                                                                                                            |
| model.model_config.add_bias_linear                        | bool                  | 可选   | True       | 如果开启此项，则将在所有线性层中包含一个偏差项（QKV 投影、core attention 之后以及 MLP 层中的两个）。                                                                                                                                                                         |
| model.model_config.tie_word_embeddings                    | bool                  | 可选   | True       | 是否共享输入和输出 embedding 权重。                                                                                                                                                                                                                |
| model.model_config.use_flash_attention                    | bool                  | 可选   | True       | 是否在注意力层中使用 Flash Attention。                                                                                                                                                                                                            |
| model.model_config.use_contiguous_weight_layout_attention | bool                  | 可选   | False      | 确定 Self Attention 的 QKV 线性投影中的权重排列。仅影响 Self Attention 层。                                                                                                                                                                               |
| model.model_config.hidden_dropout                         | float                 | 可选   | 0.1        | Transformer 隐藏状态的 Dropout 概率。                                                                                                                                                                                                          |
| model.model_config.attention_dropout                      | float                 | 可选   | 0.1        | 后注意力层的 Dropout 概率。                                                                                                                                                                                                                     |
| model.model_config.position_embedding_type                | string                | 可选   | 'rope'     | 用于注意层的位置嵌入类型。                                                                                                                                                                                                                          |
| model.model_config.params_dtype                           | string                | 可选   | 'float32'  | 初始化权重时使用的 dtype。可以配置为 `'float32'`、`'float16'`、`'bfloat16'`。                                                                                                                                                                            |
| model.model_config.compute_dtype                          | string                | 可选   | 'bfloat16' | Linear 层的计算 dtype。可以配置为 `'float32'`、`'float16'`、`'bfloat16'`。                                                                                                                                                                          |
| model.model_config.layernorm_compute_dtype                | string                | 可选   | 'float32'  | LayerNorm 层的计算 dtype。可以配置为 `'float32'`、`'float16'`、`'bfloat16'`。                                                                                                                                                                       |
| model.model_config.softmax_compute_dtype                  | string                | 可选   | 'float32'  | 用于在注意力计算期间计算 softmax 的 dtype。可以配置为 `'float32'`、`'float16'`、`'bfloat16'`。                                                                                                                                                               |
| model.model_config.rotary_dtype                           | string                | 可选   | 'float32'  | 自定义旋转位置嵌入的计算 dtype。可以配置为 `'float32'`、`'float16'`、`'bfloat16'`。                                                                                                                                                                         |
| model.model_config.init_method_std                        | float                 | 可选   | 0.02       | 默认初始化方法的零均值正态的标准偏差，对应 HuggingFace 中的 `initializer_range` 。如果提供了 `init_method` 和 `output_layer_init_method` ，则不使用此方法。                                                                                                                   |
| model.model_config.moe_grouped_gemm                       | bool                  | 可选   | False      | 当每个等级有多个专家时，在单次内核启动中压缩多个本地（可能很小）gemm，以利用分组 GEMM 功能来提高利用率和性能。                                                                                                                                                                           |
| model.model_config.num_moe_experts                        | int                   | 可选   | None       | 用于 MoE 层的专家数量，对应 HuggingFace 中的 `n_routed_experts` 。设置后，将用 MoE 层替换 MLP。设置为 None 则不使用 MoE。                                                                                                                                              |
| model.model_config.num_experts_per_tok                    | int                   | 可选   | 2          | 每个 token 路由到的专家数量。                                                                                                                                                                                                                     |
| model.model_config.moe_ffn_hidden_size                    | int                   | 可选   | None       | MoE 前馈网络隐藏层大小，对应 HuggingFace 中的 `moe_intermediate_size` 。                                                                                                                                                                              |
| model.model_config.moe_router_dtype                       | string                | 可选   | 'float32'  | 用于路由和专家输出加权平均的数据类型。对应 HuggingFace 中的 `router_dense_type` 。                                                                                                                                                                             |
| model.model_config.gated_linear_unit                      | bool                  | 可选   | False      | 对 MLP 中的第一个线性层使用门控线性单元。                                                                                                                                                                                                                |
| model.model_config.norm_topk_prob                         | bool                  | 可选   | True       | 是否使用 top-k 概率进行归一化。                                                                                                                                                                                                                    |
| model.model_config.moe_router_pre_softmax                 | bool                  | 可选   | False      | 为 MoE 启用 pre-softmax（pre-sigmoid）路由，这意味着 softmax 会在 top-k 选择之前进行。默认情况下，softmax 会在 top-k 选择之后进行。                                                                                                                                        |
| model.model_config.moe_token_drop_policy                  | string                | 可选   | 'probs'    | 丢弃 token 的策略。可以是 `'probs'` 或 `'position'`。如果是 `'probs'` ，则丢弃概率最低的 token。 如果是 `'position'` ，则丢弃每个批次末尾的 token。                                                                                                                           |
| model.model_config.moe_router_topk_scaling_factor         | float                 | 可选   | None       | Top-K 路由选择中路由得分的缩放因子，对应 HuggingFace 中的 `routed_scaling_factor` 。仅在启用 `moe_router_pre_softmax` 时有效。默认为 `None`，表示不缩放。                                                                                                                    |
| model.model_config.moe_aux_loss_coeff                     | float                 | 可选   | 0.0        | 辅助损耗的缩放系数。建议初始值为 `1e-2`。                                                                                                                                                                                                               |
| model.model_config.moe_router_load_balancing_type         | string                | 可选   | 'aux_loss' | 路由器的负载均衡策略。 `'aux_loss'` 对应于 GShard 和 SwitchTransformer 中使用的负载均衡损失；`'seq_aux_loss'` 对应于 DeepSeekV2 和 DeepSeekV3 中使用的负载均衡损失，用于计算每个样本的损失；`'sinkhorn'` 对应于 S-BASE 中使用的均衡算法，`'none'` 表示无负载均衡。                                              |
| model.model_config.moe_permute_fusion                     | bool                  | 可选   | False      | 是否使用 moe_token_permute 融合算子，默认为 `False`。                                                                                                                                                                                               |
| model.model_config.moe_router_force_expert_balance        | bool                  | 可选   | False      | 是否在专家路由中使用强制负载均衡。此选项仅用于性能测试，不用于一般用途，默认为 `False`。                                                                                                                                                                                       |
| model.model_config.use_interleaved_weight_layout_mlp      | bool                  | 可选   | True       | 确定 MLP 的 linear_fc1 投影中的权重排列。仅影响 MLP 层。 <br>1. 为 True 时，使用交错排布：`[Gate_weights[0], Hidden_weights[0], Gate_weights[1], Hidden_weights[1], ...]`。<br> 2. 为 False 时，使用连续排布：`[Gate_weights, Hidden_weights]`。<br>注意：这会影响张量内存布局，但不会影响数学等价性。 |
| model.model_config.moe_router_enable_expert_bias          | bool                  | 可选   | False      | 是否在无辅助损失负载均衡策略中，采用动态专家偏差的 TopK 路由。路由决策基于路由得分与专家偏差之和。                                                                                                                                                                                   |
| model.model_config.enable_expert_relocation               | bool                  | 可选   | False      | 是否启用动态专家迁移功能，以实现 MoE 模型中的负载平衡。启用后，专家将根据其负载历史记录在设备之间动态重新分配，以提高训练效率和负载平衡，默认为 `False`。                                                                                                                                                    |
| model.model_config.expert_relocation_initial_iteration    | int                   | 可选   | 20         | 启动专家迁移的初始迭代。专家迁移将在经过这么多次训练迭代后开始。                                                                                                                                                                                                       |
| model.model_config.expert_relocation_freq                 | int                   | 可选   | 50         | 训练迭代中专家迁移的频率。初始迭代后，每 N 次迭代执行一次专家迁移。                                                                                                                                                                                                    |
| model.model_config.print_expert_load                      | bool                  | 可选   | False      | 是否打印专家负载信息。启用后，将在训练期间打印详细的专家负载统计信息，默认为 `False`。                                                                                                                                                                                        |
| model.model_config.moe_router_num_groups                  | int                   | 可选   | None       | 用于分组路由的专家分组数量，等价于 HuggingFace 中的 `n_group`。                                                                                                                                                                                            |
| model.model_config.moe_router_group_topk                  | int                   | 可选   | None       | 组限制路由的选定组数，等价于 HuggingFace 中的 `topk_group`。                                                                                                                                                                                            |
| model.model_config.moe_router_topk                        | int                   | 可选   | 2          | 每个 token 路由到的专家数量，等价于 HuggingFace 中的 `num_experts_per_tok`。配合 `moe_router_num_groups` 和 `moe_router_group_topk` 一起使用时，先分组 `moe_router_num_groups`，然后选出 `moe_router_group_topk`，再从 `moe_router_group_topk` 中选出 `moe_router_topk` 个专家。   |

### 模型训练配置

启动模型训练时，除了模型相关参数，还需要设置trainer、runner_config、学习率以及优化器等训练所需模块的参数。MindSpore Transformers提供了如下配置项。

| 参数                                          | 说明                                                                                                                                                                   | 类型     |
|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|
| trainer.type                                | 设置trainer类，通常不同应用场景的模型会设置不同的trainer类。                                                                                                                                | str    |
| trainer.model_name                          | 设置模型名称，格式为`'{name}_xxb'`，表示模型的某一规格。                                                                                                                                    | str    |
| runner_config.epochs                        | 设置模型训练的轮数。                                                                                                                                                           | int    |
| runner_config.batch_size                    | 设置批处理数据的样本数，该配置会覆盖数据集配置中的`batch_size`。                                                                                                                               | int    |
| runner_config.sink_mode                     | 是否开启数据下沉模式。                                                                                                                                                          | bool   |
| runner_config.sink_size                     | 设置每次从Host下发到Device的迭代数量，仅`sink_mode=True`时生效，此参数将在后续版本中废弃。                                                                                                           | int    |
| runner_config.gradient_accumulation_steps   | 设置梯度累积步数，默认值为1，表示不开启梯度累积。                                                                                                                                            | int    |
| runner_wrapper.type                         | 设置wrapper类，一般设置`'MFTrainOneStepCell'`即可。                                                                                                                               | str    |
| runner_wrapper.local_norm                   | 设置打印单卡上各参数的梯度范数。                                                                                                                                                     | bool   |
| runner_wrapper.scale_sense.type             | 设置梯度缩放类，一般设置`'DynamicLossScaleUpdateCell'`即可。                                                                                                                          | str    |
| runner_wrapper.scale_sense.loss_scale_value | 设置loss动态尺度系数，模型loss可以根据该参数配置动态变化。                                                                                                                                    | int    |
| runner_wrapper.use_clip_grad                | 是否开启梯度剪裁，开启可避免反向梯度过大导致训练无法收敛的情况。                                                                                                                                     | bool   |
| lr_schedule.type                            | 设置lr_schedule类，lr_schedule主要用于调整模型训练中的学习率。                                                                                                                           | str    |
| lr_schedule.learning_rate                   | 设置初始化学习率大小。                                                                                                                                                          | float  |
| lr_scale                                    | 是否开启学习率缩放。                                                                                                                                                           | bool   |
| lr_scale_factor                             | 设置学习率缩放系数。                                                                                                                                                           | int    |
| layer_scale                                 | 是否开启层衰减。                                                                                                                                                             | bool   |
| layer_decay                                 | 设置层衰减系数。                                                                                                                                                             | float  |
| optimizer.type                              | 设置优化器类，优化器主要用于计算模型训练的梯度。                                                                                                                                             | str    |
| optimizer.weight_decay                      | 设置优化器权重衰减系数。                                                                                                                                                         | float  |
| optimizer.fused_num                         | 设置`fused_num`个权重进行融合，根据融合算法将融合后的权重更新到网络参数中。默认值为`10`。                                                                                                                 | int    |
| optimizer.interleave_step                   | 设置选取待融合权重的step间隔数，每`interleave_step`个step取一次权重作为候选权重进行融合。默认值为`1000`。                                                                                                 | int    |
| optimizer.fused_algo                        | 设置融合算法，支持`ema`和`sma`。默认值为`ema`。                                                                                                                                      | string |
| optimizer.ema_alpha                         | 设置融合系数，仅在`fused_algo`=`ema`时生效。默认值为`0.2`。                                                                                                                            | float  |
| train_dataset.batch_size                    | 同`runner_config.batch_size`。                                                                                                                                         | int    |
| train_dataset.input_columns                 | 设置训练数据集输入的数据列。                                                                                                                                                       | list   |
| train_dataset.output_columns                | 设置训练数据集输出的数据列。                                                                                                                                                       | list   |
| train_dataset.construct_args_key            | 设置模型`construct`输入的数据集部分`keys`, 按照字典序传入模型，当模型的传参顺序和数据集输入的顺序不一致时使用该功能。                                                                                                 | list   |
| train_dataset.column_order                  | 设置训练数据集输出数据列的顺序。                                                                                                                                                     | list   |
| train_dataset.num_parallel_workers          | 设置读取训练数据集的进程数。                                                                                                                                                       | int    |
| train_dataset.python_multiprocessing        | 是否开启Python多进程模式提升数据处理性能。                                                                                                                                             | bool   |
| train_dataset.drop_remainder                | 是否在最后一个批处理数据包含样本数小于batch_size时，丢弃该批处理数据。                                                                                                                             | bool   |
| train_dataset.repeat                        | 设置数据集重复数据次数。                                                                                                                                                         | int    |
| train_dataset.numa_enable                   | 设置NUMA的默认状态为数据读取启动状态。                                                                                                                                                | bool   |
| train_dataset.prefetch_size                 | 设置预读取数据量。                                                                                                                                                            | int    |
| train_dataset.data_loader.type              | 设置数据加载类。                                                                                                                                                             | str    |
| train_dataset.data_loader.dataset_dir       | 设置加载数据的路径。                                                                                                                                                           | str    |
| train_dataset.data_loader.shuffle           | 是否在读取数据集时对数据进行随机排序。                                                                                                                                                  | bool   |
| train_dataset.transforms                    | 设置数据增强相关选项。                                                                                                                                                          | -      |
| train_dataset_task.type                     | 设置dataset类，该类用于对数据加载类以及其他相关配置进行封装。                                                                                                                                   | str    |
| train_dataset_task.dataset_config           | 通常设置为`train_dataset`的引用，包含`train_dataset`的所有配置项。                                                                                                                     | -      |
| auto_tune                                   | 是否开启数据处理参数自动调优，详情可参考[set_enable_autotune](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.config.set_enable_autotune.html)。          | bool   |
| filepath_prefix                             | 设置数据优化后的参数配置的保存路径。                                                                                                                                                   | str    |
| autotune_per_step                           | 设置自动数据加速的配置调整step间隔，详情可参考[set_autotune_interval](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.config.set_autotune_interval.html)。 | int    |

### 并行配置

为了提升模型的性能，在大规模集群的使用场景中通常需要为模型配置并行策略，详情可参考[分布式并行](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/parallel_training.html)，MindSpore Transformers中的并行配置如下。

| 参数                                                              | 说明                                                                                                                                                                                                                                                                                  | 类型   |
|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| use_parallel                                                    | 是否开启并行模式。                                                                                                                                                                                                                                                                           | bool |
| parallel_config.data_parallel                                   | 设置数据并行数。                                                                                                                                                                                                                                                                            | int  |
| parallel_config.model_parallel                                  | 设置模型并行数。                                                                                                                                                                                                                                                                            | int  |
| parallel_config.context_parallel                                | 设置序列并行数。                                                                                                                                                                                                                                                                            | int  |
| parallel_config.pipeline_stage                                  | 设置流水线并行数。                                                                                                                                                                                                                                                                           | int  |
| parallel_config.micro_batch_num                                 | 设置流水线并行的微批次大小，在`parallel_config.pipeline_stage`大于1时，应满足`parallel_config.micro_batch_num` >= `parallel_config.pipeline_stage`。                                                                                                                                                       | int  |
| parallel_config.seq_split_num                                   | 在序列流水线并行中设置序列分割数，该数应为序列长度的除数。                                                                                                                                                                                                                                                       | int  |
| parallel_config.gradient_aggregation_group                      | 设置梯度通信算子融合组的大小。                                                                                                                                                                                                                                                                     | int  |
| parallel_config.context_parallel_algo                           | 设置长序列并行方案，可选`colossalai_cp`、`ulysses_cp`和`hybrid_cp`，仅在`context_parallel`切分数大于1时生效。                                                                                                                                                                                                 | str  |
| parallel_config.ulysses_degree_in_cp                            | 设置Ulysses序列并行维度，与`hybrid_cp`长序列并行方案同步配置，需要确保`context_parallel`可以被该参数整除且大于1，同时确保`ulysses_degree_in_cp`可以被attention head数整除。                                                                                                                                                          | int  |
| micro_batch_interleave_num                                      | 设置多副本并行数，大于1时开启多副本并行。通常在使用模型并行时开启，主要用于优化模型并行产生的通信损耗，仅使用流水并行时不建议开启。详情可参考[MicroBatchInterleaved](https://www.mindspore.cn/docs/zh-CN/master/api_python/parallel/mindspore.parallel.nn.MicroBatchInterleaved.html)。                                                                    | int  |
| parallel.parallel_mode                                          | 设置并行模式，`0`表示数据并行模式, `1`表示半自动并行模式, `2`表示自动并行模式, `3`表示混合并行模式，一般设置为半自动并行模式。                                                                                                                                                                                                            | int  |
| parallel.gradients_mean                                         | 是否在梯度AllReduce后执行平均算子。通常半自动并行模式下设为`False`，数据并行模式下设为`True`。                                                                                                                                                                                                                          | bool |
| parallel.enable_alltoall                                        | 是否在通信期间生成AllToAll通信算子。通常仅在MOE场景下设为`True`，默认值为`False`。                                                                                                                                                                                                                               | bool |
| parallel.full_batch                                             | 是否在并行模式下从数据集中读取加载完整的批数据，设置为`True`表示所有rank都读取完整的批数据，设置为`False`表示每个rank仅加载对应的批数据，设置为`False`时必须设置对应的`dataset_strategy`。                                                                                                                                                                | bool |
| parallel.dataset_strategy                                       | 仅支持`List of List`类型且仅在`full_batch=False`时生效，列表中子列表的个数需要等于`train_dataset.input_columns`的长度，并且列表中的每个子列表需要和数据集返回的数据的shape保持一致。一般在数据的第1维进行数据并行切分，所以子列表的第1位数配置与`data_parallel`相同，其他位配置为`1`。具体原理可以参考[数据集切分](https://www.mindspore.cn/tutorials/zh-CN/master/parallel/dataset_slice.html)。 | list |
| parallel.search_mode                                            | 设置全自动并行策略搜索模式，可选`recursive_programming`、`dynamic_programming`和`sharding_propagation`，仅在全自动并行模式下生效，实验性接口。                                                                                                                                                                            | str  |
| parallel.strategy_ckpt_save_file                                | 设置并行切分策略文件的保存路径。                                                                                                                                                                                                                                                                    | str  |
| parallel.strategy_ckpt_config.only_trainable_params             | 是否仅保存（或加载）可训练参数的切分策略信息，默认为`True`，当网络中存在冻结的参数但又需要切分时将该参数设为`False`。                                                                                                                                                                                                                   | bool |
| parallel.enable_parallel_optimizer                              | 是否开启优化器并行。<br/>1. 在数据并行模式下将模型权重参数按device数进行切分。<br/>2. 在半自动并行模式下将模型权重参数按`parallel_config.data_parallel`进行切分。                                                                                                                                                                         | bool |
| parallel.parallel_optimizer_config.gradient_accumulation_shard  | 设置累计的梯度变量是否在数据并行的维度上进行切分，仅`enable_parallel_optimizer=True`时生效。                                                                                                                                                                                                                      | bool |
| parallel.parallel_optimizer_config.parallel_optimizer_threshold | 设置优化器权重参数切分的阈值，仅`enable_parallel_optimizer=True`时生效。                                                                                                                                                                                                                                | int  |
| parallel.parallel_optimizer_config.optimizer_weight_shard_size  | 设置优化器权重参数切分通信域的大小，要求该值可以整除`parallel_config.data_parallel`，仅`enable_parallel_optimizer=True`时生效。                                                                                                                                                                                     | int  |
| parallel.pipeline_config.pipeline_interleave                    | 使能interleave，使用Seq-Pipe或ZeroBubbleV（也称为DualPipeV）流水线并行时需设置为`true`。                                                                                                                                                                                                                  | bool |
| parallel.pipeline_config.pipeline_scheduler                     | 流水线调度策略，目前只支持`"seqpipe"`和`"zero_bubble_v"`。                                                                                                                                                                                                                                         | str  |

> 配置并行策略时应满足：device_num = data_parallel × model_parallel × context_parallel × pipeline_stage。

### 模型优化配置

1. MindSpore Transformers提供重计算相关配置，以降低模型在训练时的内存占用，详情可参考[重计算](https://www.mindspore.cn/mindformers/docs/zh-CN/master/advanced_development/performance_optimization.html#重计算)。

   | 参数                                                 | 说明                             | 类型              |
   |----------------------------------------------------|--------------------------------|-----------------|
   | recompute_config.recompute                         | 是否开启重计算。                       | bool/list/tuple |
   | recompute_config.select_recompute                  | 开启选择重计算，只针对attention层的算子进行重计算。 | bool/list       |
   | recompute_config.parallel_optimizer_comm_recompute | 是否对由优化器并行引入的AllGather通信进行重计算。  | bool/list       |
   | recompute_config.mp_comm_recompute                 | 是否对由模型并行引入的通信进行重计算。            | bool            |
   | recompute_config.recompute_slice_activation        | 是否对保留在内存中的Cell输出切片。            | bool            |
   | recompute_config.select_recompute_exclude          | 关闭指定算子的重计算，只对Primitive算子有效。    | bool/list       |
   | recompute_config.select_comm_recompute_exclude     | 关闭指定算子的通讯重计算，只对Primitive算子有效。  | bool/list       |

2. MindSpore Transformers提供细粒度激活值SWAP相关配置，以降低模型在训练时的内存占用，详情可参考[细粒度激活值SWAP](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/memory_optimization.html#%E7%BB%86%E7%B2%92%E5%BA%A6%E6%BF%80%E6%B4%BB%E5%80%BCswap)。

   | 参数                           | 说明                                                                       | 类型   |
   |------------------------------|--------------------------------------------------------------------------|------|
   | swap_config.swap             | 是否开启激活值SWAP。                                                             | bool |
   | swap_config.default_prefetch | 设置激活值卸载至host时的内存释放时机与开始取回device的时机，仅在开启激活值SWAP且未设置layer_swap与op_swap时生效。 | int  |
   | swap_config.layer_swap       | 选择特定的层使能激活值SWAP。                                                         | list |
   | swap_config.op_swap          | 选择特定层中的特定算子使能激活值SWAP。                                                    | list |

### Callbacks配置

MindSpore Transformers提供封装后的Callbacks函数类，主要实现在模型训练过程中返回模型的训练状态并输出、保存模型权重文件等一些操作，目前支持以下几个Callbacks函数类。

1. MFLossMonitor

   该回调函数类主要用于在训练过程中对训练进度、模型Loss、学习率等信息进行打印，有如下几个可配置项：

   | 参数名称                            | 数据类型  | 是否可选 | 默认值   | 取值说明                                                                                                         |
   |---------------------------------|-------|------|-------|--------------------------------------------------------------------------------------------------------------|
   | learning_rate                   | float | 可选   | None  | 设置 `MFLossMonitor` 中初始化学习率。用于日志打印和训练进度计算。若未设置，则尝试从优化器或其他配置中获取。                                               |
   | per_print_times                 | int   | 可选   | 1     | 设置 `MFLossMonitor` 中日志信息的打印频率，单位为“步”。默认值为 `1`，表示每训练一步打印一次日志信息。                                               |
   | micro_batch_num                 | int   | 可选   | 1     | 设置训练中每一步处理的微批次（micro batch）数量，用于计算实际 loss 值。若未配置，则与 [并行配置](#并行配置) 中 `parallel_config.micro_batch_num` 一致。    |
   | micro_batch_interleave_num      | int   | 可选   | 1     | 设置训练中每一步的多副本微批次大小，用于 loss 计算。若未配置，则与 [并行配置](#并行配置) 中 `micro_batch_interleave_num` 一致。                        |
   | origin_epochs                   | int   | 可选   | None  | 设置 `MFLossMonitor` 中的总训练轮数（epochs）。若未配置，则与 [模型训练配置](#模型训练配置) 中 `runner_config.epochs` 一致。                    |
   | dataset_size                    | int   | 可选   | None  | 设置 `MFLossMonitor` 中数据集的样本总数。若未配置，则自动使用实际加载的数据集大小。                                                           |
   | initial_epoch                   | int   | 可选   | 0     | 设置 `MFLossMonitor` 中训练起始轮数，默认值为 `0`，表示从第0轮开始计数。断点续训时可用于恢复训练进度。                                               |
   | initial_step                    | int   | 可选   | 0     | 设置 `MFLossMonitor` 中训练起始步数，默认值为 `0`。断点续训时可用于对齐日志和进度条。                                                        |
   | global_batch_size               | int   | 可选   | 0     | 设置 `MFLossMonitor` 中的全局批大小（即每个训练 step 所使用的总样本数）。若未配置，则根据数据集大小和并行策略自动计算。                                      |
   | gradient_accumulation_steps     | int   | 可选   | 1     | 设置 `MFLossMonitor` 中的梯度累积步数。若未配置，则与 [模型训练配置](#模型训练配置) 中 `gradient_accumulation_steps` 一致。用于 loss 归一化和训练进度估算。 |
   | check_for_nan_in_loss_and_grad  | bool  | 可选   | False | 是否在 `MFLossMonitor` 中开启损失值和梯度的 NaN/Inf 检测。开启后，若检测到溢出（NaN 或 INF），则终止训练，默认值为`False`。建议在调试阶段开启以提升训练稳定性。         |

2. SummaryMonitor

   该回调函数类主要用于收集Summary数据，详情可参考[mindspore.SummaryCollector](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.SummaryCollector.html)。

3. CheckpointMonitor

   该回调函数类主要用于在模型训练过程中保存模型权重文件，有如下可配置项：

   | 参数名称                            | 数据类型   | 是否可选 | 默认值    | 取值说明                                                                                                                                         |
   |---------------------------------|--------|------|--------|----------------------------------------------------------------------------------------------------------------------------------------------|
   | prefix                          | string | 可选   | 'CKP'  | 设置保存权重文件名的前缀。例如生成 `CKP-100.ckpt`。若未配置，则使用默认值 `'CKP'`。                                                                                        |
   | directory                       | string | 可选   | None   | 设置权重文件的保存目录。若未配置，则默认保存在 `output_dir` 指定路径下的 `checkpoint/` 子目录中。                                                                              |
   | save_checkpoint_seconds         | int    | 可选   | 0      | 以时间间隔方式设置自动保存权重的周期（单位：秒）。与 `save_checkpoint_steps` 互斥，优先级更高。例如每 3600 秒保存一次。                                                                  |
   | save_checkpoint_steps           | int    | 可选   | 1      | 以训练步数间隔方式设置自动保存权重的周期（单位：steps）。与 `save_checkpoint_seconds` 互斥，若两者均设置，以时间优先。例如每1000步保存一次。                                                     |
   | keep_checkpoint_max             | int    | 可选   | 5      | 最多保留的权重文件数量。当保存数量超过该值时，系统将按创建时间顺序删除最早的文件，确保总数不超过此限制。用于控制磁盘空间使用。                                                                              |
   | keep_checkpoint_per_n_minutes   | int    | 可选   | 0      | 每隔 N 分钟保留一个权重。这是一种基于时间窗口的保留策略，常用于长期训练中平衡存储与恢复灵活性。例如设置为 `60` 表示每小时至少保留一个权重。                                                                   |
   | integrated_save                 | bool   | 可选   | True   | 是否开启聚合保存权重文件：<br/>• `True`：在保存权重文件时聚合所有device的权重，即所有device权重一致；<br/>• `False`：所有device各自保存自己的权重。<br/>在半自动并行模式下建议设为 `False`，以避免保存权重文件时出现内存问题。 |
   | save_network_params             | bool   | 可选   | False  | 是否仅保存模型权重，默认值为 `False`。                                                                                                                      |
   | save_trainable_params           | bool   | 可选   | False  | 是否额外单独保存可训练参数（即部分微调时模型的参数权重）。                                                                                                                |
   | async_save                      | bool   | 可选   | False  | 是否异步执行权重保存。开启后保存操作不会阻塞训练主流程，提升训练效率，但需注意 I/O 资源竞争可能导致延迟写入。                                                                                    |
   | remove_redundancy               | bool   | 可选   | False  | 保存权重时是否去除模型权重的冗余，默认值为 `False`。                                                                                                               |
   | checkpoint_format               | string | 可选   | 'ckpt' | 保存的模型权重格式，默认值为 `ckpt`。可选 `ckpt`，`safetensors`。                                                                                               |
   | embedding_local_norm_threshold  | float  | 可选   | 1.0    | 健康监测中用于检测 embedding 层梯度或输出范数异常的阈值。若 norm 超过该值，可能触发告警或数据跳过机制，防止训练发散。默认值为 `1.0`，可根据模型规模调整。                                                     |

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

| 参数名称                            | 数据类型 | 是否可选 | 默认值   | 取值说明                                                                                                                            |
|---------------------------------|------|------|-------|---------------------------------------------------------------------------------------------------------------------------------|
| processor.type                  | str  | 必选   | None  | 设置使用的数据处理类（Processor）的名称，例如 `LlamaProcessor`、`Qwen2Processor` 等。该类决定整体输入数据的预处理流程，需与模型结构匹配。                                      |
| processor.return_tensors        | str  | 可选   | 'ms'  | 设置数据处理后返回的张量类型。可设置为 `'ms'`，表示 MindSpore Tensor。                                                                                 |
| processor.image_processor.type  | str  | 必选   | None  | 设置图像数据处理类的类型。负责图像归一化、缩放、裁剪等操作，需与模型视觉编码器兼容。                                                                                      |
| processor.tokenizer.type        | str  | 必选   | None  | 设置文本分词器（Tokenizer）的类型，例如 `LlamaTokenizer`、`Qwen2Tokenizer` 等。决定文本如何被切分为子词或 token，需与语言模型部分一致。                                    |
| processor.tokenizer.vocab_file  | str  | 必选   | None  | 设置 tokenizer 所需的词汇表文件路径（如 `vocab.txt` 或 `tokenizer.model`），具体文件类型取决于 tokenizer 实现。必须与 `processor.tokenizer.type` 对应，否则可能导致加载失败。 |

### 模型评估配置

MindSpore Transformers提供模型评估功能，同时支持模型边训练边评估功能，以下是模型评估相关配置。

| 参数名称                | 数据类型   | 是否可选 | 默认值   | 取值说明                                                             |
|---------------------|--------|------|-------|------------------------------------------------------------------|
| eval_dataset        | dict   | 必选   | 无     | 用于评估的数据集配置，使用方式与 `train_dataset` 相同。                             |
| eval_dataset_task   | dict   | 必选   | 无     | 评估任务的配置，使用方式与数据集任务配置一致（如预处理、批大小等），用于定义评估流程。                      |
| metric.type         | string | 必选   | 无     | 设置评估的类型，如 `'Accuracy'`、`'F1'` 等，具体取值需与支持的评估指标一致。                 |
| do_eval             | bool   | 可选   | False | 是否开启边训练边评估功能。                                                    |
| eval_step_interval  | int    | 可选   | 100   | 设置评估的 step 间隔，默认值为 `100`，小于等于 0 表示关闭按 step 间隔评估。                 |
| eval_epoch_interval | int    | 可选   | -1    | 设置评估的 epoch 间隔，默认值为 `-1`，小于 0 表示关闭按 epoch 间隔评估；不建议在数据下沉模式下使用该配置。 |

### Profile配置

MindSpore Transformers提供Profile作为模型性能调优的主要工具，详情可参考[性能调优指南](https://www.mindspore.cn/mindformers/docs/zh-CN/master/advanced_development/performance_optimization.html)。以下是Profile相关配置。

| 参数名称                  | 数据类型   | 是否可选 | 默认值   | 取值说明                                                                                                                                        |
|-----------------------|--------|------|-------|---------------------------------------------------------------------------------------------------------------------------------------------|
| profile               | bool   | 可选   | False | 是否开启性能采集工具，默认值为 `False`，详情可参考[mindspore.Profiler](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Profiler.html)。 |
| profile_start_step    | int    | 可选   | 1     | 设置开始采集性能数据的 step 数，默认值为 `1`。                                                                                                                |
| profile_stop_step     | int    | 可选   | 10    | 设置停止采集性能数据的 step 数，默认值为 `10`。                                                                                                               |
| profile_communication | bool   | 可选   | False | 设置是否在多设备训练中收集通信性能数据，使用单卡训练时，该参数无效，默认值为 `False`。                                                                                             |
| profile_memory        | bool   | 可选   | True  | 设置是否收集 Tensor 内存数据，默认值为 `True`。                                                                                                             |
| profile_rank_ids      | list   | 可选   | None  | 设置开启性能采集的 rank ids，默认值为 `None`，表示所有 rank id 均开启性能采集。                                                                                        |
| profile_pipeline      | bool   | 可选   | False | 设置是否按流水线并行每个 stage 的其中一张卡开启性能采集，默认值为 `False`。                                                                                               |
| profile_output        | string | 必选   | 无     | 设置保存性能采集生成文件的文件夹路径。                                                                                                                         |
| profiler_level        | int    | 可选   | 1     | 设置采集数据的级别，可选值为 `(0, 1, 2)`，默认值为 `1`。                                                                                                        |
| with_stack            | bool   | 可选   | False | 设置是否收集 Python 侧的调用栈数据，默认值为 `False`。                                                                                                         |
| data_simplification   | int    | 可选   | False | 设置是否开启数据精简，开启后将在导出性能采集数据后删除 FRAMEWORK 目录以及其他多余数据，默认为 `False`。                                                                               |
| init_start_profile    | bool   | 可选   | False | 设置是否在 Profiler 初始化时开启采集性能数据，设置 `profile_start_step` 时该参数不生效，开启 `profile_memory` 时需要将该参数设为 `True`。                                           |
| mstx                  | bool   | 可选   | False | 设置是否收集 mstx 时间戳记录，包括训练 step、HCCL 通信算子等，默认值为 `False`。                                                                                        |

### 指标监控配置

指标监控配置主要用于配置训练过程中各指标的记录方式，详情可参考[训练指标监控](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/monitor.html)。以下是MindSpore Transformers中通用的指标监控配置项说明：

| 参数名称                                             | 数据类型                  | 是否可选 | 默认值      | 取值说明                                                                                                                                   |
|--------------------------------------------------|-----------------------|------|----------|----------------------------------------------------------------------------------------------------------------------------------------|
| monitor_config.monitor_on                        | bool                  | 可选   | False    | 设置是否开启监控。默认为`False`，此时以下所有参数不生效。                                                                                                       |
| monitor_config.dump_path                         | string                | 可选   | './dump' | 设置训练过程中 `local_norm`、`device_local_norm`、`local_loss` 指标文件的保存路径。未设置或设置为 `null` 时取默认值 `'./dump'`。                                       |
| monitor_config.target                            | list(string)          | 可选   | ['.*']   | 设置指标 `优化器状态` 和 `local_norm` 所监控的的目标参数的名称（片段），可为正则表达式。未设置或设置为 `null` 时取默认值 `['.*']`，即指定所有参数。                                            |
| monitor_config.invert                            | bool                  | 可选   | False    | 设置反选 `monitor_config.target` 所指定的参数，默认为`False`。                                                                                        |
| monitor_config.step_interval                     | int                   | 可选   | 1        | 设置记录指标的频率。默认为 `1`，即每个 step 记录一次。                                                                                                       |
| monitor_config.local_loss_format                 | string / list(string) | 可选   | null     | 设置指标 `local_loss` 的记录形式，可选值为字符串 `'tensorboard'` 和 `'log'`（分别表示写入 Tensorboard 和写入日志），或由两者组成的列表，或 `null`。未设置时默认为 `null`，表示不监控该指标。        |
| monitor_config.device_local_loss_format          | string / list(string) | 可选   | null     | 设置指标 `device_local_loss` 的记录形式，可选值为字符串 `'tensorboard'` 和 `'log'`（分别表示写入 Tensorboard 和写入日志），或由两者组成的列表，或 `null`。未设置时默认为 `null`，表示不监控该指标。 |
| monitor_config.local_norm_format                 | string / list(string) | 可选   | null     | 设置指标 `local_norm` 的记录形式，可选值为字符串 `'tensorboard'` 和 `'log'`（分别表示写入 Tensorboard 和写入日志），或由两者组成的列表，或 `null`。未设置时默认为`null`，表示不监控该指标。         |
| monitor_config.device_local_norm_format          | string / list(string) | 可选   | null     | 设置指标 `device_local_norm` 的记录形式，可选值为字符串 `'tensorboard'` 和 `'log'`（分别表示写入 Tensorboard 和写入日志），或由两者组成的列表，或 `null`。未设置时默认为 `null`，表示不监控该指标。 |
| monitor_config.optimizer_state_format            | string / list(string) | 可选   | null     | 设置指标 `优化器状态` 的记录形式，可选值为字符串 `'tensorboard'` 和 `'log'`（分别表示写入 Tensorboard 和写入日志），或由两者组成的列表，或 `null`。未设置时默认为 `null`，表示不监控该指标。             |
| monitor_config.weight_state_format               | string / list(string) | 可选   | null     | 设置指标 `权重L2-norm` 的记录形式，可选值为字符串 `'tensorboard'` 和 `'log'`（分别表示写入 Tensorboard 和写入日志），或由两者组成的列表，或 `null`。未设置时默认为 `null`，表示不监控该指标。         |
| monitor_config.throughput_baseline               | int / float           | 可选   | null     | 设置指标 `吞吐量线性度` 的基线值，需要为正数。未设置时默认为 `null`，表示不监控该指标。                                                                                      |
| monitor_config.print_struct                      | bool                  | 可选   | False    | 设置是否打印模型的全部可训练参数名。若为 `True`，则会在第一个 step 开始时打印所有可训练参数的名称，并在 step 结束后退出训练。默认为 `False`。                                                   |
| monitor_config.check_for_global_norm             | bool                  | 可选   | False    | 设置是否开启进程级故障快恢功能。默认为 `False`。                                                                                                           |
| monitor_config.global_norm_spike_threshold       | float                 | 可选   | 3.0      | 设置 global norm 的阈值，当 global norm 超过时触发数据跳过。默认值为 `3.0`。                                                                                 |
| monitor_config.global_norm_spike_count_threshold | int                   | 可选   | 10       | 设置连续异常 global norm 累计的次数，当次数达到该阈值则触发异常中断，终止训练。默认值为 `10`。                                                                               |

### TensorBoard配置

TensorBoard配置主要用于配置训练过程中与TensorBoard相关的参数，便于在训练过程中实时查看和监控训练信息，详情可参考[训练指标监控](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/monitor.html)。以下是MindSpore Transformers中通用的TensorBoard配置项说明：

| 参数名称                                       | 数据类型  | 是否可选 | 默认值   | 取值说明                                                     |
|--------------------------------------------|-------|------|-------|----------------------------------------------------------|
| tensorboard.tensorboard_dir                | str   | 必选   | 无     | 设置 TensorBoard 事件文件的保存路径。                                |
| tensorboard.tensorboard_queue_size         | int   | 可选   | 10    | 设置采集队列的最大缓存值，超过该值便会写入事件文件，默认值为10。                        |
| tensorboard.log_loss_scale_to_tensorboard  | bool  | 可选   | False | 设置是否将 loss scale 信息记录到事件文件，默认为`False`。                   |
| tensorboard.log_timers_to_tensorboard      | bool  | 可选   | False | 设置是否将计时器信息记录到事件文件，计时器信息包含当前训练步骤（或迭代）的时长以及吞吐量，默认为`False`。 |
| tensorboard.log_expert_load_to_tensorboard | bool  | 可选   | False | 设置是否将专家负载记录到事件文件，默认为`False`。                             |
