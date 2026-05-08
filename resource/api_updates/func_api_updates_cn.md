# mindspore.ops API接口变更

2.9.0版本与2.8.0版本相比，MindSpore中`mindspore.ops`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.ops.communication.P2POp](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.P2POp.html#mindspore.ops.communication.P2POp)|New|用于存放关于 'isend' 、 'irecv' 相关的信息，并用于 batch_isend_irecv 接口的入参。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.TCPStore](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.TCPStore.html#mindspore.ops.communication.TCPStore)|New|一种基于传输控制协议（TCP）的分布式键值存储实现方法。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.all_gather](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.all_gather.html#mindspore.ops.communication.all_gather)|New|从指定通信组中收集张量，并返回收集的张量列表。|r2.9.0: Ascend/CPU|mindspore.ops.communication
[mindspore.ops.communication.all_gather_into_tensor](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.all_gather_into_tensor.html#mindspore.ops.communication.all_gather_into_tensor)|New|汇聚指定的通信组中的tensor，并返回汇聚后的tensor。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.all_gather_into_tensor_uneven](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.all_gather_into_tensor_uneven.html#mindspore.ops.communication.all_gather_into_tensor_uneven)|New|收集并拼接各设备上的张量，各设备上的张量第一维可以不一致。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.all_gather_object](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.all_gather_object.html#mindspore.ops.communication.all_gather_object)|New|在指定通信组中聚合Python对象。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.all_reduce](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.all_reduce.html#mindspore.ops.communication.all_reduce)|New|使用指定方式对通信组内的所有设备的tensor数据进行归约操作，所有设备都得到相同的结果，返回归约操作后的张量。|r2.9.0: Ascend/CPU|mindspore.ops.communication
[mindspore.ops.communication.all_to_all](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.all_to_all.html#mindspore.ops.communication.all_to_all)|New|根据输入/输出张量列表，在所有rank之间分散和收集张量列表。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.all_to_all_single](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.all_to_all_single.html#mindspore.ops.communication.all_to_all_single)|New|使用分割大小在所有rank之间分散和收集输入，并在单个张量中返回结果。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.all_to_all_v_c](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.all_to_all_v_c.html#mindspore.ops.communication.all_to_all_v_c)|New|根据用户指定的分割大小，将输入张量分割并发送到其他设备，在接收分割块后合并为单个输出张量。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.barrier](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.barrier.html#mindspore.ops.communication.barrier)|New|同步通信域内的多个进程。|r2.9.0: Ascend/CPU|mindspore.ops.communication
[mindspore.ops.communication.batch_isend_irecv](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.batch_isend_irecv.html#mindspore.ops.communication.batch_isend_irecv)|New|批量异步发送和接收张量。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.broadcast](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.broadcast.html#mindspore.ops.communication.broadcast)|New|将张量广播到整个通信组。|r2.9.0: Ascend/CPU|mindspore.ops.communication
[mindspore.ops.communication.broadcast_object_list](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.broadcast_object_list.html#mindspore.ops.communication.broadcast_object_list)|New|广播整个输入Python对象组。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.destroy_process_group](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.destroy_process_group.html#mindspore.ops.communication.destroy_process_group)|New|销毁用户集合通信组。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.gather](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.gather.html#mindspore.ops.communication.gather)|New|从指定通信组中收集张量。|r2.9.0: Ascend/CPU|mindspore.ops.communication
[mindspore.ops.communication.gather_into_tensor](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.gather_into_tensor.html#mindspore.ops.communication.gather_into_tensor)|New|从指定通信组中收集张量。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.gather_object](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.gather_object.html#mindspore.ops.communication.gather_object)|New|在单个进程中从整个组收集Python对象。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.get_backend](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.get_backend.html#mindspore.ops.communication.get_backend)|New|获取通信进程组的后端。|r2.9.0: Ascend/CPU|mindspore.ops.communication
[mindspore.ops.communication.get_global_rank](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.get_global_rank.html#mindspore.ops.communication.get_global_rank)|New|返回与用户组中id为 group_rank 的rank对应的world组中的rank id。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.get_group_rank](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.get_group_rank.html#mindspore.ops.communication.get_group_rank)|New|获取与world通信组中的rank ID对应的指定用户通信组中的rank ID。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.get_process_group_ranks](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.get_process_group_ranks.html#mindspore.ops.communication.get_process_group_ranks)|New|获取特定组的rank，并以列表形式返回通信组中的进程rank。|r2.9.0: Ascend/CPU|mindspore.ops.communication
[mindspore.ops.communication.get_rank](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.get_rank.html#mindspore.ops.communication.get_rank)|New|获取指定集合通信组中当前设备的rank ID。|r2.9.0: Ascend/CPU|mindspore.ops.communication
[mindspore.ops.communication.get_world_size](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.get_world_size.html#mindspore.ops.communication.get_world_size)|New|获取指定集合通信组的rank size。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.init_process_group](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.init_process_group.html#mindspore.ops.communication.init_process_group)|New|初始化集合通信库，并创建一个默认的集合通信组。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.irecv](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.irecv.html#mindspore.ops.communication.irecv)|New|异步从src接收张量。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.is_available](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.is_available.html#mindspore.ops.communication.is_available)|New|检查分布式模块是否可用。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.is_initialized](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.is_initialized.html#mindspore.ops.communication.is_initialized)|New|检查默认进程组是否已初始化。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.isend](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.isend.html#mindspore.ops.communication.isend)|New|异步将张量发送到指定的目标rank。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.new_group](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.new_group.html#mindspore.ops.communication.new_group)|New|创建一个新的分布式组。|r2.9.0: Ascend/CPU|mindspore.ops.communication
[mindspore.ops.communication.recv](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.recv.html#mindspore.ops.communication.recv)|New|从src接收张量。|r2.9.0: Ascend/CPU|mindspore.ops.communication
[mindspore.ops.communication.recv_object_list](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.recv_object_list.html#mindspore.ops.communication.recv_object_list)|New|同步接收源的进程的Python对象列表。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.reduce](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.reduce.html#mindspore.ops.communication.reduce)|New|对指定通信组中的进程进行张量归约操作，将结果发送到目标dst（全局rank），并返回发送到目标进程的张量。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.reduce_scatter](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.reduce_scatter.html#mindspore.ops.communication.reduce_scatter)|New|对指定通信组中的张量进行归约和分散操作，并返回归约和分散后的张量。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.reduce_scatter_tensor](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.reduce_scatter_tensor.html#mindspore.ops.communication.reduce_scatter_tensor)|New|对指定通信组中的张量进行归约和分散操作，并返回归约和分散后的张量。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.reduce_scatter_tensor_uneven](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.reduce_scatter_tensor_uneven.html#mindspore.ops.communication.reduce_scatter_tensor_uneven)|New|根据 input_split_sizes 对指定通信组中的张量进行归约操作，并分散到输出张量。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.scatter](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.scatter.html#mindspore.ops.communication.scatter)|New|在指定通信组中的进程之间均匀分散张量。|r2.9.0: Ascend/CPU|mindspore.ops.communication
[mindspore.ops.communication.scatter_object_list](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.scatter_object_list.html#mindspore.ops.communication.scatter_object_list)|New|将 scatter_object_input_list 中的可pickle对象分散到整个组。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.scatter_tensor](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.scatter_tensor.html#mindspore.ops.communication.scatter_tensor)|New|在指定通信组中的进程之间均匀分散张量。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.send](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.send.html#mindspore.ops.communication.send)|New|将张量发送到指定的目标rank。|r2.9.0: Ascend/CPU|mindspore.ops.communication
[mindspore.ops.communication.send_object_list](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.send_object_list.html#mindspore.ops.communication.send_object_list)|New|将输入的Python对象列表同步发送到目的卡上。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.communication.set_comm_ops_inplace](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.communication.set_comm_ops_inplace.html#mindspore.ops.communication.set_comm_ops_inplace)|New|设置通信函数的inplace属性。|r2.9.0: Ascend|mindspore.ops.communication
[mindspore.ops.lightning_indexer](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.lightning_indexer.html#mindspore.ops.lightning_indexer)|New|本算子基于 DeepSeek Sparse Attention（DSA）算法，为每个 query token 计算 Top-k 稀疏索引。|r2.9.0: Ascend|神经网络
[mindspore.ops.sparse_flash_attention](https://mindspore.cn/docs/zh-CN/r2.9.0/api_python/ops/mindspore.ops.sparse_flash_attention.html#mindspore.ops.sparse_flash_attention)|New|针对大序列长度推理场景的稀疏注意力计算模块。|r2.9.0: Ascend|神经网络

2.8.0版本与2.7.2版本相比，MindSpore中`mindspore.ops`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.ops.nsa_compress](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.nsa_compress.html#mindspore.ops.nsa_compress)|New|使用 NSA Compress 算法在 KV 序列维度进行压缩，以降低长上下文训练中的注意力计算开销。|r2.8.0: Ascend|Array操作
[mindspore.ops.nsa_compress_attention](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.nsa_compress_attention.html#mindspore.ops.nsa_compress_attention)|New|使用NSA Compress Attention算法进行注意力压缩计算（Ascend）。|r2.8.0: Ascend|Array操作
[mindspore.ops.sparse_segment_mean](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.sparse_segment_mean.html#mindspore.ops.sparse_segment_mean)|Changed|r2.7.2: 计算输入张量稀疏段的平均值。 => r2.8.0: ops.sparse_segment_mean 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: GPU/CPU => r2.8.0: Deprecated|Array操作
[mindspore.ops.padding](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.padding.html#mindspore.ops.padding)|Changed|r2.7.2: 通过填充0，将输入Tensor的最后一个维度从1扩展到指定大小。 => r2.8.0: ops.padding 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend/GPU/CPU => r2.8.0: Deprecated|图像函数
[mindspore.ops.approximate_equal](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.approximate_equal.html#mindspore.ops.approximate_equal)|Changed|r2.7.2: 返回一个布尔型tensor，表示两个tensor在容忍度内是否逐元素相等。 => r2.8.0: ops.approximate_equal 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend/GPU/CPU => r2.8.0: Deprecated|比较函数
[mindspore.ops.nsa_select_attention](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.nsa_select_attention.html#mindspore.ops.nsa_select_attention)|New|本算子用于在训练场景中计算原生稀疏注意力（Native Sparse Attention）算法的选择性注意力机制。|r2.8.0: Ascend|神经网络
[mindspore.ops.batch_dot](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.batch_dot.html#mindspore.ops.batch_dot)|Changed|r2.7.2: 计算 x1 和 x2 中的向量点积。 => r2.8.0: ops.batch_dot 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend/GPU/CPU => r2.8.0: Deprecated|线性代数函数
[mindspore.ops.accumulate_n](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.accumulate_n.html#mindspore.ops.accumulate_n)|Changed|r2.7.2: 逐元素计算列表中各个tensor的和。 => r2.8.0: ops.accumulate_n 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend/GPU => r2.8.0: Deprecated|逐元素运算

2.7.2版本与2.7.1版本相比，MindSpore中 `mindspore.ops` API接口没有变化。

2.7.0版本与2.6.0版本相比，MindSpore中`mindspore.ops`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别|
|:----|:----|:----|:----|:----|
[mindspore.ops.ring_attention_update](https://mindspore.cn/docs/zh-CN/r2.7.0/api_python/ops/mindspore.ops.ring_attention_update.html#mindspore.ops.ring_attention_update)|New|RingAttentionUpdate算子功能是将两次FlashAttention的输出根据其不同的softmax的max和sum更新。|r2.7.0: Ascend|神经网络

2.6.0版本与2.5.0版本相比，MindSpore中`mindspore.ops`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.ops.reverse](https://mindspore.cn/docs/zh-CN/r2.5.0/api_python/ops/mindspore.ops.reverse.html#mindspore.ops.reverse)|Deleted|此接口将在未来版本弃用，请使用 mindspore.ops.flip() 代替。||Array操作
[mindspore.ops.roll](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.roll.html#mindspore.ops.roll)|Changed|r2.5.0: 沿轴移动Tensor的元素。 => r2.6.0: 按维度移动tensor的元素。|r2.5.0: GPU => r2.6.0: Ascend/GPU|Array操作
[mindspore.ops.unique_with_pad](https://mindspore.cn/docs/zh-CN/r2.5.0/api_python/ops/mindspore.ops.unique_with_pad.html#mindspore.ops.unique_with_pad)|Deleted|对输入一维Tensor中元素去重，返回一维Tensor中的唯一元素（使用pad_num填充）和相对索引。||Array操作
[mindspore.ops.move_to](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.move_to.html#mindspore.ops.move_to)|New|拷贝tensor到目标设备，包含同步和异步两种方式，默认是同步方式。|r2.6.0: Ascend/CPU|Tensor创建
[mindspore.ops.fused_infer_attention_score](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.fused_infer_attention_score.html#mindspore.ops.fused_infer_attention_score)|New|这是一个适配增量和全量推理场景的FlashAttention函数，既可以支持全量计算场景（PromptFlashAttention），也可支持增量计算场景（IncreFlashAttention）。|r2.6.0: Ascend|神经网络
[mindspore.ops.moe_token_permute](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.moe_token_permute.html#mindspore.ops.moe_token_permute)|New|根据 indices 对 tokens 进行排列。|r2.6.0: Ascend|神经网络
[mindspore.ops.moe_token_unpermute](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.moe_token_unpermute.html#mindspore.ops.moe_token_unpermute)|New|根据排序的索引对已排列的标记进行反排列，并可选择将标记与其对应的概率合并。|r2.6.0: Ascend|神经网络
[mindspore.ops.speed_fusion_attention](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.speed_fusion_attention.html#mindspore.ops.speed_fusion_attention)|New|本接口用于实现self-attention的融合计算。|r2.6.0: Ascend|神经网络
[mindspore.ops.scalar_cast](https://mindspore.cn/docs/zh-CN/r2.5.0/api_python/ops/mindspore.ops.scalar_cast.html#mindspore.ops.scalar_cast)|Deleted|该接口从2.3版本开始已被弃用，并将在未来版本中被移除，建议使用 int(x) 或 float(x) 代替。||类型转换
[mindspore.ops.svd](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.svd.html#mindspore.ops.svd)|Changed|计算单个或多个矩阵的奇异值分解。|r2.5.0: GPU/CPU => r2.6.0: Ascend/GPU/CPU|线性代数函数
[mindspore.ops.bessel_i0e](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.bessel_i0e.html#mindspore.ops.bessel_i0e)|Changed|r2.5.0: 逐元素计算指数缩放第一类零阶修正贝塞尔函数。 => r2.6.0: 逐元素计算输入tensor的指数缩放第一类零阶修正贝塞尔函数值。|r2.5.0: Ascend/GPU/CPU => r2.6.0: GPU/CPU|逐元素运算
[mindspore.ops.bessel_i1e](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.bessel_i1e.html#mindspore.ops.bessel_i1e)|Changed|r2.5.0: 逐元素计算指数缩放第一类一阶修正Bessel函数。 => r2.6.0: 逐元素计算输入tensor的指数缩放第一类一阶修正贝塞尔函数值。|r2.5.0: Ascend/GPU/CPU => r2.6.0: GPU/CPU|逐元素运算
