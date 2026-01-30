# mindspore.ops API接口变更

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
