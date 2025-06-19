# mindspore.ops API接口变更

与2.5.0版本相比，MindSpore中`mindspore.ops`API接口的添加、删除和支持平台的更改信息如下表所示。

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
