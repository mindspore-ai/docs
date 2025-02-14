# mindspore.ops API接口变更

与上一版本2.4.10相比，MindSpore中`mindspore.ops`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.ops.reverse](https://mindspore.cn/docs/zh-CN/r2.5.0/api_python/ops/mindspore.ops.reverse.html#mindspore.ops.reverse)|Changed|r2.4.10: mindspore.ops.reverse() 将在未来版本弃用。 => r2.5.0: 此接口将在未来版本弃用，请使用 mindspore.ops.flip() 代替。|r2.4.10: Ascend/GPU/CPU => r2.5.0: Deprecated |Array操作
[mindspore.ops.all_gather_matmul](https://mindspore.cn/docs/zh-CN/r2.5.0/api_python/ops/mindspore.ops.all_gather_matmul.html#mindspore.ops.all_gather_matmul)|New|TP 切分场景下，实现 allgather 和 matmul 的融合，融合算子内部实现通信和计算流水并行。|r2.5.0: Ascend|MC2 函数
[mindspore.ops.matmul_reduce_scatter](https://mindspore.cn/docs/zh-CN/r2.5.0/api_python/ops/mindspore.ops.matmul_reduce_scatter.html#mindspore.ops.matmul_reduce_scatter)|New|TP 切分场景下，实现 matmul 和 reducescatter 的融合，融合算子内部实现通信和计算流水并行。|r2.5.0: Ascend|MC2 函数
[mindspore.ops.rotary_position_embedding](https://mindspore.cn/docs/zh-CN/r2.5.0/api_python/ops/mindspore.ops.rotary_position_embedding.html#mindspore.ops.rotary_position_embedding)|New|旋转位置编码算法的实现。|r2.5.0: Ascend|图像函数
[mindspore.ops.rotated_iou](https://mindspore.cn/docs/zh-CN/r2.5.0/api_python/ops/mindspore.ops.rotated_iou.html#mindspore.ops.rotated_iou)|New|计算旋转矩形之间的重叠面积。|r2.5.0: Ascend|图像函数
[mindspore.ops.swiglu](https://mindspore.cn/docs/zh-CN/r2.5.0/api_python/ops/mindspore.ops.swiglu.html#mindspore.ops.swiglu)|New|计算Swish门线性单元函数（Swish Gated Linear Unit function）。|r2.5.0: Ascend|激活函数
[mindspore.ops.flash_attention_score](https://mindspore.cn/docs/zh-CN/r2.5.0/api_python/ops/mindspore.ops.flash_attention_score.html#mindspore.ops.flash_attention_score)|New|训练场景下，使用FlashAttention算法实现self-attention的计算。|r2.5.0: Ascend|神经网络
[mindspore.ops.incre_flash_attention](https://mindspore.cn/docs/zh-CN/r2.5.0/api_python/ops/mindspore.ops.incre_flash_attention.html#mindspore.ops.incre_flash_attention)|New|增量推理场景接口。|r2.5.0: Ascend|神经网络
[mindspore.ops.prompt_flash_attention](https://mindspore.cn/docs/zh-CN/r2.5.0/api_python/ops/mindspore.ops.prompt_flash_attention.html#mindspore.ops.prompt_flash_attention)|New|全量推理场景接口。|r2.5.0: Ascend|神经网络
[mindspore.ops.polar](https://mindspore.cn/docs/zh-CN/r2.5.0/api_python/ops/mindspore.ops.polar.html#mindspore.ops.polar)|Changed|将极坐标转化为笛卡尔坐标。|r2.4.10: GPU/CPU => r2.5.0: Ascend/GPU/CPU|逐元素运算
