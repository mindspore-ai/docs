# mindspore.ops.primitive API接口变更

与上一版本相比，MindSpore中`mindspore.ops.primitive`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.ops.InplaceUpdate](https://mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.InplaceUpdate.html#mindspore.ops.InplaceUpdate)|Changed|InplaceUpdate接口已废弃，请使用接口 mindspore.ops.InplaceUpdateV2 替换，废弃版本2.0。|r2.1:  => r2.2: Ascend/GPU/CPU|Array操作
[mindspore.ops.ScatterNonAliasingAdd](https://mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.ScatterNonAliasingAdd.html#mindspore.ops.ScatterNonAliasingAdd)|Changed|r2.1: 使用给定值通过加法操作和输入索引来更新Tensor值。 => r2.2: ScatterNonAliasingAdd接口已废弃，请使用接口 mindspore.ops.TensorScatterAdd 替换，废弃版本2.1。|r2.1:  => r2.2: Ascend/GPU/CPU|Parameter操作算子
[mindspore.ops.Dense](https://mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.Dense.html#mindspore.ops.Dense)|New|全连接融合算子。|r2.1: r2.2: Ascend/GPU/CPU|神经网络
[mindspore.ops.LRN](https://mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.LRN.html#mindspore.ops.LRN)|Changed|局部响应归一化操作LRN(Local Response Normalization)。|r2.1: Ascend/GPU/CPU => r2.2: GPU/CPU|神经网络
[mindspore.ops.MaxPoolWithArgmax](https://mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.MaxPoolWithArgmax.html#mindspore.ops.MaxPoolWithArgmax)|Changed|mindspore.ops.MaxPoolWithArgmax 从2.0版本开始已被弃用，并将在未来版本中被移除，建议使用 mindspore.ops.MaxPoolWithArgmaxV2 代替。|r2.1:  => r2.2: Ascend/GPU/CPU|神经网络
[mindspore.ops.constexpr](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.constexpr.html#mindspore.ops.constexpr)|Deleted|创建PrimiveWithInfer算子，用于在编译时推断值。||装饰器
