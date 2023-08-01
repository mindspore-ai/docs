# mindspore.ops.primitive API接口变更

与上一版本相比，MindSpore中`mindspore.ops.primitive`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
|[mindspore.ops.ScatterNonAliasingAdd](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.ScatterNonAliasingAdd.html#mindspore.ops.ScatterNonAliasingAdd)|Deprecated|使用给定值通过加法操作和输入索引来更新Tensor值。|r2.0: Ascend|Parameter操作算子
|[mindspore.ops.custom_info_register](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.custom_info_register.html#mindspore.ops.custom_info_register)|Changed|装饰器，用于将注册信息绑定到： mindspore.ops.Custom 的 func 参数。|r2.1: Ascend/GPU/CPU|r2.0: 装饰器 => r2.1: 自定义算子
|[mindspore.ops.kernel](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.kernel.html#mindspore.ops.kernel)|Changed|用于MindSpore Hybrid DSL函数书写的装饰器。|r2.1: Ascend/GPU/CPU|r2.0: 装饰器 => r2.1: 自定义算子
|[mindspore.ops.ApplyAdamWithAmsgradV2](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.ApplyAdamWithAmsgradV2.html#mindspore.ops.ApplyAdamWithAmsgradV2)|New|根据Adam算法更新变量var。|r2.1: Ascend/GPU/CPU|优化器
|[mindspore.ops.ResizeBilinear](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.ResizeBilinear.html#mindspore.ops.ResizeBilinear)|Deprecated|此接口已弃用，请使用 mindspore.ops.ResizeBilinearV2 。|r2.0: Ascend/GPU/CPU|神经网络
|[mindspore.ops.UpsampleNearest3D](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.UpsampleNearest3D.html#mindspore.ops.UpsampleNearest3D)|New|执行最近邻上采样操作。|r2.1: Ascend/GPU/CPU|神经网络
|[mindspore.ops.UpsampleTrilinear3D](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.UpsampleTrilinear3D.html#mindspore.ops.UpsampleTrilinear3D)|New|输入为五维度Tensor，跨其中三维执行三线性插值上调采样。|r2.1: Ascend/GPU/CPU|神经网络
|[mindspore.ops.SparseToDense](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.SparseToDense.html#mindspore.ops.SparseToDense)|Changed|将稀疏Tensor转换为密集Tensor。|r2.0: GPU/CPU => r2.1: CPU|稀疏算子
|[mindspore.ops.op_info_register](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.op_info_register.html#mindspore.ops.op_info_register)|Deleted|用于注册算子的装饰器。||装饰器
|[mindspore.ops.prim_attr_register](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.prim_attr_register.html#mindspore.ops.prim_attr_register)|Deleted|Primitive属性的注册器。||装饰器
