# mindspore.ops.primitive API接口变更

与2.5.0版本相比，MindSpore中`mindspore.ops.primitive`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.ops.Morph](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.Morph.html#mindspore.ops.Morph)|New|Morph 算子用于对用户自定义函数 fn 进行封装，允许其被当做自定义算子使用。|r2.6.0: |框架算子
[mindspore.ops.Svd](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.Svd.html#mindspore.ops.Svd)|Changed|计算一个或多个矩阵的奇异值分解。|r2.5.0: GPU/CPU => r2.6.0: Ascend/GPU/CPU|线性代数算子
[mindspore.ops.CustomOpBuilder](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.CustomOpBuilder.html#mindspore.ops.CustomOpBuilder)|New|CustomOpBuilder 用于初始化和配置MindSpore的自定义算子。|r2.6.0: Ascend/CPU|自定义算子
[mindspore.ops.custom_info_register](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.custom_info_register.html#mindspore.ops.custom_info_register)|Changed|装饰器，用于将注册信息绑定到： mindspore.ops.Custom 的 func 参数。|r2.5.0:  => r2.6.0: Ascend/GPU/CPU|自定义算子
[mindspore.ops.kernel](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.kernel.html#mindspore.ops.kernel)|Changed|用于MindSpore Hybrid DSL函数书写的装饰器。|r2.5.0: Ascend/GPU/CPU => r2.6.0: GPU/CPU|自定义算子
[mindspore.ops.AlltoAllV](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.AlltoAllV.html#mindspore.ops.AlltoAllV)|New|相对AlltoAll来说，AlltoAllV算子支持不等分的切分和聚合。|r2.6.0: Ascend|通信算子
