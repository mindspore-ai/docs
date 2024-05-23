# mindspore.ops.primitive API接口变更

与上一版本相比，MindSpore中`mindspore.ops.primitive`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.ops.Zeros](https://mindspore.cn/docs/zh-CN/r2.3.0rc2/api_python/ops/mindspore.ops.Zeros.html#mindspore.ops.Zeros)|Changed|创建一个值全为0的Tensor。|r2.3.0rc1:  => r2.3.0rc2: Ascend/GPU/CPU|Tensor创建
[mindspore.ops.MaxPoolWithArgmax](https://mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/ops/mindspore.ops.MaxPoolWithArgmax.html#mindspore.ops.MaxPoolWithArgmax)|Deleted|mindspore.ops.MaxPoolWithArgmax 从2.0版本开始已被弃用，并将在未来版本中被移除，建议使用 mindspore.ops.MaxPoolWithArgmaxV2 代替。|r2.3.0rc1: |神经网络
[mindspore.ops.MaxPoolWithArgmaxV2](https://mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/ops/mindspore.ops.MaxPoolWithArgmaxV2.html#mindspore.ops.MaxPoolWithArgmaxV2)|Deleted|对输入Tensor执行最大池化运算，并返回最大值和索引。|r2.3.0rc1: Ascend/GPU/CPU|神经网络
[mindspore.ops.Custom](https://mindspore.cn/docs/zh-CN/r2.3.0rc2/api_python/ops/mindspore.ops.Custom.html#mindspore.ops.Custom)|Changed|Custom 算子是MindSpore自定义算子的统一接口。|r2.3.0rc1: Ascend/GPU/CPU => r2.3.0rc2: GPU/CPU|自定义算子
