# mindspore.ops API接口变更

与上一版本相比，MindSpore中`mindspore.ops`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别|
|:----|:----|:----|:----|:----|
[mindspore.ops.eq](https://mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/ops/mindspore.ops.eq.html#mindspore.ops.eq)|New|逐元素比较两个输入Tensor是否相等。|r2.3.0rc1: Ascend/GPU/CPU|比较函数 |
[mindspore.ops.scalar_cast](https://mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/ops/mindspore.ops.scalar_cast.html#mindspore.ops.scalar_cast)|Changed|r2.2: 将输入Scalar转换为其他类型。 => r2.3.0rc1: 该接口从2.3版本开始已被弃用，并将在未来版本中被移除，建议使用 int(x) 或 float(x) 代替。|r2.2: Ascend/GPU/CPU => r2.3.0rc1: |类型转换 |
