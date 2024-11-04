# mindspore.ops API接口变更

与上一版本相比，MindSpore中`mindspore.ops`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别|
|:----|:----|:----|:----|:----|
[mindspore.ops.unique_with_pad](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.unique_with_pad.html#mindspore.ops.unique_with_pad)|Deprecated|对输入一维Tensor中元素去重，返回一维Tensor中的唯一元素（使用pad_num填充）和相对索引。|r2.3.1: Ascend/GPU/CPU => r2.4.0: Deprecated|Array操作
[mindspore.ops.cummax](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.cummax.html#mindspore.ops.cummax)|Changed|返回一个元组（最值、索引），其中最值是输入Tensor input 沿维度 axis 的累积最大值，索引是每个最大值的索引位置。|r2.3.1: GPU/CPU => r2.4.0: Ascend/GPU/CPU|Reduction函数
[mindspore.ops.cast](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.cast.html#mindspore.ops.cast)|New|转换输入Tensor的数据类型。|r2.4.0: Ascend/GPU/CPU|类型转换
[mindspore.ops.tensordump](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.tensordump.html#mindspore.ops.tensordump)|New|将Tensor保存为Numpy的npy格式的文件。|r2.4.0: Ascend|调试函数
