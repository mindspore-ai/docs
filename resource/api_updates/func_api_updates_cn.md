# mindspore.ops.function API接口变更

与上一版本相比，MindSpore中`mindspore.ops.function`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别|
|:----|:----|:----|:----|:----
[mindspore.ops.sigmoid](https://mindspore.cn/docs/zh-CN/r1.10/api_python/ops/mindspore.ops.sigmoid.html#mindspore.ops.sigmoid)|新增|逐元素计算Sigmoid激活函数。|r1.10: Ascend/GPU/CPU|激活函数
[mindspore.ops.unsorted_segment_prod](https://mindspore.cn/docs/zh-CN/r1.10/api_python/ops/mindspore.ops.unsorted_segment_prod.html#mindspore.ops.unsorted_segment_prod)|修改|沿分段计算输入Tensor元素的乘积。|r1.9: Ascend/GPU/CPU => r1.10: Ascend/GPU|Array操作
