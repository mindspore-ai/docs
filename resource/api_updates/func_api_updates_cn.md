# mindspore.ops API接口变更

与上一版本相比，MindSpore中`mindspore.ops`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别|
|:----|:----|:----|:----|:----|
[mindspore.ops.embedding](https://mindspore.cn/docs/zh-CN/r2.3.0rc2/api_python/ops/mindspore.ops.embedding.html#mindspore.ops.embedding)|New|以 input 中的值作为索引，从 weight 中查询对应的embedding向量。|r2.3.0rc2: Ascend|神经网络|
[mindspore.ops.group_norm](https://mindspore.cn/docs/zh-CN/r2.3.0rc2/api_python/ops/mindspore.ops.group_norm.html#mindspore.ops.group_norm)|New|在mini-batch输入上进行组归一化。|r2.3.0rc2: Ascend/GPU/CPU|神经网络|
[mindspore.ops.layer_norm](https://mindspore.cn/docs/zh-CN/r2.3.0rc2/api_python/ops/mindspore.ops.layer_norm.html#mindspore.ops.layer_norm)|New|在mini-batch输入上应用层归一化（Layer Normalization）。|r2.3.0rc2: Ascend|神经网络|
