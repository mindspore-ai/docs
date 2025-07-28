# mindspore.ops.primitive API接口变更

与上一版本2.6.0相比，MindSpore中`mindspore.ops.primitive`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
|[mindspore.ops.AllGatherV](https://mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/ops/mindspore.ops.AllGatherV.html#mindspore.ops.AllGatherV)|New|从指定的通信组中收集不均匀的张量，并返回全部收集的张量。|r2.7.0rc1: Ascend/GPU|通信算子
|[mindspore.ops.ReduceScatterV](https://mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/ops/mindspore.ops.ReduceScatterV.html#mindspore.ops.ReduceScatterV)|New|规约并且分发指定通信组中不均匀的张量，返回分发后的张量。|r2.7.0rc1: Ascend/GPU|通信算子
