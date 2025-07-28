# mindspore.ops.primitive API Interface Change

Compared with the previous version 2.6.0, the added, deleted and supported platforms change information of `mindspore.ops.primitive` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
|[mindspore.ops.AllGatherV](https://mindspore.cn/docs/en/r2.7.0rc1/api_python/ops/mindspore.ops.AllGatherV.html#mindspore.ops.AllGatherV)|New|Gathers uneven tensors from the specified communication group and returns the tensor which is all gathered.|r2.7.0rc1: Ascend/GPU|Communication Operator
|[mindspore.ops.ReduceScatterV](https://mindspore.cn/docs/en/r2.7.0rc1/api_python/ops/mindspore.ops.ReduceScatterV.html#mindspore.ops.ReduceScatterV)|New|Reduces and scatters uneven tensors from the specified communication group and returns the tensor which is reduced and scattered.|r2.7.0rc1: Ascend/GPU|Communication Operator
