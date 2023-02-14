# mindspore.ops.function API Interface Change

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.ops.function` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.ops.sigmoid](https://mindspore.cn/docs/en/r1.10/api_python/ops/mindspore.ops.sigmoid.html#mindspore.ops.sigmoid)|New|Computes Sigmoid of input element-wise.|r1.10: Ascend/GPU/CPU|Activation Functions
[mindspore.ops.unsorted_segment_prod](https://mindspore.cn/docs/en/r1.10/api_python/ops/mindspore.ops.unsorted_segment_prod.html#mindspore.ops.unsorted_segment_prod)|Changed|Computes the product of a tensor along segments.|r1.9: Ascend/GPU/CPU => r1.10: Ascend/GPU|Array Operation
