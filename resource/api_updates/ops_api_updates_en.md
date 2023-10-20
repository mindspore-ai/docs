# mindspore.ops.primitive API Interface Change

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.ops.primitive` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.ops.InplaceUpdate](https://mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.InplaceUpdate.html#mindspore.ops.InplaceUpdate)|Changed|r2.1: The InplaceUpdate interface is deprecated. => r2.2: Please use InplaceUpdateV2 instead.|r2.1:  => r2.2: Ascend/GPU/CPU|Array Operation
[mindspore.ops.constexpr](https://mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.constexpr.html#mindspore.ops.constexpr)|Deleted|Creates a PrimitiveWithInfer operator that can infer the value at compile time.||Decorators
[mindspore.ops.Dense](https://mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.Dense.html#mindspore.ops.Dense)|New|The dense connected fusion operator.|r2.1: r2.2: Ascend/GPU/CPU|Neural Network
[mindspore.ops.LRN](https://mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.LRN.html#mindspore.ops.LRN)|Changed|Local Response Normalization.|r2.1: Ascend/GPU/CPU => r2.2: GPU/CPU|Neural Network
[mindspore.ops.MaxPoolWithArgmax](https://mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.MaxPoolWithArgmax.html#mindspore.ops.MaxPoolWithArgmax)|Changed|r2.1: mindspore.ops.MaxPoolWithArgmax is deprecated from version 2.0 and will be removed in a future version, use mindspore.ops.MaxPoolWithArgmaxV2 instead. => r2.2: Please use MaxPoolWithArgmaxV2 instead.|r2.1:  => r2.2: Ascend/GPU/CPU|Neural Network
[mindspore.ops.ScatterNonAliasingAdd](https://mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.ScatterNonAliasingAdd.html#mindspore.ops.ScatterNonAliasingAdd)|Changed|r2.1: The ScatterNonAliasingAdd Interface is deprecated from version 2.1. => r2.2: Please use TensorScatterAdd instead.|r2.1:  => r2.2: Ascend/GPU/CPU|Parameter Operation Operator
