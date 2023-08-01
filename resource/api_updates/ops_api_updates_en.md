# mindspore.ops.primitive API Interface Change

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.ops.primitive` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
|[mindspore.ops.op_info_register](https://mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.op_info_register.html#mindspore.ops.op_info_register)|Deleted|A decorator which is used to register an operator.||Decorators
|[mindspore.ops.prim_attr_register](https://mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.prim_attr_register.html#mindspore.ops.prim_attr_register)|Deleted|Primitive attributes register.||Decorators
|[mindspore.ops.UpsampleNearest3D](https://mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.UpsampleNearest3D.html#mindspore.ops.UpsampleNearest3D)|New|Performs nearest neighbor upsampling operation.|r2.1: Ascend/GPU/CPU|Image Processing
|[mindspore.ops.UpsampleTrilinear3D](https://mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.UpsampleTrilinear3D.html#mindspore.ops.UpsampleTrilinear3D)|New|Performs upsampling with trilinear interpolation across 3dims for 5dim input Tensor.|r2.1: Ascend/GPU/CPU|Image Processing
|[mindspore.ops.ResizeBilinear](https://mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.ResizeBilinear.html#mindspore.ops.ResizeBilinear)|Deprecated|This API is deprecated, please use the mindspore.ops.ResizeBilinearV2 instead.|r2.0: Ascend/GPU/CPU|Neural Network
|[mindspore.ops.ApplyAdamWithAmsgradV2](https://mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.ApplyAdamWithAmsgradV2.html#mindspore.ops.ApplyAdamWithAmsgradV2)|New|Update var according to the Adam algorithm.|r2.1: Ascend/GPU/CPU|Optimizer
|[mindspore.ops.ScatterNonAliasingAdd](https://mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.ScatterNonAliasingAdd.html#mindspore.ops.ScatterNonAliasingAdd)|Deprecated|r2.0: Applies sparse addition to the input using individual values or slices. => r2.1: The ScatterNonAliasingAdd Interface is deprecated from version 2.1.|r2.0: Ascend|Parameter Operation Operator
|[mindspore.ops.SparseToDense](https://mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.SparseToDense.html#mindspore.ops.SparseToDense)|Changed|Converts a sparse representation into a dense tensor.|r2.0: GPU/CPU => r2.1: CPU|Sparse Operator
|[mindspore.ops.custom_info_register](https://mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.custom_info_register.html#mindspore.ops.custom_info_register)|Changed|A decorator which is used to bind the registration information to the func parameter of mindspore.ops.Custom.|r2.1: Ascend/GPU/CPU|r2.0: Decorators => r2.1: Customizing Operator
|[mindspore.ops.kernel](https://mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.kernel.html#mindspore.ops.kernel)|Changed|The decorator of the Hybrid DSL function for the Custom Op.|r2.1: Ascend/GPU/CPU|r2.0: Decorators => r2.1: Customizing Operator
