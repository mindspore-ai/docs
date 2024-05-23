# mindspore.ops.primitive API Interface Change

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.ops.primitive` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.ops.Custom](https://mindspore.cn/docs/en/r2.3.0rc2/api_python/ops/mindspore.ops.Custom.html#mindspore.ops.Custom)|Changed|Custom primitive is used for user defined operators and is to enhance the expressive ability of built-in primitives.|r2.3.0rc1: Ascend/GPU/CPU => r2.3.0rc2: GPU/CPU|Customizing Operator
[mindspore.ops.MaxPoolWithArgmax](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/ops/mindspore.ops.MaxPoolWithArgmax.html#mindspore.ops.MaxPoolWithArgmax)|Deleted|Please use mindspore.ops.MaxPoolWithArgmaxV2 instead.|r2.3.0rc1: |Neural Network
[mindspore.ops.MaxPoolWithArgmaxV2](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/ops/mindspore.ops.MaxPoolWithArgmaxV2.html#mindspore.ops.MaxPoolWithArgmaxV2)|Deleted|Performs max pooling on the input Tensor and returns both max values and indices.|r2.3.0rc1: Ascend/GPU/CPU|Neural Network
[mindspore.ops.Zeros](https://mindspore.cn/docs/en/r2.3.0rc2/api_python/ops/mindspore.ops.Zeros.html#mindspore.ops.Zeros)|Changed|Zeros will be deprecated in the future.|r2.3.0rc1:  => r2.3.0rc2: Ascend/GPU/CPU|Tensor Construction
