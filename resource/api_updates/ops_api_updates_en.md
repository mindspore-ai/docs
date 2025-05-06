# mindspore.ops.primitive API Interface Change

Compared with the previous version 2.5.0, the added, deleted and supported platforms change information of `mindspore.ops.primitive` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.ops.AlltoAllV](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/ops/mindspore.ops.AlltoAllV.html#mindspore.ops.AlltoAllV)|New|AllToAllV which support uneven scatter and gather compared with AllToAll.|r2.6.0rc1: Ascend|Communication Operator
[mindspore.ops.CustomOpBuilder](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/ops/mindspore.ops.CustomOpBuilder.html#mindspore.ops.CustomOpBuilder)|New|CustomOpBuilder is used to initialize and configure custom operators for MindSpore.|r2.6.0rc1: Ascend/CPU|Customizing Operator
[mindspore.ops.custom_info_register](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/ops/mindspore.ops.custom_info_register.html#mindspore.ops.custom_info_register)|Changed|A decorator which is used to bind the registration information to the func parameter of mindspore.ops.Custom.|r2.5.0:  => r2.6.0rc1: Ascend/GPU/CPU|Customizing Operator
[mindspore.ops.kernel](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/ops/mindspore.ops.kernel.html#mindspore.ops.kernel)|Changed|The decorator of the Hybrid DSL function for the Custom Op.|r2.5.0: Ascend/GPU/CPU => r2.6.0rc1: GPU/CPU|Customizing Operator
[mindspore.ops.Svd](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/ops/mindspore.ops.Svd.html#mindspore.ops.Svd)|Changed|Computes the singular value decompositions of one or more matrices.|r2.5.0: GPU/CPU => r2.6.0rc1: Ascend/GPU/CPU|Linear Algebraic Operator
[mindspore.ops.Morph](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/ops/mindspore.ops.Morph.html#mindspore.ops.Morph)|New|The Morph Primitive is used to encapsulate a user-defined function fn, allowing it to be used as a custom Primitive.|r2.6.0rc1: |operations--Frame Operators
