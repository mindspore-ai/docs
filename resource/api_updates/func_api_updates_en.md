# mindspore.ops API Interface Change

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.ops` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.ops.eq](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/ops/mindspore.ops.eq.html#mindspore.ops.eq)|New|Computes the equivalence between two tensors element-wise.|r2.3.0rc1: Ascend/GPU/CPU|Comparison Functions
[mindspore.ops.scalar_cast](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/ops/mindspore.ops.scalar_cast.html#mindspore.ops.scalar_cast)|Changed|r2.2: Casts the input scalar to another type. => r2.3.0rc1: The interface is deprecated from version 2.3 and will be removed in a future version, please use int(x) or float(x) instead.|r2.2: Ascend/GPU/CPU => r2.3.0rc1: |Type Cast
