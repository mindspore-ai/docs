# mindspore.ops API Interface Change

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.ops` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.ops.unique_with_pad](https://mindspore.cn/docs/en/r2.4.0/api_python/ops/mindspore.ops.unique_with_pad.html#mindspore.ops.unique_with_pad)|Deprecated|Returns unique elements and relative indexes in 1-D tensor, filled with padding num.|r2.3.1: Ascend/GPU/CPU => r2.4.0: Deprecated|Array Operation
[mindspore.ops.tensordump](https://mindspore.cn/docs/en/r2.4.0/api_python/ops/mindspore.ops.tensordump.html#mindspore.ops.tensordump)|New|Save Tensor in numpy's npy format.|r2.4.0: Ascend|Debugging Functions
[mindspore.ops.cummax](https://mindspore.cn/docs/en/r2.4.0/api_python/ops/mindspore.ops.cummax.html#mindspore.ops.cummax)|Changed|Returns a tuple (values,indices) where 'values' is the cumulative maximum value of input Tensor input along the dimension axis, and indices is the index location of each maximum value.|r2.3.1: GPU/CPU => r2.4.0: Ascend/GPU/CPU|Reduction Functions
[mindspore.ops.cast](https://mindspore.cn/docs/en/r2.4.0/api_python/ops/mindspore.ops.cast.html#mindspore.ops.cast)|New|Returns a tensor with the new specified data type.|r2.4.0: Ascend/GPU/CPU|Type Cast