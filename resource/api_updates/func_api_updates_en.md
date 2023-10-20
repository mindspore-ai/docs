# mindspore.ops API Interface Change

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.ops` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.ops.glu](https://mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.glu.html#mindspore.ops.glu)|Changed|Computes GLU (Gated Linear Unit activation function) of input tensors.|r2.1: Ascend/CPU => r2.2: Ascend/GPU/CPU|Activation Functions
[mindspore.ops.accumulate_n](https://mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.accumulate_n.html#mindspore.ops.accumulate_n)|Changed|Computes accumulation of all input tensors element-wise.|r2.1: Ascend => r2.2: Ascend/GPU|Element-wise Operations
[mindspore.ops.clip_by_norm](https://mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.clip_by_norm.html#mindspore.ops.clip_by_norm)|New|Clip norm of a set of input Tensors.|r2.2: Ascend/GPU/CPU|Gradient Clipping
[mindspore.ops.lrn](https://mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.lrn.html#mindspore.ops.lrn)|Changed|Local Response Normalization.|r2.1: Ascend/GPU/CPU => r2.2: GPU/CPU|Neural Network
[mindspore.ops.eps](https://mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.eps.html#mindspore.ops.eps)|New|Create a Tensor with the same data type and shape as input, and the element value is the minimum value that the corresponding data type can express.|r2.2: Ascend/GPU/CPU|Tensor Creation
