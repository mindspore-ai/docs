# mindspore.ops API Interface Change

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.ops` operators in MindSpore, is shown in the following table.

## Differences between 2.3.0 and 2.3.0rc2

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
|[mindspore.ops.rms_norm](https://mindspore.cn/docs/en/r2.3.0/api_python/ops/mindspore.ops.rms_norm.html#mindspore.ops.rms_norm)|New|The RmsNorm(Root Mean Square Layer Normalization) operator is a normalization operation.|r2.3.0: Ascend|Neural Network

## Differences between 2.3.0rc2 and 2.3.0rc1

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
|[mindspore.ops.embedding](https://mindspore.cn/docs/en/r2.3.0rc2/api_python/ops/mindspore.ops.embedding.html#mindspore.ops.embedding)|New|Retrieve the word embeddings in weight using indices specified in input.|r2.3.0rc2: Ascend|Neural Network
|[mindspore.ops.group_norm](https://mindspore.cn/docs/en/r2.3.0rc2/api_python/ops/mindspore.ops.group_norm.html#mindspore.ops.group_norm)|New|Group Normalization over a mini-batch of inputs.|r2.3.0rc2: Ascend/GPU/CPU|Neural Network
|[mindspore.ops.layer_norm](https://mindspore.cn/docs/en/r2.3.0rc2/api_python/ops/mindspore.ops.layer_norm.html#mindspore.ops.layer_norm)|New|Applies the Layer Normalization to the input tensor.|r2.3.0rc2: Ascend|Neural Network

## Differences between 2.3.0rc1 and 2.2.14

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
|[mindspore.ops.eq](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/ops/mindspore.ops.eq.html#mindspore.ops.eq)|New|Computes the equivalence between two tensors element-wise.|r2.3.0rc1: Ascend/GPU/CPU|Comparison Functions
|[mindspore.ops.scalar_cast](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/ops/mindspore.ops.scalar_cast.html#mindspore.ops.scalar_cast)|Changed|r2.2: Casts the input scalar to another type. => r2.3.0rc1: The interface is deprecated from version 2.3 and will be removed in a future version, please use int(x) or float(x) instead.|r2.2: Ascend/GPU/CPU => r2.3.0rc1: |Type Cast
