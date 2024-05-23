# mindspore.ops API Interface Change

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.ops` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.ops.embedding](https://mindspore.cn/docs/en/r2.3.0rc2/api_python/ops/mindspore.ops.embedding.html#mindspore.ops.embedding)|New|Retrieve the word embeddings in weight using indices specified in input.|r2.3.0rc2: Ascend|Neural Network
[mindspore.ops.group_norm](https://mindspore.cn/docs/en/r2.3.0rc2/api_python/ops/mindspore.ops.group_norm.html#mindspore.ops.group_norm)|New|Group Normalization over a mini-batch of inputs.|r2.3.0rc2: Ascend/GPU/CPU|Neural Network
[mindspore.ops.layer_norm](https://mindspore.cn/docs/en/r2.3.0rc2/api_python/ops/mindspore.ops.layer_norm.html#mindspore.ops.layer_norm)|New|Applies the Layer Normalization to the input tensor.|r2.3.0rc2: Ascend|Neural Network
