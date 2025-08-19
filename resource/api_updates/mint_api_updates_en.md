# mindspore.mint API Interface Change

Compared with the previous version 2.6.0, the added, deleted and supported platforms change information of `mindspore.mint` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.mint.nn.Threshold](https://mindspore.cn/docs/en/r2.7.0/api_python/mint/mindspore.mint.nn.Threshold.html#mindspore.mint.nn.Threshold)|New|Compute the Threshold activation function element-wise.|r2.7.0: Ascend|Non-linear Activations (weighted sum, nonlinearity)
[mindspore.mint.nn.functional.threshold](https://mindspore.cn/docs/en/r2.7.0/api_python/mint/mindspore.mint.nn.functional.threshold.html#mindspore.mint.nn.functional.threshold)|New|Compute the Threshold activation function element-wise.|r2.7.0: Ascend|Non-linear activation functions
[mindspore.mint.nn.functional.threshold_](https://mindspore.cn/docs/en/r2.7.0/api_python/mint/mindspore.mint.nn.functional.threshold_.html#mindspore.mint.nn.functional.threshold_)|New|Update the input tensor in-place by computing the Threshold activation function element-wise.|r2.7.0: Ascend|Non-linear activation functions
[mindspore.mint.floor_divide](https://mindspore.cn/docs/en/r2.7.0/api_python/mint/mindspore.mint.floor_divide.html#mindspore.mint.floor_divide)|New|Divides the first input tensor by the second input tensor element-wise and round down to the closest integer.|r2.7.0: Ascend|Pointwise Operations
[mindspore.mint.distributed.TCPStore](https://mindspore.cn/docs/en/r2.7.0/api_python/mint/mindspore.mint.distributed.TCPStore.html#mindspore.mint.distributed.TCPStore)|New|A TCP-based distributed key-value store implementation.|r2.7.0: Ascend|mindspore.mint.distributed
[mindspore.mint.distributed.all_gather_into_tensor_uneven](https://mindspore.cn/docs/en/r2.7.0/api_python/mint/mindspore.mint.distributed.all_gather_into_tensor_uneven.html#mindspore.mint.distributed.all_gather_into_tensor_uneven)|New|Gathers and concatenates tensors across devices with uneven first dimensions.|r2.7.0: Ascend|mindspore.mint.distributed
[mindspore.mint.distributed.reduce_scatter_tensor_uneven](https://mindspore.cn/docs/en/r2.7.0/api_python/mint/mindspore.mint.distributed.reduce_scatter_tensor_uneven.html#mindspore.mint.distributed.reduce_scatter_tensor_uneven)|New|Reduce tensors from the specified communication group and scatter to the output tensor according to input_split_sizes.|r2.7.0: Ascend|mindspore.mint.distributed
