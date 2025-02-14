# mindspore.nn API Interface Change

Compared with the previous version 2.4.10, the added, deleted and supported platforms change information of `mindspore.nn` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.nn.AdamOffload](https://mindspore.cn/docs/en/r2.4.10/api_python/nn/mindspore.nn.AdamOffload.html#mindspore.nn.AdamOffload)|Deleted|This optimizer will offload Adam optimizer to host CPU and keep parameters being updated on the device, to minimize the memory cost.|r2.4.10: Ascend/GPU/CPU|Optimizer
[mindspore.nn.thor](https://mindspore.cn/docs/en/r2.4.10/api_python/nn/mindspore.nn.thor.html#mindspore.nn.thor)|Deleted|Updates gradients by second-order algorithm--THOR.|r2.4.10: Ascend/GPU|Optimizer

