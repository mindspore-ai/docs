# mindspore.ops.primitive API Interface Change

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.ops.primitive` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.ops.Cummax](https://mindspore.cn/docs/en/r2.4.0/api_python/ops/mindspore.ops.Cummax.html#mindspore.ops.Cummax)|Changed|Refer to mindspore.ops.cummax() for more details.|r2.3.1: GPU/CPU => r2.4.0: Ascend/GPU/CPU|Array Operation
[mindspore.ops.UniqueWithPad](https://mindspore.cn/docs/en/r2.4.0/api_python/ops/mindspore.ops.UniqueWithPad.html#mindspore.ops.UniqueWithPad)|Deprecated|r2.3.1: Returns unique elements and relative indexes in 1-D tensor, filled with padding num. => r2.4.0: 'ops.UniqueWithPad' is deprecated from version 2.4 and will be removed in a future version.|r2.3.1: Ascend/GPU/CPU => r2.4.0: Deprecated|Array Operation
[mindspore.ops.silent_check.ASDBase](https://mindspore.cn/docs/en/r2.3.1/api_python/ops/mindspore.ops.silent_check.ASDBase.html#mindspore.ops.silent_check.ASDBase)|Deleted|ASDBase is the base class of operator with feature value detection in python.|r2.3.1: Ascend|Feature Value Detection
[mindspore.ops.ForiLoop](https://mindspore.cn/docs/en/r2.4.0/api_python/ops/mindspore.ops.ForiLoop.html#mindspore.ops.ForiLoop)|New|Provide a useful op for loop from lower to upper.|r2.4.0: Ascend/GPU/CPU|operations--Frame Operators
[mindspore.ops.Scan](https://mindspore.cn/docs/en/r2.4.0/api_python/ops/mindspore.ops.Scan.html#mindspore.ops.Scan)|New|Scan a function over an array while the processing of the current element depends on the execution result of the previous element.|r2.4.0: Ascend/GPU/CPU|operations--Frame Operators
[mindspore.ops.WhileLoop](https://mindspore.cn/docs/en/r2.4.0/api_python/ops/mindspore.ops.WhileLoop.html#mindspore.ops.WhileLoop)|New|Provide a useful op for reducing compilation times of while loop.|r2.4.0: Ascend/GPU/CPU|operations--Frame Operators
