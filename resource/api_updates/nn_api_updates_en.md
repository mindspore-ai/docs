# mindspore.nn API Interface Change

Compared with the previous version 2.5.0, the added, deleted and supported platforms change information of `mindspore.nn` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.nn.MicroBatchInterleaved](https://mindspore.cn/docs/en/r2.5.0/api_python/nn/mindspore.nn.MicroBatchInterleaved.html#mindspore.nn.MicroBatchInterleaved)|Deleted|This function splits the input at the 0th into interleave_num pieces and then performs the computation of the wrapped cell.|r2.5.0: Ascend/GPU|Wrapper Layer
[mindspore.nn.PipelineGradReducer](https://mindspore.cn/docs/en/r2.5.0/api_python/nn/mindspore.nn.PipelineGradReducer.html#mindspore.nn.PipelineGradReducer)|Deleted|PipelineGradReducer is a gradient reducer for pipeline parallelism.|r2.5.0: Ascend/GPU|Wrapper Layer
