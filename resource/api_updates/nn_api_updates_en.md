# mindspore.nn API Interface Change

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.nn` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.nn.ResizeBilinear](https://mindspore.cn/docs/en/r2.2/api_python/nn/mindspore.nn.ResizeBilinear.html#mindspore.nn.ResizeBilinear)|Deleted|'nn.ResizeBilinear' is deprecated from version 2.0 and will be removed in a future version, use mindspore.ops.ResizeBilinearV2 or mindspore.ops.interpolate() instead.||Image Processing Layer
[mindspore.nn.MAELoss](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/nn/mindspore.nn.MAELoss.html#mindspore.nn.MAELoss)|New|MAELoss creates a criterion to measure the average absolute error between \(x\) and \(y\) element-wise, where \(x\) is the input and \(y\) is the labels.|r2.3.0rc1: Ascend/GPU/CPU|Loss Function
[mindspore.nn.PipelineGradReducer](https://mindspore.cn/docs/en/r2.3.0rc1/api_python/nn/mindspore.nn.PipelineGradReducer.html#mindspore.nn.PipelineGradReducer)|New|PipelineGradReducer is a gradient reducer for pipeline parallelism.|r2.2: r2.3.0rc1: Ascend/GPU|Wrapper Layer
