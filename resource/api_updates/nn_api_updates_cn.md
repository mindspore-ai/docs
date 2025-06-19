# mindspore.nn API接口变更

与2.5.0版本相比，MindSpore中`mindspore.nn`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.nn.MicroBatchInterleaved](https://mindspore.cn/docs/zh-CN/r2.5.0/api_python/nn/mindspore.nn.MicroBatchInterleaved.html#mindspore.nn.MicroBatchInterleaved)|Deleted|这个函数的作用是将输入在第零维度拆成 interleave_num 份，然后执行包裹的cell的计算。|r2.5.0: Ascend/GPU|封装层
[mindspore.nn.PipelineGradReducer](https://mindspore.cn/docs/zh-CN/r2.5.0/api_python/nn/mindspore.nn.PipelineGradReducer.html#mindspore.nn.PipelineGradReducer)|Deleted|用于流水线并行的梯度聚合。|r2.5.0: Ascend/GPU|封装层
