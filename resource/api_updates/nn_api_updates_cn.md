# mindspore.nn API接口变更

与上一版本相比，MindSpore中`mindspore.nn`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.nn.ResizeBilinear](https://mindspore.cn/docs/zh-CN/r2.2/api_python/nn/mindspore.nn.ResizeBilinear.html#mindspore.nn.ResizeBilinear)|Deleted|nn.ResizeBilinear 从2.0版本开始已被弃用，并将在未来版本中被移除，建议使用 mindspore.ops.ResizeBilinearV2 或 mindspore.ops.interpolate 代替。||图像处理层
[mindspore.nn.PipelineGradReducer](https://mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/nn/mindspore.nn.PipelineGradReducer.html#mindspore.nn.PipelineGradReducer)|New|用于流水线并行的梯度聚合。|r2.2: r2.3.0rc1: Ascend/GPU|封装层
[mindspore.nn.MAELoss](https://mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/nn/mindspore.nn.MAELoss.html#mindspore.nn.MAELoss)|New|衡量 \(x\) 和 \(y\) 之间的平均绝对误差。|r2.3.0rc1: Ascend/GPU/CPU|损失函数
