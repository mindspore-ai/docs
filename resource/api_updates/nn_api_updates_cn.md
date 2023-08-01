# mindspore.nn API接口变更

与上一版本相比，MindSpore中`mindspore.nn`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
|[mindspore.nn.LRScheduler](https://mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.LRScheduler.html#mindspore.nn.LRScheduler)|New|动态学习率的基类。|r2.1: Ascend/GPU/CPU|LRScheduler类
|[mindspore.nn.LinearLR](https://mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.LinearLR.html#mindspore.nn.LinearLR)|New|线性改变用于衰减参数组学习率的乘法因子，直到 last_epoch 数达到预定义的阈值 total_iters。|r2.1: Ascend/GPU/CPU|LRScheduler类
|[mindspore.nn.StepLR](https://mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.StepLR.html#mindspore.nn.StepLR)|New|每 step_size 个epoch按 gamma 衰减每个参数组的学习率。|r2.1: Ascend/GPU/CPU|LRScheduler类
|[mindspore.nn.optim_ex.Adam](https://mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.optim_ex.Adam.html#mindspore.nn.optim_ex.Adam)|New|Adaptive Moment Estimation (Adam)算法的实现。|r2.1: Ascend/GPU/CPU|实验性优化器
|[mindspore.nn.optim_ex.AdamW](https://mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.optim_ex.AdamW.html#mindspore.nn.optim_ex.AdamW)|New|Adaptive Moment Estimation Weight Decay(AdamW)算法的实现。|r2.1: Ascend/GPU/CPU|实验性优化器
|[mindspore.nn.optim_ex.Optimizer](https://mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.optim_ex.Optimizer.html#mindspore.nn.optim_ex.Optimizer)|New|用于参数更新的优化器基类。|r2.1: Ascend/GPU/CPU|实验性优化器
|[mindspore.nn.optim_ex.SGD](https://mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.optim_ex.SGD.html#mindspore.nn.optim_ex.SGD)|New|随机梯度下降算法。|r2.1: Ascend/GPU/CPU|实验性优化器
|[mindspore.nn.AdaptiveAvgPool2d](https://mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.AdaptiveAvgPool2d.html#mindspore.nn.AdaptiveAvgPool2d)|Changed|对输入Tensor，提供二维的自适应平均池化操作。|r2.0: GPU => r2.1: Ascend/GPU/CPU|池化层
