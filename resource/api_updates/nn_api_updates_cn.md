# mindspore.nn API接口变更

与上一版本相比，MindSpore中`mindspore.nn`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.nn.LRScheduler](https://mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.LRScheduler.html#mindspore.nn.LRScheduler)|Deleted|动态学习率的基类。|Ascend/GPU/CPU|LRScheduler类
[mindspore.nn.LinearLR](https://mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.LinearLR.html#mindspore.nn.LinearLR)|Deleted|线性改变用于衰减参数组学习率的乘法因子，直到 last_epoch 数达到预定义的阈值 total_iters。|Ascend/GPU/CPU|LRScheduler类
[mindspore.nn.StepLR](https://mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.StepLR.html#mindspore.nn.StepLR)|Deleted|每 step_size 个epoch按 gamma 衰减每个参数组的学习率。|Ascend/GPU/CPU|LRScheduler类
[mindspore.nn.optim_ex.Adam](https://mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.optim_ex.Adam.html#mindspore.nn.optim_ex.Adam)|Deleted|Adaptive Moment Estimation (Adam)算法的实现。|Ascend/GPU/CPU|实验性优化器
[mindspore.nn.optim_ex.AdamW](https://mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.optim_ex.AdamW.html#mindspore.nn.optim_ex.AdamW)|Deleted|Adaptive Moment Estimation Weight Decay(AdamW)算法的实现。|Ascend/GPU/CPU|实验性优化器
[mindspore.nn.optim_ex.Optimizer](https://mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.optim_ex.Optimizer.html#mindspore.nn.optim_ex.Optimizer)|Deleted|用于参数更新的优化器基类。|Ascend/GPU/CPU|实验性优化器
[mindspore.nn.optim_ex.SGD](https://mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.optim_ex.SGD.html#mindspore.nn.optim_ex.SGD)|Deleted|随机梯度下降算法。|Ascend/GPU/CPU|实验性优化器
[mindspore.nn.CellDict](https://mindspore.cn/docs/zh-CN/r2.2/api_python/nn/mindspore.nn.CellDict.html#mindspore.nn.CellDict)|New|构造Cell字典。|r2.2: Ascend/GPU/CPU|容器
[mindspore.nn.LRN](https://mindspore.cn/docs/zh-CN/r2.2/api_python/nn/mindspore.nn.LRN.html#mindspore.nn.LRN)|Changed|局部响应归一化操作LRN(Local Response Normalization)。|r2.1: Ascend/GPU/CPU => r2.2: GPU/CPU|非线性激活函数层
