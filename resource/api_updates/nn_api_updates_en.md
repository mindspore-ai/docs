# mindspore.nn API Interface Change

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.nn` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
|[mindspore.nn.optim_ex.Adam](https://mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.optim_ex.Adam.html#mindspore.nn.optim_ex.Adam)|New|Implements Adam algorithm..|r2.1: Ascend/GPU/CPU|Experimental Optimizer
|[mindspore.nn.optim_ex.AdamW](https://mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.optim_ex.AdamW.html#mindspore.nn.optim_ex.AdamW)|New|Implements Adam Weight Decay algorithm.|r2.1: Ascend/GPU/CPU|Experimental Optimizer
|[mindspore.nn.optim_ex.Optimizer](https://mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.optim_ex.Optimizer.html#mindspore.nn.optim_ex.Optimizer)|New|Base class for all optimizers.|r2.1: Ascend/GPU/CPU|Experimental Optimizer
|[mindspore.nn.optim_ex.SGD](https://mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.optim_ex.SGD.html#mindspore.nn.optim_ex.SGD)|New|Stochastic Gradient Descent optimizer.|r2.1: Ascend/GPU/CPU|Experimental Optimizer
|[mindspore.nn.LRScheduler](https://mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.LRScheduler.html#mindspore.nn.LRScheduler)|New|Basic class of learning rate schedule.|r2.1: Ascend/GPU/CPU|LRScheduler Class
|[mindspore.nn.LinearLR](https://mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.LinearLR.html#mindspore.nn.LinearLR)|New|Decays the learning rate of each parameter group by linearly changing small multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.|r2.1: Ascend/GPU/CPU|LRScheduler Class
|[mindspore.nn.StepLR](https://mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.StepLR.html#mindspore.nn.StepLR)|New|Decays the learning rate of each parameter group by gamma every step_size epochs.|r2.1: Ascend/GPU/CPU|LRScheduler Class
|[mindspore.nn.AdaptiveAvgPool2d](https://mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.AdaptiveAvgPool2d.html#mindspore.nn.AdaptiveAvgPool2d)|Changed|This operator applies a 2D adaptive average pooling to an input signal composed of multiple input planes.|r2.0: GPU => r2.1: Ascend/GPU/CPU|Pooling Layer
