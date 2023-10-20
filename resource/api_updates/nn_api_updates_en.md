# mindspore.nn API Interface Change

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.nn` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.nn.CellDict](https://mindspore.cn/docs/en/r2.2/api_python/nn/mindspore.nn.CellDict.html#mindspore.nn.CellDict)|New|Holds Cells in a dictionary.|r2.2: Ascend/GPU/CPU|Container
[mindspore.nn.optim_ex.Adam](https://mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.optim_ex.Adam.html#mindspore.nn.optim_ex.Adam)|Deleted|Implements Adam algorithm..|Ascend/GPU/CPU|Experimental Optimizer
[mindspore.nn.optim_ex.AdamW](https://mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.optim_ex.AdamW.html#mindspore.nn.optim_ex.AdamW)|Deleted|Implements Adam Weight Decay algorithm.|Ascend/GPU/CPU|Experimental Optimizer
[mindspore.nn.optim_ex.Optimizer](https://mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.optim_ex.Optimizer.html#mindspore.nn.optim_ex.Optimizer)|Deleted|Base class for all optimizers.|Ascend/GPU/CPU|Experimental Optimizer
[mindspore.nn.optim_ex.SGD](https://mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.optim_ex.SGD.html#mindspore.nn.optim_ex.SGD)|Deleted|Stochastic Gradient Descent optimizer.|Ascend/GPU/CPU|Experimental Optimizer
[mindspore.nn.LRScheduler](https://mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.LRScheduler.html#mindspore.nn.LRScheduler)|Deleted|Basic class of learning rate schedule.|Ascend/GPU/CPU|LRScheduler Class
[mindspore.nn.LinearLR](https://mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.LinearLR.html#mindspore.nn.LinearLR)|Deleted|Decays the learning rate of each parameter group by linearly changing small multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.|Ascend/GPU/CPU|LRScheduler Class
[mindspore.nn.StepLR](https://mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.StepLR.html#mindspore.nn.StepLR)|Deleted|Decays the learning rate of each parameter group by gamma every step_size epochs.|Ascend/GPU/CPU|LRScheduler Class
[mindspore.nn.LRN](https://mindspore.cn/docs/en/r2.2/api_python/nn/mindspore.nn.LRN.html#mindspore.nn.LRN)|Changed|Local Response Normalization.|r2.1: Ascend/GPU/CPU => r2.2: GPU/CPU|Nonlinear Activation Layer
