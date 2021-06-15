# API Updates

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.nn` operators in MindSpore, is shown in the following table.

|API|Status|Support Platform|Class
|:----|:----|:----|:----
|[mindspore.nn.ForwardValueAndGrad](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.ForwardValueAndGrad.html#mindspore.nn.ForwardValueAndGrad)|New|r1.2: Ascend/GPU/CPU|Wrapper Functions
|[mindspore.nn.TimeDistributed](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.TimeDistributed.html#mindspore.nn.TimeDistributed)|New|r1.2: Ascend/GPU/CPU|Wrapper Functions
|[mindspore.nn.SparseToDense](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.SparseToDense.html#mindspore.nn.SparseToDense)|New|r1.2: CPU|Utilities
|[mindspore.nn.BatchNorm3d](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.BatchNorm3d.html#mindspore.nn.BatchNorm3d)|New|r1.2: Ascend/GPU/CPU|Normalization Layers
|[mindspore.nn.InstanceNorm2d](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.InstanceNorm2d.html#mindspore.nn.InstanceNorm2d)|New|r1.2: GPU|Normalization Layers
|[mindspore.nn.SyncBatchNorm](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.SyncBatchNorm.html#mindspore.nn.SyncBatchNorm)|New|r1.2: Ascend|Normalization Layers
|[mindspore.nn.BCEWithLogitsLoss](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.BCEWithLogitsLoss.html#mindspore.nn.BCEWithLogitsLoss)|New|r1.2: Ascend|Loss Functions
|[mindspore.nn.DiceLoss](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.DiceLoss.html#mindspore.nn.DiceLoss)|New|r1.2: Ascend/GPU/CPU|Loss Functions
|[mindspore.nn.FocalLoss](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.FocalLoss.html#mindspore.nn.FocalLoss)|New|r1.2: Ascend/GPU|Loss Functions
|[mindspore.nn.MAELoss](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.MAELoss.html#mindspore.nn.MAELoss)|New|r1.2: Ascend/GPU/CPU|Loss Functions
|[mindspore.nn.MultiClassDiceLoss](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.MultiClassDiceLoss.html#mindspore.nn.MultiClassDiceLoss)|New|r1.2: Ascend/GPU|Loss Functions
|[mindspore.nn.RMSELoss](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.RMSELoss.html#mindspore.nn.RMSELoss)|New|r1.2: Ascend/GPU/CPU|Loss Functions
|[mindspore.nn.Conv3d](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.Conv3d.html#mindspore.nn.Conv3d)|New|r1.2: Ascend|Convolution Layers
|[mindspore.nn.Conv3dTranspose](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.Conv3dTranspose.html#mindspore.nn.Conv3dTranspose)|New|r1.2: Ascend|Convolution Layers
|[mindspore.nn.Norm](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.Norm.html#mindspore.nn.Norm)|Changed|r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Utilities
|[mindspore.nn.ClipByNorm](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.ClipByNorm.html#mindspore.nn.ClipByNorm)|Changed|r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Utilities
|[mindspore.nn.Pad](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.Pad.html#mindspore.nn.Pad)|Changed|r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Utilities
|[mindspore.nn.ResizeBilinear](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.ResizeBilinear.html#mindspore.nn.ResizeBilinear)|Changed|r1.1: Ascend => r1.2: Ascend/CPU|Utilities
|[mindspore.nn.Range](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.Range.html#mindspore.nn.Range)|Changed|r1.1: Ascend => r1.2: Ascend/GPU/CPU|Utilities
|[mindspore.nn.FakeQuantWithMinMaxObserver](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.FakeQuantWithMinMaxObserver.html#mindspore.nn.FakeQuantWithMinMaxObserver)|Changed|r1.1: To Be Developed => r1.2: Ascend/GPU|Quantized Functions
|[mindspore.nn.Conv2dBnFoldQuantOneConv](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.Conv2dBnFoldQuantOneConv.html#mindspore.nn.Conv2dBnFoldQuantOneConv)|Changed|r1.1: To Be Developed => r1.2: Ascend/GPU|Quantized Functions
|[mindspore.nn.Conv2dBnAct](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.Conv2dBnAct.html#mindspore.nn.Conv2dBnAct)|Changed|r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Quantized Functions
|[mindspore.nn.DenseBnAct](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.DenseBnAct.html#mindspore.nn.DenseBnAct)|Changed|r1.1: Ascend => r1.2: Ascend/GPU|Quantized Functions
|[mindspore.nn.AvgPool1d](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.AvgPool1d.html#mindspore.nn.AvgPool1d)|Changed|r1.1: Ascend => r1.2: Ascend/GPU/CPU|Pooling layers
|[mindspore.nn.AvgPool2d](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.AvgPool2d.html#mindspore.nn.AvgPool2d)|Changed|r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Pooling layers
|[mindspore.nn.MaxPool1d](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.MaxPool1d.html#mindspore.nn.MaxPool1d)|Changed|r1.1: Ascend => r1.2: Ascend/GPU/CPU|Pooling layers
|[mindspore.nn.LazyAdam](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.LazyAdam.html#mindspore.nn.LazyAdam)|Changed|r1.1: Ascend => r1.2: Ascend/GPU|Optimizer Functions
|[mindspore.nn.RMSProp](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.RMSProp.html#mindspore.nn.RMSProp)|Changed|r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Optimizer Functions
|[mindspore.nn.GroupNorm](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.GroupNorm.html#mindspore.nn.GroupNorm)|Changed|r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Normalization Layers
|[mindspore.nn.BatchNorm1d](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.BatchNorm1d.html#mindspore.nn.BatchNorm1d)|Changed|r1.1: Ascend/GPU => r1.2: Ascend|Normalization Layers
|[mindspore.nn.LayerNorm](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.LayerNorm.html#mindspore.nn.LayerNorm)|Changed|r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Normalization Layers
|[mindspore.nn.HSigmoid](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.HSigmoid.html#mindspore.nn.HSigmoid)|Changed|r1.1: GPU => r1.2: GPU/CPU|Non-linear Activations
|[mindspore.nn.HSwish](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.HSwish.html#mindspore.nn.HSwish)|Changed|r1.1: GPU => r1.2: GPU/CPU|Non-linear Activations
|[mindspore.nn.LeakyReLU](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.LeakyReLU.html#mindspore.nn.LeakyReLU)|Changed|r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Non-linear Activations
|[mindspore.nn.GELU](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.GELU.html#mindspore.nn.GELU)|Changed|r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Non-linear Activations
|[mindspore.nn.ELU](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.ELU.html#mindspore.nn.ELU)|Changed|r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Non-linear Activations
|[mindspore.nn.get_activation](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.get_activation.html#mindspore.nn.get_activation)|Changed|r1.1: To Be Developed => r1.2: Ascend/GPU/CPU|Non-linear Activations
|[mindspore.nn.Moments](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.Moments.html#mindspore.nn.Moments)|Changed|r1.1: Ascend => r1.2: Ascend/GPU|Math Functions
|[mindspore.nn.BCELoss](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.BCELoss.html#mindspore.nn.BCELoss)|Changed|r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Loss Functions
|[mindspore.nn.L1Loss](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.L1Loss.html#mindspore.nn.L1Loss)|Changed|r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Loss Functions
|[mindspore.nn.MSELoss](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.MSELoss.html#mindspore.nn.MSELoss)|Changed|r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Loss Functions
|[mindspore.nn.ImageGradients](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.ImageGradients.html#mindspore.nn.ImageGradients)|Changed|r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Images Functions
|[mindspore.nn.Conv1d](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.Conv1d.html#mindspore.nn.Conv1d)|Changed|r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Convolution Layers
|[mindspore.nn.GraphKernel](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/nn/mindspore.nn.GraphKernel.html#mindspore.nn.GraphKernel)|Changed|r1.1: To Be Developed => r1.2: Ascend/GPU|Cell

>
