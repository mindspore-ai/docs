# API Updates

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.nn` operators in MindSpore, is shown in the following table.

|API|Status|Support Platform|Class
|:----|:----|:----|:----
|mindspore.nn.Conv3dTranspose|New|r1.2: Ascend|Convolution Layers
|mindspore.nn.Conv3d|New|r1.2: Ascend|Convolution Layers
|mindspore.nn.RMSELoss|New|r1.2: Ascend/GPU/CPU|Loss Functions
|mindspore.nn.MultiClassDiceLoss|New|r1.2: Ascend/GPU|Loss Functions
|mindspore.nn.MAELoss|New|r1.2: Ascend/GPU/CPU|Loss Functions
|mindspore.nn.FocalLoss|New|r1.2: Ascend/GPU|Loss Functions
|mindspore.nn.DiceLoss|New|r1.2: Ascend/GPU/CPU|Loss Functions
|mindspore.nn.BCEWithLogitsLoss|New|r1.2: Ascend|Loss Functions
|mindspore.nn.SyncBatchNorm|New|r1.2: Ascend|Normalization Layers
|mindspore.nn.InstanceNorm2d|New|r1.2: GPU|Normalization Layers
|mindspore.nn.BatchNorm3d|New|r1.2: Ascend/GPU/CPU|Normalization Layers
|mindspore.nn.TimeDistributed|New|r1.2: Ascend/GPU/CPU|Wrapper Functions
|mindspore.nn.ForwardValueAndGrad|New|r1.2: Ascend/GPU/CPU|Wrapper Functions
|mindspore.nn.Conv1d| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Convolution Layers
|mindspore.nn.ImageGradients| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Images Functions
|mindspore.nn.MSELoss| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Loss Functions
|mindspore.nn.BCELoss| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Loss Functions
|mindspore.nn.L1Loss| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Loss Functions
|mindspore.nn.Moments| Changed |r1.1: Ascend => r1.2: Ascend/GPU|Math Functions
|mindspore.nn.HSigmoid| Changed |r1.1: GPU => r1.2: GPU/CPU|Non-linear Activations
|mindspore.nn.HSwish| Changed |r1.1: GPU => r1.2: GPU/CPU|Non-linear Activations
|mindspore.nn.LeakyReLU| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Non-linear Activations
|mindspore.nn.ELU| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Non-linear Activations
|mindspore.nn.GroupNorm| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Normalization Layers
|mindspore.nn.BatchNorm1d| Changed |r1.1: Ascend/GPU => r1.2: Ascend|Normalization Layers
|mindspore.nn.RMSProp| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Optimizer Functions
|mindspore.nn.LazyAdam| Changed |r1.1: Ascend => r1.2: Ascend/GPU|Optimizer Functions
|mindspore.nn.AvgPool2d| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Pooling layers
|mindspore.nn.AvgPool1d| Changed |r1.1: Ascend => r1.2: Ascend/GPU/CPU|Pooling layers
|mindspore.nn.MaxPool1d| Changed |r1.1: Ascend => r1.2: Ascend/GPU/CPU|Pooling layers
|mindspore.nn.Conv2dBnFoldQuantOneConv| Changed |r1.1: To Be Developed => r1.2: Ascend/GPU|Quantized Functions
|mindspore.nn.FakeQuantWithMinMaxObserver| Changed |r1.1: To Be Developed => r1.2: Ascend/GPU|Quantized Functions
|mindspore.nn.Conv2dBnAct| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Quantized Functions
|mindspore.nn.DenseBnAct| Changed |r1.1: Ascend => r1.2: Ascend/GPU|Quantized Functions
|mindspore.nn.ClipByNorm| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Utilities
|mindspore.nn.Norm| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Utilities
|mindspore.nn.Pad| Changed |r1.1: Ascend/GPU => r1.2: Ascend/GPU/CPU|Utilities
|mindspore.nn.Range| Changed |r1.1: Ascend => r1.2: Ascend/GPU/CPU|Utilities
|mindspore.nn.ResizeBilinear| Changed |r1.1: Ascend => r1.2: Ascend/CPU|Utilities

>
