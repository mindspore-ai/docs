# API Updates

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.nn` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
|[mindspore.nn.AdaSumByDeltaWeightWrapCell](https://www.mindspore.cn/docs/en/r1.7/api_python/nn/mindspore.nn.AdaSumByDeltaWeightWrapCell.html)|New|Enable the adasum in “auto_parallel/semi_auto_parallel” mode.|r1.7: Ascend/GPU|Optimizer
|[mindspore.nn.AdaSumByGradWrapCell](https://www.mindspore.cn/docs/en/r1.7/api_python/nn/mindspore.nn.AdaSumByGradWrapCell.html)|New|Enable the adasum in “auto_parallel/semi_auto_parallel” mode.|r1.7: Ascend/GPU|Optimizer
|mindspore.nn.ActQuant|Deleted|Quantization aware training activation function.|Ascend/GPU|Quantized Functions
|mindspore.nn.Conv2dBnAct|Deleted|A combination of convolution, Batchnorm, and activation layer.|Ascend/GPU/CPU|Quantized Functions
|mindspore.nn.Conv2dBnFoldQuant|Deleted|2D convolution with Batch Normalization operation folded construct.|Ascend/GPU|Quantized Functions
|mindspore.nn.Conv2dBnFoldQuantOneConv|Deleted|2D convolution which use the convolution layer statistics once to calculate Batch Normalization operation folded construct.|Ascend/GPU|Quantized Functions
|mindspore.nn.Conv2dBnWithoutFoldQuant|Deleted|2D convolution and batchnorm without fold with fake quantized construct.|Ascend/GPU|Quantized Functions
|mindspore.nn.Conv2dQuant|Deleted|2D convolution with fake quantized operation layer.|Ascend/GPU|Quantized Functions
|mindspore.nn.DenseBnAct|Deleted|A combination of Dense, Batchnorm, and the activation layer.|Ascend/GPU/CPU|Quantized Functions
|mindspore.nn.DenseQuant|Deleted|The fully connected layer with fake quantized operation.|Ascend/GPU|Quantized Functions
|mindspore.nn.FakeQuantWithMinMaxObserver|Deleted|Quantization aware operation which provides the fake quantization observer function on data with min and max.|Ascend/GPU|Quantized Functions
|mindspore.nn.MulQuant|Deleted|Adds fake quantized operation after  Mul  operation.|Ascend/GPU|Quantized Functions
|mindspore.nn.TensorAddQuant|Deleted|Adds fake quantized operation after TensorAdd operation.|Ascend/GPU|Quantized Functions
|[mindspore.nn.Conv3d](https://www.mindspore.cn/docs/en/r1.7/api_python/nn/mindspore.nn.Conv3d.html)| Changed |3D convolution layer.|r1.6: Ascend/GPU => r1.7: Ascend/GPU/CPU|Convolutional Neural Network Layer
|[mindspore.nn.BCEWithLogitsLoss](https://www.mindspore.cn/docs/en/r1.7/api_python/nn/mindspore.nn.BCEWithLogitsLoss.html)| Changed  platform-diff |Adds sigmoid activation function to input logits, and uses the given logits to compute binary cross entropy between the logits and the labels.|r1.6: Ascend/GPU => r1.7: Ascend/GPU/CPU|Loss Function
|[mindspore.nn.Adam](https://www.mindspore.cn/docs/en/r1.7/api_python/nn/mindspore.nn.Adam.html)| Changed  platform-diff |Implements the Adaptive Moment Estimation (Adam) algorithm.|r1.6: GPU/  /CPU => r1.7: Ascend/GPU/CPU|Optimizer