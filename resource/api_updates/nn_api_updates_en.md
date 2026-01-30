# mindspore.nn API Interface Change

Compared with the version 2.7.2, the added, deleted and supported platforms change information of `mindspore.nn` operators in version 2.8.0, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.nn.Unfold](https://mindspore.cn/docs/en/r2.8.0/api_python/nn/mindspore.nn.Unfold.html#mindspore.nn.Unfold)|Changed|r2.7.2: Extracts patches from images. => r2.8.0: nn.Unfold is deprecated from version 2.8.0 and will be removed in a future version, please use mindspore.ops.unfold() instead.|r2.7.2: Ascend/GPU => r2.8.0: Deprecated|Convolutional Layer
[mindspore.nn.DiceLoss](https://mindspore.cn/docs/en/r2.8.0/api_python/nn/mindspore.nn.DiceLoss.html#mindspore.nn.DiceLoss)|Changed|r2.7.2: The Dice coefficient is a set similarity loss, which is used to calculate the similarity between two samples. => r2.8.0: nn.DiceLoss is deprecated from version 2.8.0 and will be removed in a future version.|r2.7.2: Ascend/GPU/CPU => r2.8.0: Deprecated|Loss Function
[mindspore.nn.FocalLoss](https://mindspore.cn/docs/en/r2.8.0/api_python/nn/mindspore.nn.FocalLoss.html#mindspore.nn.FocalLoss)|Changed|r2.7.2: It is a loss function to solve the imbalance of categories and the difference of classification difficulty. => r2.8.0: nn.FocalLoss is deprecated from version 2.8.0 and will be removed in a future version.|r2.7.2: Ascend => r2.8.0: Deprecated|Loss Function
[mindspore.nn.MultiClassDiceLoss](https://mindspore.cn/docs/en/r2.8.0/api_python/nn/mindspore.nn.MultiClassDiceLoss.html#mindspore.nn.MultiClassDiceLoss)|Changed|r2.7.2: When there are multiple classifications, label is transformed into multiple binary classifications by one hot. => r2.8.0: nn.MultiClassDiceLoss is deprecated from version 2.8.0 and will be removed in a future version.|r2.7.2: Ascend/GPU/CPU => r2.8.0: Deprecated|Loss Function
[mindspore.nn.SampledSoftmaxLoss](https://mindspore.cn/docs/en/r2.8.0/api_python/nn/mindspore.nn.SampledSoftmaxLoss.html#mindspore.nn.SampledSoftmaxLoss)|Changed|r2.7.2: Computes the sampled softmax training loss. => r2.8.0: nn.SampledSoftmaxLoss is deprecated from version 2.8.0 and will be removed in a future version.|r2.7.2: GPU => r2.8.0: Deprecated|Loss Function
[mindspore.nn.LazyAdam](https://mindspore.cn/docs/en/r2.8.0/api_python/nn/mindspore.nn.LazyAdam.html#mindspore.nn.LazyAdam)|Changed|r2.7.2: Implements the Adaptive Moment Estimation (Adam) algorithm. => r2.8.0: nn.LazyAdam is deprecated from version 2.0 and will be removed in a future version, please use mindspore.nn.Adam instead.|r2.7.2: Ascend/GPU/CPU => r2.8.0: Deprecated|Optimizer
[mindspore.nn.TimeDistributed](https://mindspore.cn/docs/en/r2.8.0/api_python/nn/mindspore.nn.TimeDistributed.html#mindspore.nn.TimeDistributed)|Changed|r2.7.2: The time distributed layer. => r2.8.0: nn.TimeDistributed is deprecated from version 2.8.0 and will be removed in a future version.|r2.7.2: Ascend/GPU/CPU => r2.8.0: Deprecated|Wrapper Layer

Compared with the version 2.7.1, the information of `mindspore.nn` operators in MindSpore has no changes in version 2.7.2.

Compared with the version 2.6.0, the information of `mindspore.nn` operators in MindSpore has no changes in version 2.7.0.

Compared with the version 2.6.0, the information of `mindspore.nn` operators in MindSpore in version 2.7.0 has no changes.
