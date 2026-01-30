# mindspore.nn API接口变更

2.8.0版本与2.7.2版本相比，MindSpore中`mindspore.nn`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.nn.LazyAdam](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/nn/mindspore.nn.LazyAdam.html#mindspore.nn.LazyAdam)|Changed|r2.7.2: Adaptive Moment Estimation (Adam)算法的实现。 => r2.8.0: nn.LazyAdam 从2.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend/GPU/CPU => r2.8.0: Deprecated|优化器
[mindspore.nn.Unfold](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/nn/mindspore.nn.Unfold.html#mindspore.nn.Unfold)|Changed|r2.7.2: 从图像中提取滑窗的区域块。 => r2.8.0: nn.Unfold 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend/GPU => r2.8.0: Deprecated|卷积神经网络层
[mindspore.nn.TimeDistributed](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/nn/mindspore.nn.TimeDistributed.html#mindspore.nn.TimeDistributed)|Changed|r2.7.2: 时间序列封装层。 => r2.8.0: nn.TimeDistributed 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend/GPU/CPU => r2.8.0: Deprecated|封装层
[mindspore.nn.DiceLoss](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/nn/mindspore.nn.DiceLoss.html#mindspore.nn.DiceLoss)|Changed|r2.7.2: Dice系数是一个集合相似性loss，用于计算两个样本之间的相似性。 => r2.8.0: nn.DiceLoss 从2.8版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend/GPU/CPU => r2.8.0: Deprecated|损失函数
[mindspore.nn.FocalLoss](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/nn/mindspore.nn.FocalLoss.html#mindspore.nn.FocalLoss)|Changed|r2.7.2: FocalLoss函数解决了类别不平衡的问题。 => r2.8.0: nn.FocalLoss 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend => r2.8.0: Deprecated|损失函数
[mindspore.nn.MultiClassDiceLoss](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/nn/mindspore.nn.MultiClassDiceLoss.html#mindspore.nn.MultiClassDiceLoss)|Changed|r2.7.2: 对于多标签问题，可以将标签通过one-hot编码转换为多个二分类标签。 => r2.8.0: nn.MultiClassDiceLoss 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend/GPU/CPU => r2.8.0: Deprecated|损失函数
[mindspore.nn.SampledSoftmaxLoss](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/nn/mindspore.nn.SampledSoftmaxLoss.html#mindspore.nn.SampledSoftmaxLoss)|Changed|r2.7.2: 抽样交叉熵损失函数。 => r2.8.0: nn.SampledSoftmaxLoss 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: GPU => r2.8.0: Deprecated|损失函数
