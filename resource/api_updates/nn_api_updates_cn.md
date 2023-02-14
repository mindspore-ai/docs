# mindspore.nn API接口变更

与上一版本相比，MindSpore中`mindspore.nn`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.nn.ProximalAdagrad](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.ProximalAdagrad.html#mindspore.nn.ProximalAdagrad)|修改|ProximalAdagrad算法的实现。|r2.0.0-alpha: Ascend/GPU/CPU => master: Ascend/GPU|优化器
[mindspore.nn.ReflectionPad3d](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.ReflectionPad3d.html#mindspore.nn.ReflectionPad3d)|新增|根据 padding 对输入 x 进行填充。|master: Ascend/GPU/CPU|填充层
[mindspore.nn.WithGradCell](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.WithGradCell.html#mindspore.nn.WithGradCell)|删除|Cell that returns the gradients.|Ascend/GPU/CPU|封装层
[mindspore.nn.Identity](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Identity.html#mindspore.nn.Identity)|新增|返回与输入具有相同shape和值的Tensor。|master: Ascend/GPU/CPU|工具
[mindspore.nn.Unflatten](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Unflatten.html#mindspore.nn.Unflatten)|新增|根据 axis 和 unflattened_size 折叠指定维度为给定形状。|master: Ascend/GPU/CPU|工具
[mindspore.nn.CTCLoss](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.CTCLoss.html#mindspore.nn.CTCLoss)|修改|CTCLoss损失函数。|r2.0.0-alpha: Ascend/CPU => master: Ascend/GPU/CPU|损失函数
[mindspore.nn.MultiLabelSoftMarginLoss](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.MultiLabelSoftMarginLoss.html#mindspore.nn.MultiLabelSoftMarginLoss)|新增|基于最大熵计算用于多标签优化的损失。|master: Ascend/GPU/CPU|损失函数
[mindspore.nn.PoissonNLLLoss](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.PoissonNLLLoss.html#mindspore.nn.PoissonNLLLoss)|新增|计算泊松负对数似然损失。|r2.0.0-alpha: master: Ascend/GPU/CPU|损失函数
[mindspore.nn.SoftMarginLoss](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.SoftMarginLoss.html#mindspore.nn.SoftMarginLoss)|修改|针对二分类问题的损失函数。|r2.0.0-alpha: Ascend/GPU => master: Ascend|损失函数
[mindspore.nn.TripletMarginLoss](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.TripletMarginLoss.html#mindspore.nn.TripletMarginLoss)|新增|执行三元组损失函数的操作。|master: GPU|损失函数
