# mindspore.nn API接口变更

与上一版本相比，MindSpore中`mindspore.nn`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.nn.WarmUpLR](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.WarmUpLR.html#mindspore.nn.WarmUpLR)|Changed|预热学习率。|r2.0.0-alpha: Ascend/GPU => r2.0: Ascend/GPU/CPU|LearningRateSchedule类
[mindspore.nn.MultiheadAttention](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.MultiheadAttention.html#mindspore.nn.MultiheadAttention)|New|论文 Attention Is All You Need 中所述的多头注意力的实现。|r2.0: Ascend/GPU/CPU|Transformer层
[mindspore.nn.Transformer](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.Transformer.html#mindspore.nn.Transformer)|New|Transformer模块，包括编码器和解码器。|r2.0: Ascend/GPU/CPU|Transformer层
[mindspore.nn.TransformerDecoder](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.TransformerDecoder.html#mindspore.nn.TransformerDecoder)|New|Transformer的解码器。|r2.0: Ascend/GPU/CPU|Transformer层
[mindspore.nn.TransformerDecoderLayer](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.TransformerDecoderLayer.html#mindspore.nn.TransformerDecoderLayer)|New|Transformer的解码器层。|r2.0.0-alpha: r2.0.0-alpha: r2.0.0-alpha: r2.0: Ascend/GPU/CPU|Transformer层
[mindspore.nn.TransformerEncoder](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.TransformerEncoder.html#mindspore.nn.TransformerEncoder)|New|Transformer编码器模块，多层 TransformerEncoderLayer 的堆叠，包括MultiheadAttention层和FeedForward层。|r2.0: Ascend/GPU/CPU|Transformer层
[mindspore.nn.TransformerEncoderLayer](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.TransformerEncoderLayer.html#mindspore.nn.TransformerEncoderLayer)|New|Transformer的编码器层。|r2.0: Ascend/GPU/CPU|Transformer层
[mindspore.nn.ProximalAdagrad](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.ProximalAdagrad.html#mindspore.nn.ProximalAdagrad)|Changed|ProximalAdagrad算法的实现。|r2.0.0-alpha: Ascend/GPU/CPU => r2.0: Ascend/GPU|优化器
[mindspore.nn.ResizeBilinear](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.ResizeBilinear.html#mindspore.nn.ResizeBilinear)|Changed|r2.0.0-alpha: 使用双线性插值调整输入Tensor为指定的大小。 => r2.0: nn.ResizeBilinear 从2.0版本开始已被弃用，并将在未来版本中被移除，建议使用 mindspore.ops.ResizeBilinearV2 或 mindspore.ops.interpolate 代替。|r2.0.0-alpha: Ascend/CPU/GPU => r2.0: |图像处理层
[mindspore.nn.Upsample](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.Upsample.html#mindspore.nn.Upsample)|New|详情请参考 mindspore.ops.interpolate() 。|r2.0: Ascend/GPU/CPU|图像处理层
[mindspore.nn.ReflectionPad3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.ReflectionPad3d.html#mindspore.nn.ReflectionPad3d)|New|使用反射的方式，以 input 的边界为对称轴，对 input 进行填充。|r2.0: Ascend/GPU/CPU|填充层
[mindspore.nn.WithGradCell](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.WithGradCell.html#mindspore.nn.WithGradCell)|Deleted|Cell that returns the gradients.|Ascend/GPU/CPU|封装层
[mindspore.nn.Identity](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.Identity.html#mindspore.nn.Identity)|New|网络占位符，返回与输入完全一致。|r2.0: Ascend/GPU/CPU|工具
[mindspore.nn.Unflatten](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.Unflatten.html#mindspore.nn.Unflatten)|New|根据 axis 和 unflattened_size 折叠指定维度为给定形状。|r2.0: Ascend/GPU/CPU|工具
[mindspore.nn.CTCLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.CTCLoss.html#mindspore.nn.CTCLoss)|Changed|CTCLoss损失函数。|r2.0.0-alpha: Ascend/CPU => r2.0: Ascend/GPU/CPU|损失函数
[mindspore.nn.MultiLabelSoftMarginLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.MultiLabelSoftMarginLoss.html#mindspore.nn.MultiLabelSoftMarginLoss)|New|基于最大熵计算用于多标签优化的损失。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.nn.MultiMarginLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.MultiMarginLoss.html#mindspore.nn.MultiMarginLoss)|New|多分类场景下用于计算 \(x\) 和 \(y\) 之间的合页损失（Hinge Loss），其中 x 为一个2-D Tensor，y 为一个表示类别索引的1-D Tensor， \(0 \leq y \leq \text{x.size}(1)-1\)。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.nn.MultilabelMarginLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.MultilabelMarginLoss.html#mindspore.nn.MultilabelMarginLoss)|New|创建一个损失函数，用于最小化多分类任务的合页损失。|r2.0: Ascend/GPU|损失函数
[mindspore.nn.PoissonNLLLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.PoissonNLLLoss.html#mindspore.nn.PoissonNLLLoss)|New|计算泊松负对数似然损失。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.nn.TripletMarginLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.TripletMarginLoss.html#mindspore.nn.TripletMarginLoss)|New|执行三元组损失函数的操作。|r2.0: GPU|损失函数
[mindspore.nn.Moments](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.Moments.html#mindspore.nn.Moments)|Deleted|沿指定轴 axis 计算输入 x 的均值和方差。|Ascend/GPU/CPU|数学运算
[mindspore.nn.AdaptiveAvgPool3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.AdaptiveAvgPool3d.html#mindspore.nn.AdaptiveAvgPool3d)|Changed|对输入Tensor，提供三维的自适应平均池化操作。|r2.0.0-alpha: GPU => r2.0: Ascend/GPU/CPU|池化层
[mindspore.nn.AvgPool3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.AvgPool3d.html#mindspore.nn.AvgPool3d)|Changed|在一个输入Tensor上应用3D平均池化运算，输入Tensor可看成是由一系列3D平面组成的。|r2.0.0-alpha: Ascend/CPU => r2.0: Ascend/GPU/CPU|池化层
[mindspore.nn.FractionalMaxPool2d](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.FractionalMaxPool2d.html#mindspore.nn.FractionalMaxPool2d)|Deleted|对多个输入平面组成的输入上应用2D分数最大池化。|CPU|池化层
[mindspore.nn.MaxPool3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.MaxPool3d.html#mindspore.nn.MaxPool3d)|Changed|在一个输入Tensor上应用3D最大池化运算，输入Tensor可看成是由一系列3D平面组成的。|r2.0.0-alpha: GPU => r2.0: Ascend/GPU/CPU|池化层
