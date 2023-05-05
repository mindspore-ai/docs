# mindspore.nn API接口变更

与上一版本相比，MindSpore中`mindspore.nn`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.nn.Dropout1d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.Dropout1d.html#mindspore.nn.Dropout1d)|New|在训练期间，以服从伯努利分布的概率 p 随机将输入Tensor的某些通道归零（对于shape为 $(N, C, L)$ 的三维Tensor，其通道特征图指的是后一维 $L$ 的一维特征图）。|r2.0: Ascend/GPU/CPU|Dropout层
[mindspore.nn.WarmUpLR](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.WarmUpLR.html#mindspore.nn.WarmUpLR)|Changed|预热学习率。|r1.10: Ascend/GPU => r2.0: Ascend/GPU/CPU|LearningRateSchedule类
[mindspore.nn.MultiheadAttention](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.MultiheadAttention.html#mindspore.nn.MultiheadAttention)|New|论文 Attention Is All You Need 中所述的多头注意力的实现。|r2.0: Ascend/GPU/CPU|Transformer层
[mindspore.nn.Transformer](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.Transformer.html#mindspore.nn.Transformer)|New|Transformer模块，包括编码器和解码器。|r2.0: Ascend/GPU/CPU|Transformer层
[mindspore.nn.TransformerDecoder](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.TransformerDecoder.html#mindspore.nn.TransformerDecoder)|New|Transformer的解码器。|r2.0: Ascend/GPU/CPU|Transformer层
[mindspore.nn.TransformerDecoderLayer](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.TransformerDecoderLayer.html#mindspore.nn.TransformerDecoderLayer)|New|Transformer的解码器层。|r2.0: Ascend/GPU/CPU|Transformer层
[mindspore.nn.TransformerEncoder](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.TransformerEncoder.html#mindspore.nn.TransformerEncoder)|New|Transformer编码器模块，多层 TransformerEncoderLayer 的堆叠，包括MultiheadAttention层和FeedForward层。|r2.0: Ascend/GPU/CPU|Transformer层
[mindspore.nn.TransformerEncoderLayer](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.TransformerEncoderLayer.html#mindspore.nn.TransformerEncoderLayer)|New|Transformer的编码器层。|r2.0: Ascend/GPU/CPU|Transformer层
[mindspore.nn.CentralCrop](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.CentralCrop.html#mindspore.nn.CentralCrop)|Deleted|根据指定比例裁剪出图像的中心区域。|Ascend/GPU/CPU|图像处理层
[mindspore.nn.ImageGradients](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.ImageGradients.html#mindspore.nn.ImageGradients)|Deleted|计算每个颜色通道的图像渐变，返回为两个Tensor，分别表示高和宽方向上的变化率。|Ascend/GPU/CPU|图像处理层
[mindspore.nn.MSSSIM](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.MSSSIM.html#mindspore.nn.MSSSIM)|Deleted|多尺度计算两个图像之间的结构相似性（SSIM）。|Ascend/GPU|图像处理层
[mindspore.nn.PSNR](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.PSNR.html#mindspore.nn.PSNR)|Deleted|在批处理中计算两个图像的峰值信噪比（PSNR）。|Ascend/GPU/CPU|图像处理层
[mindspore.nn.PixelShuffle](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.PixelShuffle.html#mindspore.nn.PixelShuffle)|New|对 input 应用像素重组操作，它实现了步长为 $1/r$ 的子像素卷积。|r2.0: Ascend/GPU/CPU|图像处理层
[mindspore.nn.PixelUnshuffle](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.PixelUnshuffle.html#mindspore.nn.PixelUnshuffle)|New|对 input 应用逆像素重组操作，这是像素重组的逆操作。|r2.0: Ascend/GPU/CPU|图像处理层
[mindspore.nn.ResizeBilinear](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.ResizeBilinear.html#mindspore.nn.ResizeBilinear)|Changed|r1.10: 使用双线性插值调整输入Tensor为指定的大小。 => r2.0: nn.ResizeBilinear 从2.0版本开始已被弃用，并将在未来版本中被移除，建议使用 mindspore.ops.ResizeBilinearV2 或 mindspore.ops.interpolate 代替。|r1.10: Ascend/CPU/GPU => r2.0: |图像处理层
[mindspore.nn.SSIM](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.SSIM.html#mindspore.nn.SSIM)|Deleted|计算两个图像之间的结构相似性（SSIM）。|Ascend/GPU/CPU|图像处理层
[mindspore.nn.Upsample](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.Upsample.html#mindspore.nn.Upsample)|New|详情请参考 mindspore.ops.interpolate() 。|r2.0: Ascend/GPU/CPU|图像处理层
[mindspore.nn.ReflectionPad3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.ReflectionPad3d.html#mindspore.nn.ReflectionPad3d)|New|使用反射的方式，以 input 的边界为对称轴，对 input 进行填充。|r2.0: Ascend/GPU/CPU|填充层
[mindspore.nn.ReplicationPad1d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.ReplicationPad1d.html#mindspore.nn.ReplicationPad1d)|New|根据 padding 对输入 x 的W维度上进行填充。|r2.0: GPU|填充层
[mindspore.nn.ReplicationPad2d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.ReplicationPad2d.html#mindspore.nn.ReplicationPad2d)|New|根据 padding 对输入 x 的HW维度上进行填充。|r2.0: GPU|填充层
[mindspore.nn.ReplicationPad3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.ReplicationPad3d.html#mindspore.nn.ReplicationPad3d)|New|根据 padding 对输入 x 的DHW维度上进行填充。|r2.0: GPU|填充层
[mindspore.nn.WithGradCell](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.WithGradCell.html#mindspore.nn.WithGradCell)|Deleted|Cell that returns the gradients.|Ascend/GPU/CPU|封装层
[mindspore.nn.ChannelShuffle](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.ChannelShuffle.html#mindspore.nn.ChannelShuffle)|New|将shape为 $(\*, C, H, W)$ 的Tensor的通道划分成 $g$ 组，得到shape为 $(\*, C \frac g, g, H, W)$ 的Tensor，并沿着 $C$ 和 $\frac{g}{}$， $g$ 对应轴进行转置，将Tensor还原成原有的shape。|r2.0: Ascend/GPU/CPU|工具
[mindspore.nn.ClipByNorm](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.ClipByNorm.html#mindspore.nn.ClipByNorm)|Deleted|对输入Tensor的值进行裁剪，使用 $L_2$ 范数控制梯度。|Ascend/GPU/CPU|工具
[mindspore.nn.Identity](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.Identity.html#mindspore.nn.Identity)|New|网络占位符，返回与输入完全一致。|r2.0: Ascend/GPU/CPU|工具
[mindspore.nn.L1Regularizer](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.L1Regularizer.html#mindspore.nn.L1Regularizer)|Deleted|对权重计算L1正则化。|Ascend/GPU/CPU|工具
[mindspore.nn.Norm](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Norm.html#mindspore.nn.Norm)|Deleted|计算向量的范数，目前包括欧几里得范数，即 $L_2$-norm。|Ascend/GPU/CPU|工具
[mindspore.nn.OneHot](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.OneHot.html#mindspore.nn.OneHot)|Deleted|对输入进行one-hot编码并返回。|Ascend/GPU/CPU|工具
[mindspore.nn.Range](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Range.html#mindspore.nn.Range)|Deleted|根据指定步长在范围[start, limit)中创建数字序列。|Ascend/GPU/CPU|工具
[mindspore.nn.Roll](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Roll.html#mindspore.nn.Roll)|Deleted|沿轴移动Tensor的元素。|Ascend/GPU|工具
[mindspore.nn.Tril](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Tril.html#mindspore.nn.Tril)|Deleted|返回一个Tensor，指定主对角线以上的元素被置为零。|Ascend/GPU/CPU|工具
[mindspore.nn.Triu](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Triu.html#mindspore.nn.Triu)|Deleted|返回一个Tensor，指定主对角线以下的元素被置为0。|Ascend/GPU/CPU|工具
[mindspore.nn.Unflatten](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.Unflatten.html#mindspore.nn.Unflatten)|New|根据 axis 和 unflattened_size 折叠指定维度为给定形状。|r2.0: Ascend/GPU/CPU|工具
[mindspore.nn.CTCLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.CTCLoss.html#mindspore.nn.CTCLoss)|New|CTCLoss损失函数。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.nn.GaussianNLLLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.GaussianNLLLoss.html#mindspore.nn.GaussianNLLLoss)|New|服从高斯分布的负对数似然损失。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.nn.HingeEmbeddingLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.HingeEmbeddingLoss.html#mindspore.nn.HingeEmbeddingLoss)|New|Hinge Embedding 损失函数。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.nn.KLDivLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.KLDivLoss.html#mindspore.nn.KLDivLoss)|New|计算输入 logits 和 labels 的KL散度。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.nn.MarginRankingLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.MarginRankingLoss.html#mindspore.nn.MarginRankingLoss)|New|排序损失函数，用于创建一个衡量给定损失的标准。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.nn.MultiLabelSoftMarginLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.MultiLabelSoftMarginLoss.html#mindspore.nn.MultiLabelSoftMarginLoss)|New|基于最大熵计算用于多标签优化的损失。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.nn.MultiMarginLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.MultiMarginLoss.html#mindspore.nn.MultiMarginLoss)|New|多分类场景下用于计算 $x$ 和 $y$ 之间的合页损失（Hinge Loss），其中 x 为一个2-D Tensor，y 为一个表示类别索引的1-D Tensor， $0 \leq y \leq \text{x.size}(1)-1$。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.nn.MultilabelMarginLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.MultilabelMarginLoss.html#mindspore.nn.MultilabelMarginLoss)|New|创建一个损失函数，用于最小化多分类任务的合页损失。|r2.0: Ascend/GPU|损失函数
[mindspore.nn.PoissonNLLLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.PoissonNLLLoss.html#mindspore.nn.PoissonNLLLoss)|New|计算泊松负对数似然损失。|r2.0: Ascend/GPU/CPU|损失函数
[mindspore.nn.SoftMarginLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.SoftMarginLoss.html#mindspore.nn.SoftMarginLoss)|Changed|针对二分类问题的损失函数。|r1.10: Ascend => r2.0: Ascend/GPU|损失函数
[mindspore.nn.TripletMarginLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.TripletMarginLoss.html#mindspore.nn.TripletMarginLoss)|New|执行三元组损失函数的操作。|r2.0: GPU|损失函数
[mindspore.nn.Moments](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Moments.html#mindspore.nn.Moments)|Deleted|沿指定轴 axis 计算输入 x 的均值和方差。|Ascend/GPU/CPU|数学运算
[mindspore.nn.AdaptiveAvgPool3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.AdaptiveAvgPool3d.html#mindspore.nn.AdaptiveAvgPool3d)|Changed|对输入Tensor，提供三维的自适应平均池化操作。|r1.10: GPU => r2.0: Ascend/GPU/CPU|池化层
[mindspore.nn.AdaptiveMaxPool3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.AdaptiveMaxPool3d.html#mindspore.nn.AdaptiveMaxPool3d)|New|对输入Tensor执行三维自适应最大池化操作。|r2.0: GPU/CPU|池化层
[mindspore.nn.AvgPool3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.AvgPool3d.html#mindspore.nn.AvgPool3d)|New|在一个输入Tensor上应用3D平均池化运算，输入Tensor可看成是由一系列3D平面组成的。|r2.0: Ascend/GPU/CPU|池化层
[mindspore.nn.FractionalMaxPool3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.FractionalMaxPool3d.html#mindspore.nn.FractionalMaxPool3d)|New|在输入上应用三维分数最大池化。|r2.0: GPU/CPU|池化层
[mindspore.nn.LPPool1d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.LPPool1d.html#mindspore.nn.LPPool1d)|New|在一个输入Tensor上应用1D LP池化运算，可被视为组成一个1D输入平面。|r2.0: Ascend/GPU/CPU|池化层
[mindspore.nn.LPPool2d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.LPPool2d.html#mindspore.nn.LPPool2d)|New|在一个输入Tensor上应用2D LP池化运算，可被视为组成一个2D输入平面。|r2.0: Ascend/GPU/CPU|池化层
[mindspore.nn.MaxPool3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.MaxPool3d.html#mindspore.nn.MaxPool3d)|New|在一个输入Tensor上应用3D最大池化运算，输入Tensor可看成是由一系列3D平面组成的。|r2.0: Ascend/GPU/CPU|池化层
[mindspore.nn.MaxUnpool1d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.MaxUnpool1d.html#mindspore.nn.MaxUnpool1d)|New|计算 mindspore.nn.MaxPool1d 的逆过程。|r2.0: GPU/CPU|池化层
[mindspore.nn.MaxUnpool2d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.MaxUnpool2d.html#mindspore.nn.MaxUnpool2d)|New|mindspore.nn.MaxPool2d 的逆过程。|r2.0: GPU/CPU|池化层
[mindspore.nn.MaxUnpool3d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.MaxUnpool3d.html#mindspore.nn.MaxUnpool3d)|New|mindspore.nn.MaxPool3d 的逆过程。|r2.0: GPU/CPU|池化层
[mindspore.nn.MatrixDiag](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.MatrixDiag.html#mindspore.nn.MatrixDiag)|Deleted|根据对角线值返回一批对角矩阵。|Ascend|矩阵处理
[mindspore.nn.MatrixDiagPart](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.MatrixDiagPart.html#mindspore.nn.MatrixDiagPart)|Deleted|返回批对角矩阵的对角线部分。|Ascend|矩阵处理
[mindspore.nn.MatrixSetDiag](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.MatrixSetDiag.html#mindspore.nn.MatrixSetDiag)|Deleted|将输入的对角矩阵的对角线值置换为输入的对角线值。|Ascend|矩阵处理
[mindspore.nn.Accuracy](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Accuracy.html#mindspore.nn.Accuracy)|Deleted|计算数据分类的正确率，包括二分类和多分类。|Ascend/GPU/CPU|评价指标
[mindspore.nn.BleuScore](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.BleuScore.html#mindspore.nn.BleuScore)|Deleted|计算BLEU分数。|Ascend/GPU/CPU|评价指标
[mindspore.nn.ConfusionMatrix](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.ConfusionMatrix.html#mindspore.nn.ConfusionMatrix)|Deleted|计算混淆矩阵(confusion matrix)，通常用于评估分类模型的性能，包括二分类和多分类场景。|Ascend/GPU/CPU|评价指标
[mindspore.nn.ConfusionMatrixMetric](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.ConfusionMatrixMetric.html#mindspore.nn.ConfusionMatrixMetric)|Deleted|计算与混淆矩阵相关的度量。|Ascend/GPU/CPU|评价指标
[mindspore.nn.CosineSimilarity](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.CosineSimilarity.html#mindspore.nn.CosineSimilarity)|Deleted|计算余弦相似度。|Ascend/GPU/CPU|评价指标
[mindspore.nn.Dice](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Dice.html#mindspore.nn.Dice)|Deleted|集合相似性度量。|Ascend/GPU/CPU|评价指标
[mindspore.nn.F1](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.F1.html#mindspore.nn.F1)|Deleted|计算F1 score。|Ascend/GPU/CPU|评价指标
[mindspore.nn.Fbeta](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Fbeta.html#mindspore.nn.Fbeta)|Deleted|计算Fbeta评分。|Ascend/GPU/CPU|评价指标
[mindspore.nn.HausdorffDistance](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.HausdorffDistance.html#mindspore.nn.HausdorffDistance)|Deleted|计算Hausdorff距离。|Ascend/GPU/CPU|评价指标
[mindspore.nn.Loss](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Loss.html#mindspore.nn.Loss)|Deleted|计算loss的平均值。|Ascend/GPU/CPU|评价指标
[mindspore.nn.MAE](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.MAE.html#mindspore.nn.MAE)|Deleted|计算平均绝对误差MAE（Mean Absolute Error）。|Ascend/GPU/CPU|评价指标
[mindspore.nn.MSE](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.MSE.html#mindspore.nn.MSE)|Deleted|测量均方差MSE（Mean Squared Error）。|Ascend/GPU/CPU|评价指标
[mindspore.nn.MeanSurfaceDistance](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.MeanSurfaceDistance.html#mindspore.nn.MeanSurfaceDistance)|Deleted|计算从 y_pred 到 y 的平均表面距离。|Ascend/GPU/CPU|评价指标
[mindspore.nn.Metric](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Metric.html#mindspore.nn.Metric)|Deleted|用于计算评估指标的基类。|Ascend/GPU/CPU|评价指标
[mindspore.nn.OcclusionSensitivity](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.OcclusionSensitivity.html#mindspore.nn.OcclusionSensitivity)|Deleted|用于计算神经网络对给定图像的遮挡灵敏度（Occlusion Sensitivity），表示了图像的哪些部分对神经网络的分类决策最重要。|Ascend/GPU/CPU|评价指标
[mindspore.nn.Perplexity](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Perplexity.html#mindspore.nn.Perplexity)|Deleted|计算困惑度（perplexity）。|Ascend/GPU/CPU|评价指标
[mindspore.nn.Precision](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Precision.html#mindspore.nn.Precision)|Deleted|计算数据分类的精度，包括单标签场景和多标签场景。|Ascend/GPU/CPU|评价指标
[mindspore.nn.ROC](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.ROC.html#mindspore.nn.ROC)|Deleted|计算ROC曲线。|Ascend/GPU/CPU|评价指标
[mindspore.nn.Recall](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Recall.html#mindspore.nn.Recall)|Deleted|计算数据分类的召回率，包括单标签场景和多标签场景。|Ascend/GPU/CPU|评价指标
[mindspore.nn.RootMeanSquareDistance](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.RootMeanSquareDistance.html#mindspore.nn.RootMeanSquareDistance)|Deleted|计算从 y_pred 到 y 的均方根表面距离。|Ascend/GPU/CPU|评价指标
[mindspore.nn.Top1CategoricalAccuracy](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Top1CategoricalAccuracy.html#mindspore.nn.Top1CategoricalAccuracy)|Deleted|计算top-1分类正确率。|Ascend/GPU/CPU|评价指标
[mindspore.nn.Top5CategoricalAccuracy](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Top5CategoricalAccuracy.html#mindspore.nn.Top5CategoricalAccuracy)|Deleted|计算top-5分类正确率。|Ascend/GPU/CPU|评价指标
[mindspore.nn.TopKCategoricalAccuracy](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.TopKCategoricalAccuracy.html#mindspore.nn.TopKCategoricalAccuracy)|Deleted|计算top-k分类正确率。|Ascend/GPU/CPU|评价指标
[mindspore.nn.auc](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.auc.html#mindspore.nn.auc)|Deleted|使用梯形法则计算曲线下面积AUC（Area Under the Curve，AUC）。|Ascend/GPU/CPU|评价指标
[mindspore.nn.get_metric_fn](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.get_metric_fn.html#mindspore.nn.get_metric_fn)|Deleted|根据输入的 name 获取metric的方法。|Ascend/GPU/CPU|评价指标
[mindspore.nn.names](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.names.html#mindspore.nn.names)|Deleted|获取所有metric的名称。|Ascend/GPU/CPU|评价指标
[mindspore.nn.rearrange_inputs](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.rearrange_inputs.html#mindspore.nn.rearrange_inputs)|Deleted|此装饰器用于根据类的 indexes 属性对输入重新排列。|Ascend/GPU/CPU|评价指标
[mindspore.nn.GLU](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.GLU.html#mindspore.nn.GLU)|New|门线性单元函数（Gated Linear Unit function）。|r2.0: Ascend/GPU/CPU|非线性激活函数层
[mindspore.nn.PReLU](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.PReLU.html#mindspore.nn.PReLU)|Changed|PReLU激活层（PReLU Activation Operator）。|r1.10: Ascend/GPU => r2.0: Ascend/GPU/CPU|非线性激活函数层
[mindspore.nn.Softmax2d](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.Softmax2d.html#mindspore.nn.Softmax2d)|New|应用于2D特征数据的Softmax函数。|r2.0: Ascend/GPU/CPU|非线性激活函数层
