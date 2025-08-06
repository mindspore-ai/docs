# Dynamic Shape Support Status of nn Interface

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/api_python/dynamic_shape_nn.md)

> The following list provides nn interfaces that support dynamic shape functionality in PYNATIVE mode. However, some nn interfaces may have incomplete data type support. If you encounter such issues, you can resolve them by manually incorporating the [Cast](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Cast.html) operator.
>
> nn interfaces outside of this list have limited support for dynamic shape functionality and may fail to execute. Additionally, in graph mode, dynamic shape functionality is also limited and may result in execution failures.
>
> If you encounter issues that the execution of dynamic shape operator fails, it is recommended to avoid introducing dynamic shape in the network. For example, you can adjust the inputs of nn interfaces to construct a fully static shape network or confine dynamic shape to a local scope within the network.

| API name  | Ascend |  GPU  |   CPU  |
| :--- |:-------- | :------- |:---------|
|[mindspore.nn.Adam](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.Adam.html)|✔️|✔️|✔️|
|[mindspore.nn.AdaptiveAvgPool1d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.AdaptiveAvgPool1d.html)|✔️|✔️|✔️|
|[mindspore.nn.AdaptiveAvgPool2d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.AdaptiveAvgPool2d.html)|✔️|✔️|✔️|
|[mindspore.nn.AdaptiveAvgPool3d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.AdaptiveAvgPool3d.html)|✔️|✔️|✔️|
|[mindspore.nn.AdaptiveMaxPool1d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.AdaptiveMaxPool1d.html)|✔️|✔️|✔️|
|[mindspore.nn.AvgPool1d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.AvgPool1d.html)|✔️|✔️|✔️|
|[mindspore.nn.AvgPool2d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.AvgPool2d.html)|✔️|✔️|✔️|
|[mindspore.nn.AvgPool3d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.AvgPool3d.html)|✔️|✔️|✔️|
|[mindspore.nn.BatchNorm1d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.BatchNorm1d.html)|✔️|✔️|✔️|
|[mindspore.nn.BatchNorm2d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.BatchNorm2d.html)|✔️|✔️|✔️|
|[mindspore.nn.BatchNorm3d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.BatchNorm3d.html)|✔️|✔️|✔️|
|[mindspore.nn.BCELoss](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.BCELoss.html)|✔️|✔️|✔️|
|[mindspore.nn.BCEWithLogitsLoss](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.BCEWithLogitsLoss.html)|✔️|✔️|✔️|
|[mindspore.nn.ConstantPad1d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.ConstantPad1d.html)|✔️|✔️|✔️|
|[mindspore.nn.ConstantPad2d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.ConstantPad2d.html)|✔️|✔️|✔️|
|[mindspore.nn.Conv1d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.Conv1d.html)|✔️|✔️|✔️|
|[mindspore.nn.Conv1dTranspose](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.Conv1dTranspose.html)|✔️|✔️|✔️|
|[mindspore.nn.Conv2d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.Conv2d.html)|✔️|✔️|✔️|
|[mindspore.nn.Conv2dTranspose](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.Conv2dTranspose.html)|✔️|✔️|✔️|
|[mindspore.nn.Conv3d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.Conv3d.html)|✔️|✔️|✔️|
|[mindspore.nn.Conv3dTranspose](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.Conv3dTranspose.html)|✔️|✔️|✔️|
|[mindspore.nn.CosineEmbeddingLoss](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.CosineEmbeddingLoss.html)|✔️|✔️|✔️|
|[mindspore.nn.CrossEntropyLoss](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.CrossEntropyLoss.html)|✔️|✔️|✔️|
|[mindspore.nn.CTCLoss](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.CTCLoss.html)|✔️|✔️|✔️|
|[mindspore.nn.Dense](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.Dense.html)|✔️|✔️|✔️|
|[mindspore.nn.Embedding](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.Embedding.html)|✔️|✔️|✔️|
|[mindspore.nn.EmbeddingLookup](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.EmbeddingLookup.html)|✔️|✔️|✔️|
|[mindspore.nn.GLU](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.GLU.html)|✔️|✔️|✔️|
|[mindspore.nn.GroupNorm](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.GroupNorm.html)|✔️|✔️|✔️|
|[mindspore.nn.GRU](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.GRU.html)|❌|❌|✔️|
|[mindspore.nn.GRUCell](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.GRUCell.html)|✔️|✔️|✔️|
|[mindspore.nn.InstanceNorm1d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.InstanceNorm1d.html)|❌|✔️|❌|
|[mindspore.nn.InstanceNorm2d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.InstanceNorm2d.html)|❌|✔️|❌|
|[mindspore.nn.InstanceNorm3d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.InstanceNorm3d.html)|❌|✔️|❌|
|[mindspore.nn.KLDivLoss](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.KLDivLoss.html)|✔️|✔️|✔️|
|[mindspore.nn.L1Loss](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.L1Loss.html)|✔️|✔️|✔️|
|[mindspore.nn.LeakyReLU](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.LeakyReLU.html)|✔️|✔️|✔️|
|[mindspore.nn.LRN](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.LRN.html)|✔️|✔️|✔️|
|[mindspore.nn.LSTM](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.LSTM.html)|✔️|✔️|✔️|
|[mindspore.nn.MarginRankingLoss](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.MarginRankingLoss.html)|✔️|✔️|✔️|
|[mindspore.nn.MaxPool1d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.MaxPool1d.html)|✔️|✔️|✔️|
|[mindspore.nn.MaxPool2d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.MaxPool2d.html)|✔️|✔️|✔️|
|[mindspore.nn.MaxPool3d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.MaxPool3d.html)|✔️|✔️|✔️|
|[mindspore.nn.MaxUnpool2d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.MaxUnpool2d.html)|❌|✔️|✔️|
|[mindspore.nn.MSELoss](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.MSELoss.html)|✔️|✔️|✔️|
|[mindspore.nn.MultiLabelSoftMarginLoss](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.MultiLabelSoftMarginLoss.html)|✔️|✔️|✔️|
|[mindspore.nn.PixelShuffle](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.PixelShuffle.html)|✔️|✔️|✔️|
|[mindspore.nn.ReflectionPad1d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.ReflectionPad1d.html)|✔️|❌|✔️|
|[mindspore.nn.ReplicationPad2d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.ReplicationPad2d.html)|❌|✔️|❌|
|[mindspore.nn.RReLU](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.RReLU.html)|✔️|✔️|✔️|
|[mindspore.nn.SmoothL1Loss](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.SmoothL1Loss.html)|✔️|✔️|✔️|
|[mindspore.nn.Softmax2d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.Softmax2d.html)|✔️|✔️|✔️|
|[mindspore.nn.SoftmaxCrossEntropyWithLogits](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.SoftmaxCrossEntropyWithLogits.html)|✔️|✔️|✔️|
|[mindspore.nn.ZeroPad2d](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.ZeroPad2d.html)|✔️|✔️|✔️|
|[mindspore.mint.nn](https://www.mindspore.cn/docs/en/br_base/api_python/mindspore.mint.html#mindspore-mint-nn)|✔️|❌|❌|
