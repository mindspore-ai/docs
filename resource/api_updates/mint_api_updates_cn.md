# mindspore.mint API接口变更

2.8.0版本与2.7.2版本相比，MindSpore中`mindspore.mint`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.mint.nn.CosineEmbeddingLoss](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/mint/mindspore.mint.nn.CosineEmbeddingLoss.html#mindspore.mint.nn.CosineEmbeddingLoss)|New|余弦相似度损失函数，用于测量两个Tensor之间的相似性。|r2.8.0: Ascend|损失函数
[mindspore.mint.nn.functional.cosine_embedding_loss](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/mint/mindspore.mint.nn.functional.cosine_embedding_loss.html#mindspore.mint.nn.functional.cosine_embedding_loss)|New|余弦相似度损失函数，用于测量两个Tensor之间的相似性。|r2.8.0: Ascend|损失函数
[mindspore.mint.nn.functional.adaptive_max_pool2d](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/mint/mindspore.mint.nn.functional.adaptive_max_pool2d.html#mindspore.mint.nn.functional.adaptive_max_pool2d)|New|对输入Tensor，提供二维自适应最大池化操作。|r2.8.0: Ascend|池化函数
[mindspore.mint.nn.AdaptiveMaxPool2d](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/mint/mindspore.mint.nn.AdaptiveMaxPool2d.html#mindspore.mint.nn.AdaptiveMaxPool2d)|New|对输入Tensor，提供二维自适应最大池化操作。|r2.8.0: Ascend|池化层
