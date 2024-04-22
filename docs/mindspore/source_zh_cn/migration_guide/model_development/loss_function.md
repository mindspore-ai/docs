# 损失函数

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3.q1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3.q1/docs/mindspore/source_zh_cn/migration_guide/model_development/loss_function.md)

在阅读本章节之前，请先阅读MindSpore官网教程[损失函数](https://www.mindspore.cn/tutorials/zh-CN/r2.3.0rc1/advanced/modules/loss.html)。

MindSpore官网教程损失函数中讲解了内置、自定义和多标签损失函数，以及在模型训练中的使用指导，这里就MindSpore的损失函数与PyTorch的损失函数在功能和接口差异方面给出差异列表。

| torch.nn | torch.nn.functional | mindspore.nn | mindspore.ops | 差异说明 |
| -------- | ------------------- | ------------ | ------------- | ------- |
| torch.nn.L1Loss | torch.nn.functional.l1_loss | mindspore.nn.L1Loss| mindspore.ops.l1_loss| 一致 |
| torch.nn.MSELoss | torch.nn.functional.mse_loss | mindspore.nn.MSELoss| mindspore.ops.mse_loss| 一致 |
| torch.nn.CrossEntropyLoss | torch.nn.functional.cross_entropy | mindspore.nn.CrossEntropyLoss| mindspore.ops.cross_entropy| [nn接口差异](https://www.mindspore.cn/docs/zh-CN/r2.3.0rc1/note/api_mapping/pytorch_diff/CrossEntropyLoss.html) |
| torch.nn.CTCLoss | torch.nn.functional.ctc_loss | mindspore.nn.CTCLoss| mindspore.ops.ctc_loss| 一致 |
| torch.nn.NLLLoss | torch.nn.functional.nll_loss | mindspore.nn.NLLLoss| mindspore.ops.nll_loss| 一致 |
| torch.nn.PoissonNLLLoss | torch.nn.functional.poisson_nll_loss | mindspore.nn.PoissonNLLLoss| - | 一致 |
| torch.nn.GaussianNLLLoss | torch.nn.functional.gaussian_nll_loss | mindspore.nn.GaussianNLLLoss| mindspore.ops.gaussian_nll_loss | 一致 |
| torch.nn.KLDivLoss | torch.nn.functional.kl_div | mindspore.nn.KLDivLoss| mindspore.ops.kl_div| MindSpore不支持 `log_target` 参数 |
| torch.nn.BCELoss | torch.nn.functional.binary_cross_entropy | mindspore.nn.BCELoss| mindspore.ops.binary_cross_entropy| 一致 |
| torch.nn.BCEWithLogitsLoss | torch.nn.functional.binary_cross_entropy_with_logits | mindspore.nn.BCEWithLogitsLoss| mindspore.ops.binary_cross_entropy_with_logits| 一致 |
| torch.nn.MarginRankingLoss | torch.nn.functional.margin_ranking_loss | mindspore.nn.MarginRankingLoss| mindspore.ops.margin_ranking_loss | 一致 |
| torch.nn.HingeEmbeddingLoss | torch.nn.functional.hinge_embedding_loss | mindspore.nn.HingeEmbeddingLoss| mindspore.ops.hinge_embedding_loss | 一致 |
| torch.nn.MultiLabelMarginLoss | torch.nn.functional.multilabel_margin_loss | mindspore.nn.MultiLabelMarginLoss | mindspore.ops.multilabel_margin_loss| 一致 |
| torch.nn.HuberLoss | torch.nn.functional.huber_loss | mindspore.nn.HuberLoss | mindspore.ops.huber_loss| 一致 |
| torch.nn.SmoothL1Loss | torch.nn.functional.smooth_l1_loss | mindspore.nn.SmoothL1Loss | mindspore.ops.smooth_l1_loss| 一致 |
| torch.nn.SoftMarginLoss | torch.nn.functional.soft_margin_loss | mindspore.nn.SoftMarginLoss| mindspore.ops.soft_margin_loss | 一致 |
| torch.nn.MultiLabelSoftMarginLoss | torch.nn.functional.multilabel_soft_margin_loss | mindspore.nn.MultiLabelSoftMarginLoss| mindspore.ops.multilabel_soft_margin_loss| 一致 |
| torch.nn.CosineEmbeddingLoss | torch.nn.functional.cosine_embedding_loss | mindspore.nn.CosineEmbeddingLoss| mindspore.ops.cosine_embedding_loss| 一致 |
| torch.nn.MultiMarginLoss | torch.nn.functional.multi_margin_loss | mindspore.nn.MultiMarginLoss | mindspore.ops.multi_margin_loss | 一致 |
| torch.nn.TripletMarginLoss | torch.nn.functional.triplet_margin_loss | mindspore.nn.TripletMarginLoss| mindspore.ops.triplet_margin_loss | [功能一致，参数个数或顺序不一致](https://www.mindspore.cn/docs/zh-CN/r2.3.0rc1/note/api_mapping/pytorch_diff/TripletMarginLoss.html) |
| torch.nn.TripletMarginWithDistanceLoss | torch.nn.functional.triplet_margin_with_distance_loss | mindspore.nn.TripletMarginWithDistanceLoss | - | 一致 |