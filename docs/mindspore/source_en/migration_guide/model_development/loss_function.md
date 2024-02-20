# Loss Function

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/migration_guide/model_development/loss_function.md)

Before reading this chapter, please read the MindSpore official website tutorial first[Loss Function](https://www.mindspore.cn/tutorials/en/master/advanced/modules/loss.html).

The MindSpore official website tutorial on loss functions explains built-in, custom, and multi label loss functions, as well as guidance on their use in model training. Here is a list of differences in functionality and interface between MindSpore's loss function and PyTorch's loss function.

| torch.nn | torch.nn.functional | mindspore.nn | mindspore.ops | Difference Description |
| -------- | ------------------- | ------------ | ------------- | ------- |
| torch.nn.L1Loss | torch.nn.functional.l1_loss | mindspore.nn.L1Loss| mindspore.ops.l1_loss| consistent |
| torch.nn.MSELoss | torch.nn.functional.mse_loss | mindspore.nn.MSELoss| mindspore.ops.mse_loss| consistent |
| torch.nn.CrossEntropyLoss | torch.nn.functional.cross_entropy | mindspore.nn.CrossEntropyLoss| mindspore.ops.cross_entropy| [nn interface difference](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_diff/CrossEntropyLoss.html) |
| torch.nn.CTCLoss | torch.nn.functional.ctc_loss | mindspore.nn.CTCLoss| mindspore.ops.ctc_loss| consistent |
| torch.nn.NLLLoss | torch.nn.functional.nll_loss | mindspore.nn.NLLLoss| mindspore.ops.nll_loss| consistent |
| torch.nn.PoissonNLLLoss | torch.nn.functional.poisson_nll_loss | mindspore.nn.PoissonNLLLoss| - | consistent |
| torch.nn.GaussianNLLLoss | torch.nn.functional.gaussian_nll_loss | mindspore.nn.GaussianNLLLoss| mindspore.ops.gaussian_nll_loss | consistent |
| torch.nn.KLDivLoss | torch.nn.functional.kl_div | mindspore.nn.KLDivLoss| mindspore.ops.kl_div| MindSpore does not support the `log_target` parameter |
| torch.nn.BCELoss | torch.nn.functional.binary_cross_entropy | mindspore.nn.BCELoss| mindspore.ops.binary_cross_entropy| consistent |
| torch.nn.BCEWithLogitsLoss | torch.nn.functional.binary_cross_entropy_with_logits | mindspore.nn.BCEWithLogitsLoss| mindspore.ops.binary_cross_entropy_with_logits| consistent |
| torch.nn.MarginRankingLoss | torch.nn.functional.margin_ranking_loss | mindspore.nn.MarginRankingLoss| mindspore.ops.margin_ranking_loss | consistent |
| torch.nn.HingeEmbeddingLoss | torch.nn.functional.hinge_embedding_loss | mindspore.nn.HingeEmbeddingLoss| mindspore.ops.hinge_embedding_loss | consistent |
| torch.nn.MultiLabelMarginLoss | torch.nn.functional.multilabel_margin_loss | mindspore.nn.MultiLabelMarginLoss | mindspore.ops.multilabel_margin_loss| consistent |
| torch.nn.HuberLoss | torch.nn.functional.huber_loss | mindspore.nn.HuberLoss | mindspore.ops.huber_loss| consistent |
| torch.nn.SmoothL1Loss | torch.nn.functional.smooth_l1_loss | mindspore.nn.SmoothL1Loss | mindspore.ops.smooth_l1_loss| consistent |
| torch.nn.SoftMarginLoss | torch.nn.functional.soft_margin_loss | mindspore.nn.SoftMarginLoss| mindspore.ops.soft_margin_loss | consistent |
| torch.nn.MultiLabelSoftMarginLoss | torch.nn.functional.multilabel_soft_margin_loss | mindspore.nn.MultiLabelSoftMarginLoss| mindspore.ops.multilabel_soft_margin_loss| consistent |
| torch.nn.CosineEmbeddingLoss | torch.nn.functional.cosine_embedding_loss | mindspore.nn.CosineEmbeddingLoss| mindspore.ops.cosine_embedding_loss| consistent |
| torch.nn.MultiMarginLoss | torch.nn.functional.multi_margin_loss | mindspore.nn.MultiMarginLoss | mindspore.ops.multi_margin_loss | consistent |
| torch.nn.TripletMarginLoss | torch.nn.functional.triplet_margin_loss | mindspore.nn.TripletMarginLoss| mindspore.ops.triplet_margin_loss | [Functionality is consistent, but the number or order of parameters is not consistent](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_diff/TripletMarginLoss.html) |
| torch.nn.TripletMarginWithDistanceLoss | torch.nn.functional.triplet_margin_with_distance_loss | mindspore.nn.TripletMarginWithDistanceLoss | - | consistent |