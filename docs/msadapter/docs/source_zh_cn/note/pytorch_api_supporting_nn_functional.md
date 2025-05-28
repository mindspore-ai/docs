# torch.nn.functional

## Convolution functions

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[conv1d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.conv1d.html)|Beta|支持数据类型：fp16、fp32|
|[conv2d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.conv2d.html)|Beta|支持数据类型：bf16、fp16、fp32|
|[conv3d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.conv3d.html)|Not Support|N/A|
|[conv_transpose1d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.conv_transpose1d.html)|Not Support|N/A|
|[conv_transpose2d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.conv_transpose2d.html)|Not Support|N/A|
|[conv_transpose3d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.conv_transpose3d.html)|Not Support|N/A|
|[unfold](https://pytorch.org/docs/2.1/generated/torch.nn.functional.unfold.html)|Stable|N/A|
|[fold](https://pytorch.org/docs/2.1/generated/torch.nn.functional.fold.html)|Stable|N/A|

## Pooling functions

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[avg_pool1d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.avg_pool1d.html)|Beta|支持数据类型：fp16、fp32|
|[avg_pool2d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.avg_pool2d.html)|Stable|divisor_override默认值torch为None，msadapter为0；支持数据类型：fp16、fp32|
|[avg_pool3d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.avg_pool3d.html)|Not Support|N/A|
|[max_pool1d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.max_pool1d.html)|Beta|N/A|
|[max_pool2d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.max_pool2d.html)|Stable|N/A|
|[max_pool3d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.max_pool3d.html)|Not Support|N/A|
|[max_unpool1d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.max_unpool1d.html)|Not Support|N/A|
|[max_unpool2d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.max_unpool2d.html)|Not Support|N/A|
|[max_unpool3d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.max_unpool3d.html)|Not Support|N/A|
|[lp_pool1d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.lp_pool1d.html)|Not Support|N/A|
|[lp_pool2d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.lp_pool2d.html)|Not Support|N/A|
|[lp_pool3d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.lp_pool3d.html)|Not Support|N/A|
|[adaptive_max_pool1d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.adaptive_max_pool1d.html)|Not Support|N/A|
|[adaptive_max_pool2d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.adaptive_max_pool2d.html)|Not Support|N/A|
|[adaptive_max_pool3d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.adaptive_max_pool3d.html)|Not Support|N/A|
|[adaptive_avg_pool1d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.adaptive_avg_pool1d.html)|Not Support|N/A|
|[adaptive_avg_pool2d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.adaptive_avg_pool2d.html)|Not Support|N/A|
|[adaptive_avg_pool3d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.adaptive_avg_pool3d.html)|Not Support|N/A|
|[fractional_max_pool2d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.fractional_max_pool2d.html)|Not Support|N/A|
|[fractional_max_pool3d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.fractional_max_pool3d.html)|Not Support|N/A|

## Attention Mechanisms

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[scaled_dot_product_attention](https://pytorch.org/docs/2.1/generated/torch.nn.functional.scaled_dot_product_attention.html)|Not Support|N/A|

## Non-linear activation functions

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[threshold](https://pytorch.org/docs/2.1/generated/torch.nn.functional.threshold.html)|Not Support|N/A|
|[threshold_](https://pytorch.org/docs/2.1/generated/torch.nn.functional.threshold_.html)|Not Support|N/A|
|[relu](https://pytorch.org/docs/2.1/generated/torch.nn.functional.relu.html)|Stable|支持数据类型：bf16、fp16、fp32、uint8、int8、int32、int64|
|[relu_](https://pytorch.org/docs/2.1/generated/torch.nn.functional.relu_.html)|Not Support|N/A|
|[hardtanh](https://pytorch.org/docs/2.1/generated/torch.nn.functional.hardtanh.html)|Not Support|N/A|
|[hardtanh_](https://pytorch.org/docs/2.1/generated/torch.nn.functional.hardtanh_.html)|Not Support|N/A|
|[hardswish](https://pytorch.org/docs/2.1/generated/torch.nn.functional.hardswish.html)|Beta|入参不支持inplace；支持数据类型：fp16、fp32|
|[relu6](https://pytorch.org/docs/2.1/generated/torch.nn.functional.relu6.html)|Beta|支持数据类型：fp16、fp32、uint8、int8、int16、int32、int64|
|[elu](https://pytorch.org/docs/2.1/generated/torch.nn.functional.elu.html)|Beta|入参不支持inplace；支持数据类型：fp16、fp32|
|[elu_](https://pytorch.org/docs/2.1/generated/torch.nn.functional.elu_.html)|Not Support|N/A|
|[selu](https://pytorch.org/docs/2.1/generated/torch.nn.functional.selu.html)|Beta|入参不支持inplace、支持数据类型：fp16、fp32|
|[celu](https://pytorch.org/docs/2.1/generated/torch.nn.functional.celu.html)|Beta|支持数据类型：fp16、fp32|
|[leaky_relu](https://pytorch.org/docs/2.1/generated/torch.nn.functional.leaky_relu.html)|Beta|入参不支持inplace；支持数据类型：bf16、fp16、fp32、fp64|
|[leaky_relu_](https://pytorch.org/docs/2.1/generated/torch.nn.functional.leaky_relu_.html)|Not Support|N/A|
|[prelu](https://pytorch.org/docs/2.1/generated/torch.nn.functional.prelu.html)|Stable|支持数据类型：fp16、fp32|
|[rrelu](https://pytorch.org/docs/2.1/generated/torch.nn.functional.rrelu.html)|Not Support|N/A|
|[rrelu_](https://pytorch.org/docs/2.1/generated/torch.nn.functional.rrelu_.html)|Not Support|N/A|
|[glu](https://pytorch.org/docs/2.1/generated/torch.nn.functional.glu.html)|Beta|支持数据类型：fp16、fp32|
|[gelu](https://pytorch.org/docs/2.1/generated/torch.nn.functional.gelu.html)|Stable|支持数据类型：bf16、fp16、fp32|
|[logsigmoid](https://pytorch.org/docs/2.1/generated/torch.nn.functional.logsigmoid.html)|Stable|支持数据类型：fp16、fp32|
|[hardshrink](https://pytorch.org/docs/2.1/generated/torch.nn.functional.hardshrink.html)|Stable|支持数据类型：fp16、fp32|
|[tanhshrink](https://pytorch.org/docs/2.1/generated/torch.nn.functional.tanhshrink.html)|Not Support|N/A|
|[softsign](https://pytorch.org/docs/2.1/generated/torch.nn.functional.softsign.html)|Not Support|N/A|
|[softplus](https://pytorch.org/docs/2.1/generated/torch.nn.functional.softplus.html)|Stable|支持数据类型：fp16、fp32|
|[softmin](https://pytorch.org/docs/2.1/generated/torch.nn.functional.softmin.html)|Not Support|N/A|
|[softmax](https://pytorch.org/docs/2.1/generated/torch.nn.functional.softmax.html)|Stable|入参不支持_stacklevel；支持数据类型：bf16、fp16、fp32|
|[softshrink](https://pytorch.org/docs/2.1/generated/torch.nn.functional.softshrink.html)|Stable|支持数据类型：fp16、fp32|
|[gumbel_softmax](https://pytorch.org/docs/2.1/generated/torch.nn.functional.gumbel_softmax.html)|Beta|N/A|
|[log_softmax](https://pytorch.org/docs/2.1/generated/torch.nn.functional.log_softmax.html)|Beta|入参不支持_stacklevel、dim默认值torch为None、msadapter为-1；支持数据类型：bf16、fp16、fp32|
|[tanh](https://pytorch.org/docs/2.1/generated/torch.nn.functional.tanh.html)|Stable|支持数据类型：fp16、fp32、uint8、int8、int16、int32、int64、bool|
|[sigmoid](https://pytorch.org/docs/2.1/generated/torch.nn.functional.sigmoid.html)|Stable|支持数据类型：fp16、fp32、uint8、int8、int16、int32、int64、bool|
|[hardsigmoid](https://pytorch.org/docs/2.1/generated/torch.nn.functional.hardsigmoid.html)|Beta|入参不支持inplace；支持数据类型：fp16、fp32|
|[silu](https://pytorch.org/docs/2.1/generated/torch.nn.functional.silu.html)|Beta|入参不支持inplace；支持数据类型：fp16、fp32|
|[mish](https://pytorch.org/docs/2.1/generated/torch.nn.functional.mish.html)|Stable|支持数据类型：fp16、fp32|
|[batch_norm](https://pytorch.org/docs/2.1/generated/torch.nn.functional.batch_norm.html)|Stable|支持数据类型：fp16、fp32|
|[group_norm](https://pytorch.org/docs/2.1/generated/torch.nn.functional.group_norm.html)|Stable|支持数据类型：fp16、fp32|
|[instance_norm](https://pytorch.org/docs/2.1/generated/torch.nn.functional.instance_norm.html)|Not Support|N/A|
|[layer_norm](https://pytorch.org/docs/2.1/generated/torch.nn.functional.layer_norm.html)|Stable|支持数据类型：bf16、fp16、fp32|
|[local_response_norm](https://pytorch.org/docs/2.1/generated/torch.nn.functional.local_response_norm.html)|Not Support|N/A|
|[rms_norm](https://pytorch.org/docs/2.1/generated/torch.nn.functional.rms_norm.html)|Not Support|N/A|
|[normalize](https://pytorch.org/docs/2.1/generated/torch.nn.functional.normalize.html)|Beta|不支持out出参；支持数据类型：bf16、fp16、fp32|

## Linear functions

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[linear](https://pytorch.org/docs/2.1/generated/torch.nn.functional.linear.html)|Stable|支持数据类型：fp16、fp32|
|[bilinear](https://pytorch.org/docs/2.1/generated/torch.nn.functional.bilinear.html)|Not Support|N/A|

## Dropout functions

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[dropout](https://pytorch.org/docs/2.1/generated/torch.nn.functional.dropout.html)|Beta|入参不支持inplace；支持数据类型：bf16、fp16、fp32|
|[alpha_dropout](https://pytorch.org/docs/2.1/generated/torch.nn.functional.alpha_dropout.html)|Not Support|N/A|
|[feature_alpha_dropout](https://pytorch.org/docs/2.1/generated/torch.nn.functional.feature_alpha_dropout.html)|Not Support|N/A|
|[dropout1d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.dropout1d.html)|Not Support|N/A|
|[dropout2d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.dropout2d.html)|Beta|支持数据类型：bf16、fp16、fp32|
|[dropout3d](https://pytorch.org/docs/2.1/generated/torch.nn.functional.dropout3d.html)|Not Support|N/A|

## Sparse functions

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[embedding](https://pytorch.org/docs/2.1/generated/torch.nn.functional.embedding.html)|Beta|入参不支持sparse；支持数据类型：int32、int64|
|[embedding_bag](https://pytorch.org/docs/2.1/generated/torch.nn.functional.embedding_bag.html)|Not Support|N/A|
|[one_hot](https://pytorch.org/docs/2.1/generated/torch.nn.functional.one_hot.html)|Stable|支持数据类型：int32、int64|

## Distance functions

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[pairwise_distance](https://pytorch.org/docs/2.1/generated/torch.nn.functional.pairwise_distance.html)|Not Support|N/A|
|[cosine_similarity](https://pytorch.org/docs/2.1/generated/torch.nn.functional.cosine_similarity.html)|Beta|支持数据类型：fp16、fp32|
|[pdist](https://pytorch.org/docs/2.1/generated/torch.nn.functional.pdist.html)|Not Support|N/A|

## Loss functions

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[binary_cross_entropy](https://pytorch.org/docs/2.1/generated/torch.nn.functional.binary_cross_entropy.html)|Not Support|N/A|
|[binary_cross_entropy_with_logits](https://pytorch.org/docs/2.1/generated/torch.nn.functional.binary_cross_entropy_with_logits.html)|Stable|支持数据类型：fp16、fp32|
|[poisson_nll_loss](https://pytorch.org/docs/2.1/generated/torch.nn.functional.poisson_nll_loss.html)|Not Support|N/A|
|[cosine_embedding_loss](https://pytorch.org/docs/2.1/generated/torch.nn.functional.cosine_embedding_loss.html)|Not Support|N/A|
|[cross_entropy](https://pytorch.org/docs/2.1/generated/torch.nn.functional.cross_entropy.html)|Beta|入参不支持reduce、size_average；支持数据类型：fp16、fp32|
|[ctc_loss](https://pytorch.org/docs/2.1/generated/torch.nn.functional.ctc_loss.html)|Not Support|N/A|
|[gaussian_nll_loss](https://pytorch.org/docs/2.1/generated/torch.nn.functional.gaussian_nll_loss.html)|Not Support|N/A|
|[hinge_embedding_loss](https://pytorch.org/docs/2.1/generated/torch.nn.functional.hinge_embedding_loss.html)|Not Support|N/A|
|[kl_div](https://pytorch.org/docs/2.1/generated/torch.nn.functional.kl_div.html)|Beta|支持数据类型：bf16、fp16、fp32|
|[l1_loss](https://pytorch.org/docs/2.1/generated/torch.nn.functional.l1_loss.html)|Stable|支持数据类型：fp16、fp32|
|[mse_loss](https://pytorch.org/docs/2.1/generated/torch.nn.functional.mse_loss.html)|Beta|入参不支持reduce、size_average；支持数据类型：fp16、fp32|
|[margin_ranking_loss](https://pytorch.org/docs/2.1/generated/torch.nn.functional.margin_ranking_loss.html)|Not Support|N/A|
|[multilabel_margin_loss](https://pytorch.org/docs/2.1/generated/torch.nn.functional.multilabel_margin_loss.html)|Not Support|N/A|
|[multilabel_soft_margin_loss](https://pytorch.org/docs/2.1/generated/torch.nn.functional.multilabel_soft_margin_loss.html)|Not Support|N/A|
|[multi_margin_loss](https://pytorch.org/docs/2.1/generated/torch.nn.functional.multi_margin_loss.html)|Not Support|N/A|
|[nll_loss](https://pytorch.org/docs/2.1/generated/torch.nn.functional.nll_loss.html)|Beta|支持数据类型：fp32|
|[huber_loss](https://pytorch.org/docs/2.1/generated/torch.nn.functional.huber_loss.html)|Not Support|N/A|
|[smooth_l1_loss](https://pytorch.org/docs/2.1/generated/torch.nn.functional.smooth_l1_loss.html)|Beta|入参不支持reduce、size_average；支持数据类型：fp16、fp32|
|[soft_margin_loss](https://pytorch.org/docs/2.1/generated/torch.nn.functional.soft_margin_loss.html)|Not Support|N/A|
|[triplet_margin_loss](https://pytorch.org/docs/2.1/generated/torch.nn.functional.triplet_margin_loss.html)|Not Support|N/A|
|[triplet_margin_with_distance_loss](https://pytorch.org/docs/2.1/generated/torch.nn.functional.triplet_margin_with_distance_loss.html)|Not Support|N/A|

## Vision functions

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[pixel_shuffle](https://pytorch.org/docs/2.1/generated/torch.nn.functional.pixel_shuffle.html)|Beta|支持数据类型：bf16、fp16、fp32、fp64、uint8、int8、int16、int32、int64、bool|
|[pixel_unshuffle](https://pytorch.org/docs/2.1/generated/torch.nn.functional.pixel_unshuffle.html)|Beta|支持数据类型：bf16、fp16、fp32、fp64、uint8、int8、int16、int32、int64、bool|
|[pad](https://pytorch.org/docs/2.1/generated/torch.nn.functional.pad.html)|Beta|value默认值torch为None，msadapter为0；<br> 属性mode为constant时，支持bf16、fp16、fp32、fp64、uint8、int8、int16、int32、int64、bool；<br> 属性mode非constant时，支持fp16、fp32、fp64|
|[interpolate](https://pytorch.org/docs/2.1/generated/torch.nn.functional.interpolate.html)|Stable|N/A|
|[upsample](https://pytorch.org/docs/2.1/generated/torch.nn.functional.upsample.html)|Not Support|N/A|
|[upsample_nearest](https://pytorch.org/docs/2.1/generated/torch.nn.functional.upsample_nearest.html)|Not Support|N/A|
|[upsample_bilinear](https://pytorch.org/docs/2.1/generated/torch.nn.functional.upsample_bilinear.html)|Not Support|N/A|
|[grid_sample](https://pytorch.org/docs/2.1/generated/torch.nn.functional.grid_sample.html)|Stable|align_corners默认值torch为None、msadapter为False|
|[affine_grid](https://pytorch.org/docs/2.1/generated/torch.nn.functional.affine_grid.html)|Not Support|N/A|
