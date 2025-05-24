# torch.nn

## Convolution Layers

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[nn.Conv1d](https://pytorch.org/docs/2.1/generated/torch.nn.Conv1d.html)|Beta|入参不支持device|
|[nn.Conv2d](https://pytorch.org/docs/2.1/generated/torch.nn.Conv2d.html)|Beta|入参不支持device|
|[nn.Conv3d](https://pytorch.org/docs/2.1/generated/torch.nn.Conv3d.html)|Beta|入参不支持device|
|[nn.ConvTranspose1d](https://pytorch.org/docs/2.1/generated/torch.nn.ConvTranspose1d.html)|Beta|不支持out出参，入参不支持device|
|[nn.ConvTranspose2d](https://pytorch.org/docs/2.1/generated/torch.nn.ConvTranspose2d.html)|Beta|不支持out出参，入参不支持device|
|[nn.ConvTranspose3d](https://pytorch.org/docs/2.1/generated/torch.nn.ConvTranspose3d.html)|Beta|不支持out出参，入参不支持device|
|[nn.LazyConv1d](https://pytorch.org/docs/2.1/generated/torch.nn.LazyConv1d.html)|Not Support|N/A|
|[nn.LazyConv2d](https://pytorch.org/docs/2.1/generated/torch.nn.LazyConv2d.html)|Not Support|N/A|
|[nn.LazyConv3d](https://pytorch.org/docs/2.1/generated/torch.nn.LazyConv3d.html)|Not Support|N/A|
|[nn.LazyConvTranspose1d](https://pytorch.org/docs/2.1/generated/torch.nn.LazyConvTranspose1d.html)|Not Support|N/A|
|[nn.LazyConvTranspose2d](https://pytorch.org/docs/2.1/generated/torch.nn.LazyConvTranspose2d.html)|Not Support|N/A|
|[nn.LazyConvTranspose3d](https://pytorch.org/docs/2.1/generated/torch.nn.LazyConvTranspose3d.html)|Not Support|N/A|
|[nn.Unfold](https://pytorch.org/docs/2.1/generated/torch.nn.Unfold.html)|Beta|N/A|
|[nn.Fold](https://pytorch.org/docs/2.1/generated/torch.nn.Fold.html)|Beta|N/A|

## Convolution Layers

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[nn.MaxPool1d](https://pytorch.org/docs/2.1/generated/torch.nn.MaxPool1d.html)|Beta|N/A|
|[nn.MaxPool2d](https://pytorch.org/docs/2.1/generated/torch.nn.MaxPool2d.html)|Beta|N/A|
|[nn.MaxPool3d](https://pytorch.org/docs/2.1/generated/torch.nn.MaxPool3d.html)|Beta|N/A|
|[nn.MaxUnpool1d](https://pytorch.org/docs/2.1/generated/torch.nn.MaxUnpool1d.html)|Not Support|N/A|
|[nn.MaxUnpool2d](https://pytorch.org/docs/2.1/generated/torch.nn.MaxUnpool2d.html)|Not Support|N/A|
|[nn.MaxUnpool3d](https://pytorch.org/docs/2.1/generated/torch.nn.MaxUnpool3d.html)|Not Support|N/A|
|[nn.AvgPool1d](https://pytorch.org/docs/2.1/generated/torch.nn.AvgPool1d.html)|Beta|N/A|
|[nn.AvgPool2d](https://pytorch.org/docs/2.1/generated/torch.nn.AvgPool2d.html)|Beta|N/A|
|[nn.AvgPool3d](https://pytorch.org/docs/2.1/generated/torch.nn.AvgPool3d.html)|Beta|N/A|
|[nn.FractionalMaxPool2d](https://pytorch.org/docs/2.1/generated/torch.nn.FractionalMaxPool2d.html)|Not Support|N/A|
|[nn.FractionalMaxPool3d](https://pytorch.org/docs/2.1/generated/torch.nn.FractionalMaxPool3d.html)|Not Support|N/A|
|[nn.LPPool1d](https://pytorch.org/docs/2.1/generated/torch.nn.LPPool1d.html)|Not Support|N/A|
|[nn.LPPool2d](https://pytorch.org/docs/2.1/generated/torch.nn.LPPool2d.html)|Not Support|N/A|
|[nn.LPPool3d](https://pytorch.org/docs/2.1/generated/torch.nn.LPPool3d.html)|Not Support|N/A|
|[nn.AdaptiveMaxPool1d](https://pytorch.org/docs/2.1/generated/torch.nn.AdaptiveMaxPool1d.html)|Not Support|N/A|
|[nn.AdaptiveMaxPool2d](https://pytorch.org/docs/2.1/generated/torch.nn.AdaptiveMaxPool2d.html)|Not Support|N/A|
|[nn.AdaptiveMaxPool3d](https://pytorch.org/docs/2.1/generated/torch.nn.AdaptiveMaxPool3d.html)|Not Support|N/A|
|[nn.AdaptiveAvgPool1d](https://pytorch.org/docs/2.1/generated/torch.nn.AdaptiveAvgPool1d.html)|Beta|N/A|
|[nn.AdaptiveAvgPool2d](https://pytorch.org/docs/2.1/generated/torch.nn.AdaptiveAvgPool2d.html)|Beta|N/A|
|[nn.AdaptiveAvgPool3d](https://pytorch.org/docs/2.1/generated/torch.nn.AdaptiveAvgPool3d.html)|Not Support|N/A|

## Padding Layers

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[nn.ReflectionPad1d](https://pytorch.org/docs/2.1/generated/torch.nn.ReflectionPad1d.html)|Not Support|N/A|
|[nn.ReflectionPad2d](https://pytorch.org/docs/2.1/generated/torch.nn.ReflectionPad2d.html)|Not Support|N/A|
|[nn.ReflectionPad3d](https://pytorch.org/docs/2.1/generated/torch.nn.ReflectionPad3d.html)|Not Support|N/A|
|[nn.ReplicationPad1d](https://pytorch.org/docs/2.1/generated/torch.nn.ReplicationPad1d.html)|Not Support|N/A|
|[nn.ReplicationPad2d](https://pytorch.org/docs/2.1/generated/torch.nn.ReplicationPad2d.html)|Not Support|N/A|
|[nn.ReplicationPad3d](https://pytorch.org/docs/2.1/generated/torch.nn.ReplicationPad3d.html)|Not Support|N/A|
|[nn.ZeroPad1d](https://pytorch.org/docs/2.1/generated/torch.nn.ZeroPad1d.html)|Beta|N/A|
|[nn.ZeroPad2d](https://pytorch.org/docs/2.1/generated/torch.nn.ZeroPad2d.html)|Beta|N/A|
|[nn.ZeroPad3d](https://pytorch.org/docs/2.1/generated/torch.nn.ZeroPad3d.html)|Beta|N/A|
|[nn.ConstantPad1d](https://pytorch.org/docs/2.1/generated/torch.nn.ConstantPad1d.html)|Beta|N/A|
|[nn.ConstantPad2d](https://pytorch.org/docs/2.1/generated/torch.nn.ConstantPad2d.html)|Beta|N/A|
|[nn.ConstantPad3d](https://pytorch.org/docs/2.1/generated/torch.nn.ConstantPad3d.html)|Beta|N/A|
|[nn.CircularPad1d](https://pytorch.org/docs/2.1/generated/torch.nn.CircularPad1d.html)|Not Support|N/A|
|[nn.CircularPad2d](https://pytorch.org/docs/2.1/generated/torch.nn.CircularPad2d.html)|Not Support|N/A|
|[nn.CircularPad3d](https://pytorch.org/docs/2.1/generated/torch.nn.CircularPad3d.html)|Not Support|N/A|

## Non-linear Activations (weighted sum, nonlinearity)

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[nn.ELU](https://pytorch.org/docs/2.1/generated/torch.nn.ELU.html)|Not Support|N/A|
|[nn.Hardshrink](https://pytorch.org/docs/2.1/generated/torch.nn.Hardshrink.html)|Stable|N/A|
|[nn.Hardsigmoid](https://pytorch.org/docs/2.1/generated/torch.nn.Hardsigmoid.html)|Beta|N/A|
|[nn.Hardtanh](https://pytorch.org/docs/2.1/generated/torch.nn.Hardtanh.html)|Not Support|N/A|
|[nn.Hardswish](https://pytorch.org/docs/2.1/generated/torch.nn.Hardswish.html)|Beta|N/A|
|[nn.LeakyReLU](https://pytorch.org/docs/2.1/generated/torch.nn.LeakyReLU.html)|Beta|N/A|
|[nn.LogSigmoid](https://pytorch.org/docs/2.1/generated/torch.nn.LogSigmoid.html)|Not Support|N/A|
|[nn.MultiheadAttention](https://pytorch.org/docs/2.1/generated/torch.nn.MultiheadAttention.html)|Beta|入参不支持device|
|[nn.PReLU](https://pytorch.org/docs/2.1/generated/torch.nn.PReLU.html)|Beta|入参不支持device|
|[nn.ReLU](https://pytorch.org/docs/2.1/generated/torch.nn.ReLU.html)|Stable|N/A|
|[nn.ReLU6](https://pytorch.org/docs/2.1/generated/torch.nn.ReLU6.html)|Beta|N/A|
|[nn.RReLU](https://pytorch.org/docs/2.1/generated/torch.nn.RReLU.html)|Not Support|N/A|
|[nn.SELU](https://pytorch.org/docs/2.1/generated/torch.nn.SELU.html)|Beta|N/A|
|[nn.CELU](https://pytorch.org/docs/2.1/generated/torch.nn.CELU.html)|Beta|N/A|
|[nn.GELU](https://pytorch.org/docs/2.1/generated/torch.nn.GELU.html)|Stable|N/A|
|[nn.Sigmoid](https://pytorch.org/docs/2.1/generated/torch.nn.Sigmoid.html)|Beta|N/A|
|[nn.SiLU](https://pytorch.org/docs/2.1/generated/torch.nn.SiLU.html)|Stable|N/A|
|[nn.Mish](https://pytorch.org/docs/2.1/generated/torch.nn.Mish.html)|Beta|N/A|
|[nn.Softplus](https://pytorch.org/docs/2.1/generated/torch.nn.Softplus.html)|Beta|N/A|
|[nn.Softshrink](https://pytorch.org/docs/2.1/generated/torch.nn.Softshrink.html)|Not Support|N/A|
|[nn.Softsign](https://pytorch.org/docs/2.1/generated/torch.nn.Softsign.html)|Not Support|N/A|
|[nn.Tanh](https://pytorch.org/docs/2.1/generated/torch.nn.Tanh.html)|Beta|N/A|
|[nn.Tanhshrink](https://pytorch.org/docs/2.1/generated/torch.nn.Tanhshrink.html)|Not Support|N/A|
|[nn.Threshold](https://pytorch.org/docs/2.1/generated/torch.nn.Threshold.html)|Not Support|N/A|
|[nn.GLU](https://pytorch.org/docs/2.1/generated/torch.nn.GLU.html)|Beta|N/A|

## Non-linear Activations (other)

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[nn.Softmin](https://pytorch.org/docs/2.1/generated/torch.nn.Softmin.html)|Not Support|N/A|
|[nn.Softmax](https://pytorch.org/docs/2.1/generated/torch.nn.Softmax.html)|Beta|N/A|
|[nn.Softmax2d](https://pytorch.org/docs/2.1/generated/torch.nn.Softmax2d.html)|Not Support|N/A|
|[nn.LogSoftmax](https://pytorch.org/docs/2.1/generated/torch.nn.LogSoftmax.html)|Beta|N/A|
|[nn.AdaptiveLogSoftmaxWithLoss](https://pytorch.org/docs/2.1/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html)|Beta|入参不支持device|

## Normalization Layers

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[nn.BatchNorm1d](https://pytorch.org/docs/2.1/generated/torch.nn.BatchNorm1d.html)|Beta|入参不支持device、dtype|
|[nn.BatchNorm2d](https://pytorch.org/docs/2.1/generated/torch.nn.BatchNorm2d.html)|Beta|入参不支持device、dtype|
|[nn.BatchNorm3d](https://pytorch.org/docs/2.1/generated/torch.nn.BatchNorm3d.html)|Beta|入参不支持device、dtype|
|[nn.LazyBatchNorm1d](https://pytorch.org/docs/2.1/generated/torch.nn.LazyBatchNorm1d.html)|Not Support|N/A|
|[nn.LazyBatchNorm2d](https://pytorch.org/docs/2.1/generated/torch.nn.LazyBatchNorm2d.html)|Not Support|N/A|
|[nn.LazyBatchNorm3d](https://pytorch.org/docs/2.1/generated/torch.nn.LazyBatchNorm3d.html)|Not Support|N/A|
|[nn.GroupNorm](https://pytorch.org/docs/2.1/generated/torch.nn.GroupNorm.html)|Not Support|N/A|
|[nn.SyncBatchNorm](https://pytorch.org/docs/2.1/generated/torch.nn.SyncBatchNorm.html)|Not Support|N/A|
|[nn.InstanceNorm1d](https://pytorch.org/docs/2.1/generated/torch.nn.InstanceNorm1d.html)|Beta|N/A|
|[nn.InstanceNorm2d](https://pytorch.org/docs/2.1/generated/torch.nn.InstanceNorm2d.html)|Beta|N/A|
|[nn.InstanceNorm3d](https://pytorch.org/docs/2.1/generated/torch.nn.InstanceNorm3d.html)|Beta|N/A|
|[nn.LazyInstanceNorm1d](https://pytorch.org/docs/2.1/generated/torch.nn.LazyInstanceNorm1d.html)|Not Support|N/A|
|[nn.LazyInstanceNorm2d](https://pytorch.org/docs/2.1/generated/torch.nn.LazyInstanceNorm2d.html)|Not Support|N/A|
|[nn.LazyInstanceNorm3d](https://pytorch.org/docs/2.1/generated/torch.nn.LazyInstanceNorm3d.html)|Not Support|N/A|
|[nn.LayerNorm](https://pytorch.org/docs/2.1/generated/torch.nn.LayerNorm.html)|Stable|N/A|
|[nn.LocalResponseNorm](https://pytorch.org/docs/2.1/generated/torch.nn.LocalResponseNorm.html)|Not Support|N/A|
|[nn.RMSNorm](https://pytorch.org/docs/2.1/generated/torch.nn.RMSNorm.html)|Not Support|N/A|

## Recurrent Layers

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[nn.RNNBase](https://pytorch.org/docs/2.1/generated/torch.nn.RNNBase.html)|Not Support|N/A|
|[nn.RNN](https://pytorch.org/docs/2.1/generated/torch.nn.RNN.html)|Beta|N/A|
|[nn.LSTM](https://pytorch.org/docs/2.1/generated/torch.nn.LSTM.html)|Beta|N/A|
|[nn.GRU](https://pytorch.org/docs/2.1/generated/torch.nn.GRU.html)|Beta|N/A|
|[nn.RNNCell](https://pytorch.org/docs/2.1/generated/torch.nn.RNNCell.html)|Beta|入参不支持device|
|[nn.LSTMCell](https://pytorch.org/docs/2.1/generated/torch.nn.LSTMCell.html)|Beta|入参不支持device|
|[nn.GRUCell](https://pytorch.org/docs/2.1/generated/torch.nn.GRUCell.html)|Beta|入参不支持device|

## Transformer Layers

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[nn.Transformer](https://pytorch.org/docs/2.1/generated/torch.nn.Transformer.html)|Not Support|N/A|
|[nn.TransformerEncoder](https://pytorch.org/docs/2.1/generated/torch.nn.TransformerEncoder.html)|Not Support|N/A|
|[nn.TransformerDecoder](https://pytorch.org/docs/2.1/generated/torch.nn.TransformerDecoder.html)|Not Support|N/A|
|[nn.TransformerEncoderLayer](https://pytorch.org/docs/2.1/generated/torch.nn.TransformerEncoderLayer.html)|Not Support|N/A|
|[nn.TransformerDecoderLayer](https://pytorch.org/docs/2.1/generated/torch.nn.TransformerDecoderLayer.html)|Not Support|N/A|

## Linear Layers

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[nn.Identity](https://pytorch.org/docs/2.1/generated/torch.nn.Identity.html)|Stable|N/A|
|[nn.Linear](https://pytorch.org/docs/2.1/generated/torch.nn.Linear.html)|Stable|N/A|
|[nn.Bilinear](https://pytorch.org/docs/2.1/generated/torch.nn.Bilinear.html)|Not Support|N/A|
|[nn.LazyLinear](https://pytorch.org/docs/2.1/generated/torch.nn.LazyLinear.html)|Not Support|N/A|

## Dropout Layers

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[nn.Dropout](https://pytorch.org/docs/2.1/generated/torch.nn.Dropout.html)|Stable|N/A|
|[nn.Dropout1d](https://pytorch.org/docs/2.1/generated/torch.nn.Dropout1d.html)|Beta|N/A|
|[nn.Dropout2d](https://pytorch.org/docs/2.1/generated/torch.nn.Dropout2d.html)|Beta|N/A|
|[nn.Dropout3d](https://pytorch.org/docs/2.1/generated/torch.nn.Dropout3d.html)|Beta|N/A|
|[nn.AlphaDropout](https://pytorch.org/docs/2.1/generated/torch.nn.AlphaDropout.html)|Beta|N/A|
|[nn.FeatureAlphaDropout](https://pytorch.org/docs/2.1/generated/torch.nn.FeatureAlphaDropout.html)|Beta|N/A|

## Sparse Layers

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[nn.Embedding](https://pytorch.org/docs/2.1/generated/torch.nn.Embedding.html)|Beta|N/A|
|[nn.EmbeddingBag](https://pytorch.org/docs/2.1/generated/torch.nn.EmbeddingBag.html)|Not Support|N/A|

## Distance Functions

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[nn.CosineSimilarity](https://pytorch.org/docs/2.1/generated/torch.nn.CosineSimilarity.html)|Beta|N/A|
|[nn.PairwiseDistance](https://pytorch.org/docs/2.1/generated/torch.nn.PairwiseDistance.html)|Beta|N/A|

## Loss Functions

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[nn.L1Loss](https://pytorch.org/docs/2.1/generated/torch.nn.L1Loss.html)|Beta|N/A|
|[nn.MSELoss](https://pytorch.org/docs/2.1/generated/torch.nn.MSELoss.html)|Beta|N/A|
|[nn.CrossEntropyLoss](https://pytorch.org/docs/2.1/generated/torch.nn.CrossEntropyLoss.html)|Beta|N/A|
|[nn.CTCLoss](https://pytorch.org/docs/2.1/generated/torch.nn.CTCLoss.html)|Beta|N/A|
|[nn.NLLLoss](https://pytorch.org/docs/2.1/generated/torch.nn.NLLLoss.html)|Beta|N/A|
|[nn.PoissonNLLLoss](https://pytorch.org/docs/2.1/generated/torch.nn.PoissonNLLLoss.html)|Beta|N/A|
|[nn.GaussianNLLLoss](https://pytorch.org/docs/2.1/generated/torch.nn.GaussianNLLLoss.html)|Beta|N/A|
|[nn.KLDivLoss](https://pytorch.org/docs/2.1/generated/torch.nn.KLDivLoss.html)|Beta|N/A|
|[nn.BCELoss](https://pytorch.org/docs/2.1/generated/torch.nn.BCELoss.html)|Beta|N/A|
|[nn.BCEWithLogitsLoss](https://pytorch.org/docs/2.1/generated/torch.nn.BCEWithLogitsLoss.html)|Beta|N/A|
|[nn.MarginRankingLoss](https://pytorch.org/docs/2.1/generated/torch.nn.MarginRankingLoss.html)|Beta|N/A|
|[nn.HingeEmbeddingLoss](https://pytorch.org/docs/2.1/generated/torch.nn.HingeEmbeddingLoss.html)|Beta|N/A|
|[nn.MultiLabelMarginLoss](https://pytorch.org/docs/2.1/generated/torch.nn.MultiLabelMarginLoss.html)|Beta|N/A|
|[nn.HuberLoss](https://pytorch.org/docs/2.1/generated/torch.nn.HuberLoss.html)|Beta|N/A|
|[nn.SmoothL1Loss](https://pytorch.org/docs/2.1/generated/torch.nn.SmoothL1Loss.html)|Beta|N/A|
|[nn.SoftMarginLoss](https://pytorch.org/docs/2.1/generated/torch.nn.SoftMarginLoss.html)|Beta|N/A|
|[nn.MultiLabelSoftMarginLoss](https://pytorch.org/docs/2.1/generated/torch.nn.MultiLabelSoftMarginLoss.html)|Beta|N/A|
|[nn.CosineEmbeddingLoss](https://pytorch.org/docs/2.1/generated/torch.nn.CosineEmbeddingLoss.html)|Beta|N/A|
|[nn.MultiMarginLoss](https://pytorch.org/docs/2.1/generated/torch.nn.MultiMarginLoss.html)|Beta|入参不支持margin|
|[nn.TripletMarginLoss](https://pytorch.org/docs/2.1/generated/torch.nn.TripletMarginLoss.html)|Beta|N/A|
|[nn.TripletMarginWithDistanceLoss](https://pytorch.org/docs/2.1/generated/torch.nn.TripletMarginWithDistanceLoss.html)|Beta|N/A|

## Vision Layers

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[nn.PixelShuffle](https://pytorch.org/docs/2.1/generated/torch.nn.PixelShuffle.html)|Beta|N/A|
|[nn.PixelUnshuffle](https://pytorch.org/docs/2.1/generated/torch.nn.PixelUnshuffle.html)|Beta|N/A|
|[nn.Upsample](https://pytorch.org/docs/2.1/generated/torch.nn.Upsample.html)|Beta|N/A|
|[nn.UpsamplingNearest2d](https://pytorch.org/docs/2.1/generated/torch.nn.UpsamplingNearest2d.html)|Beta|N/A|
|[nn.UpsamplingBilinear2d](https://pytorch.org/docs/2.1/generated/torch.nn.UpsamplingBilinear2d.html)|Beta|N/A|

## Shuffle Layers

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[nn.ChannelShuffle](https://pytorch.org/docs/2.1/generated/torch.nn.ChannelShuffle.html)|Not Support|N/A|

## DataParallel Layers (multi-GPU, distributed)

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[nn.DataParallel](https://pytorch.org/docs/2.1/generated/torch.nn.DataParallel.html)|Not Support|N/A|
|[nn.parallel.DistributedDataParallel](https://pytorch.org/docs/2.1/generated/torch.nn.parallel.DistributedDataParallel.html)|Not Support|N/A|

## Utilities

|API名称|API状态|限制与说明|
|-------|-------|---------|
|[clip_grad_norm_](https://pytorch.org/docs/2.1/generated/torch.nn.utils.clip_grad_norm_.html)|Beta|N/A|
|[clip_grad_value_](https://pytorch.org/docs/2.1/generated/torch.nn.utils.clip_grad_value_.html)|Not Support|N/A|
|[parameters_to_vector](https://pytorch.org/docs/2.1/generated/torch.nn.utils.parameters_to_vector.html)|Not Support|N/A|
|[vector_to_parameters](https://pytorch.org/docs/2.1/generated/torch.nn.utils.vector_to_parameters.html)|Not Support|N/A|
|[prune.BasePruningMethod](https://pytorch.org/docs/2.1/generated/torch.nn.utils.prune.BasePruningMethod.html)|Not Support|N/A|
|[prune.PruningContainer](https://pytorch.org/docs/2.1/generated/torch.nn.utils.prune.PruningContainer.html)|Not Support|N/A|
|[prune.Identity](https://pytorch.org/docs/2.1/generated/torch.nn.utils.prune.Identity.html)|Beta|N/A|
|[prune.RandomUnstructured](https://pytorch.org/docs/2.1/generated/torch.nn.utils.prune.RandomUnstructured.html)|Not Support|N/A|
|[prune.L1Unstructured](https://pytorch.org/docs/2.1/generated/torch.nn.utils.prune.L1Unstructured.html)|Not Support|N/A|
|[prune.RandomStructured](https://pytorch.org/docs/2.1/generated/torch.nn.utils.prune.RandomStructured.html)|Not Support|N/A|
|[prune.LnStructured](https://pytorch.org/docs/2.1/generated/torch.nn.utils.prune.LnStructured.html)|Not Support|N/A|
|[prune.CustomFromMask](https://pytorch.org/docs/2.1/generated/torch.nn.utils.prune.CustomFromMask.html)|Not Support|N/A|
|[prune.identity](https://pytorch.org/docs/2.1/generated/torch.nn.utils.prune.identity.html)|Not Support|N/A|
|[prune.random_unstructured](https://pytorch.org/docs/2.1/generated/torch.nn.utils.prune.random_unstructured.html)|Not Support|N/A|
|[prune.l1_unstructured](https://pytorch.org/docs/2.1/generated/torch.nn.utils.prune.l1_unstructured.html)|Not Support|N/A|
|[prune.random_structured](https://pytorch.org/docs/2.1/generated/torch.nn.utils.prune.random_structured.html)|Not Support|N/A|
|[prune.ln_structured](https://pytorch.org/docs/2.1/generated/torch.nn.utils.prune.ln_structured.html)|Not Support|N/A|
|[prune.global_unstructured](https://pytorch.org/docs/2.1/generated/torch.nn.utils.prune.global_unstructured.html)|Not Support|N/A|
|[prune.custom_from_mask](https://pytorch.org/docs/2.1/generated/torch.nn.utils.prune.custom_from_mask.html)|Not Support|N/A|
|[prune.remove](https://pytorch.org/docs/2.1/generated/torch.nn.utils.prune.remove.html)|Not Support|N/A|
|[prune.is_pruned](https://pytorch.org/docs/2.1/generated/torch.nn.utils.prune.is_pruned.html)|Not Support|N/A|
|[weight_norm](https://pytorch.org/docs/2.1/generated/torch.nn.utils.weight_norm.html)|Not Support|N/A|
|[remove_weight_norm](https://docs.pytorch.org/docs/2.1/generated/torch.nn.utils.remove_weight_norm.html)|Not Support|N/A|
|[spectral_norm](https://pytorch.org/docs/2.1/generated/torch.nn.utils.spectral_norm.html)|Not Support|N/A|
|[remove_spectral_norm](https://docs.pytorch.org/docs/2.1/generated/torch.nn.utils.remove_spectral_norm.html)|Not Support|N/A|
|[skip_init](https://docs.pytorch.org/docs/2.1/generated/torch.nn.utils.skip_init.html)|Not Support|N/A|
|[parametrizations.orthogonal](https://docs.pytorch.org/docs/2.1/generated/torch.nn.utils.parametrizations.orthogonal.html)|Not Support|N/A|
|[parametrizations.spectral_norm](https://docs.pytorch.org/docs/2.1/generated/torch.nn.utils.parametrizations.spectral_norm.html)|Not Support|N/A|
|[parametrize.register_parametrization](https://docs.pytorch.org/docs/2.1/generated/torch.nn.utils.parametrize.register_parametrization.html)|Not Support|N/A|
|[parametrize.remove_parametrizations](https://docs.pytorch.org/docs/2.1/generated/torch.nn.utils.parametrize.remove_parametrizations.html)|Not Support|N/A|
|[parametrize.cached](https://docs.pytorch.org/docs/2.1/generated/torch.nn.utils.parametrize.cached.html)|Not Support|N/A|
|[parametrize.is_parametrized](https://docs.pytorch.org/docs/2.1/generated/torch.nn.utils.parametrize.is_parametrized.html)|Not Support|N/A|
|[parametrize.ParametrizationList](https://docs.pytorch.org/docs/2.1/generated/torch.nn.utils.parametrize.ParametrizationList.html)|Not Support|N/A|
|[stateless.functional_call](https://pytorch.org/docs/2.1/generated/torch.nn.utils.stateless.functional_call.html)|Not Support|N/A|
|[nn.utils.rnn.PackedSequence](https://pytorch.org/docs/2.1/generated/torch.nn.utils.rnn.PackedSequence.html)|Not Support|N/A|
|[nn.utils.rnn.pack_padded_sequence](https://pytorch.org/docs/2.1/generated/torch.nn.utils.rnn.pack_padded_sequence.html)|Not Support|N/A|
|[nn.utils.rnn.pad_packed_sequence](https://pytorch.org/docs/2.1/generated/torch.nn.utils.rnn.pad_packed_sequence.html)|Not Support|N/A|
|[nn.utils.rnn.pad_sequence](https://pytorch.org/docs/2.1/generated/torch.nn.utils.rnn.pad_sequence.html)|Not Support|N/A|
|[nn.utils.rnn.pack_sequence](https://pytorch.org/docs/2.1/generated/torch.nn.utils.rnn.pack_sequence.html)|Not Support|N/A|
|[nn.utils.rnn.unpack_sequence](https://pytorch.org/docs/2.1/generated/torch.nn.utils.rnn.unpack_sequence.html)|Not Support|N/A|
|[nn.utils.rnn.unpad_sequence](https://pytorch.org/docs/2.1/generated/torch.nn.utils.rnn.unpad_sequence.html)|Not Support|N/A|
|[nn.Flatten](https://pytorch.org/docs/2.1/generated/torch.nn.Flatten.html)|Beta|N/A|
|[nn.Unflatten](https://pytorch.org/docs/2.1/generated/torch.nn.Unflatten.html)|Beta|N/A|
