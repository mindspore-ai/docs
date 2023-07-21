# 算子支持

`Ascend` `GPU` `CPU` `模型开发` `初级` `中级` `高级`

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r0.7/docs/source_zh_cn/operator_list.md)

## mindspore.nn

| 操作名                                       | Ascend | GPU | CPU |算子类别
| :-----------                               |:------   |:------  |:-----|:---
| [mindspore.nn.Softmax](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.Softmax)                                              |  Supported |  Supported |   Supported |layer/activation
| [mindspore.nn.LogSoftmax](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.LogSoftmax)                                        |  Supported | Supported | Doing |layer/activation
| [mindspore.nn.ReLU](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.ReLU)                                                    |  Supported |  Supported |   Supported |layer/activation
| [mindspore.nn.ReLU6](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.ReLU6)                                                  |Supported  |  Supported | Supported |layer/activation
| [mindspore.nn.HSwish](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.HSwish)                                                    |  Doing |  Supported |   Doing |layer/activation
| [mindspore.nn.HSigmoid](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.HSigmoid)                                                    |  Doing |  Supported |   Doing |layer/activation
| [mindspore.nn.LeakyReLU](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.LeakyReLU)                                          |  Supported |Supported | Doing |layer/activation
| [mindspore.nn.Tanh](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.Tanh)                                                    |  Supported | Supported | Doing |layer/activation
| [mindspore.nn.GELU](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.GELU)                                                    |  Supported | Supported | Doing |layer/activation
| [mindspore.nn.Sigmoid](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.Sigmoid)                                              |  Supported |Supported | Doing |layer/activation
| [mindspore.nn.PReLU](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.PReLU)                                                  | Supported |Doing | Doing |layer/activation
| [mindspore.nn.Dropout](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.Dropout)                                              |Supported | Supported | Supported |layer/basic
| [mindspore.nn.Flatten](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.Flatten)                                              |Supported |  Supported | Supported |layer/basic
| [mindspore.nn.Dense](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.Dense)                                                  |Supported |  Supported | Supported |layer/basic
| [mindspore.nn.DenseBnAct](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.DenseBnAct)                                                  |Supported |  Doing | Supported |layer/basic
| [mindspore.nn.ClipByNorm](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.ClipByNorm)                                        |Supported | Supported | Doing |layer/basic
| [mindspore.nn.Norm](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.Norm)                                                    |Doing | Supported | Doing |layer/basic
| [mindspore.nn.OneHot](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.OneHot)                                                |  Supported |  Supported | Supported |layer/basic
| [mindspore.nn.Range](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.Range)                                                |  Supported |  Doing | Doing |layer/basic
| [mindspore.nn.SequentialCell](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.SequentialCell)                                |Supported |  Supported | Doing |layer/container
| [mindspore.nn.CellList](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.CellList)                                            |  Supported |  Supported | Doing |layer/container
| [mindspore.nn.Conv2d](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.Conv2d)                                                |  Supported |  Supported |   Supported |layer/conv
| [mindspore.nn.Conv2dTranspose](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.Conv2dTranspose)                              |  Supported |  Supported | Doing |layer/conv
| [mindspore.nn.Conv2dBnAct](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.Conv2dBnAct)                              |  Supported |  Supported | Supported |layer/conv
| [mindspore.nn.Conv1d](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.Conv1d)                              |  Supported |  Supported | Doing |layer/conv
| [mindspore.nn.Conv1dTranspose](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.Conv1dTranspose)                              |  Supported |  Supported | Doing |layer/conv
| [mindspore.nn.Embedding](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.Embedding)                                          |Supported |  Supported | Doing |layer/embedding
| [mindspore.nn.ImageGradients](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.ImageGradients)                                | Doing |Doing | Doing |layer/image
| [mindspore.nn.SSIM](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.SSIM)                                                    | Supported | Supported | Doing |layer/image
| [mindspore.nn.PSNR](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.PSNR)                                                    | Supported |Doing | Doing |layer/image
| [mindspore.nn.CentralCrop](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.CentralCrop)                                                    | Supported |Doing | Doing |layer/image
| [mindspore.nn.LSTM](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.LSTM)                                                    | Doing | Supported | Supported |layer/lstm
| [mindspore.nn.GlobalBatchNorm](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.GlobalBatchNorm)                                      |  Supported |Doing | Doing |layer/normalization
| [mindspore.nn.BatchNorm1d](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.BatchNorm1d)                                      |  Supported |Doing | Doing |layer/normalization
| [mindspore.nn.BatchNorm2d](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.BatchNorm2d)                                      |  Supported |  Supported | Doing |layer/normalization
| [mindspore.nn.GroupNorm](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.GroupNorm)                                          |  Supported |  Doing | Doing |layer/normalization
| [mindspore.nn.LayerNorm](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.LayerNorm)                                          |  Supported | Supported | Doing |layer/normalization
| [mindspore.nn.MatrixDiag](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.MatrixDiag)                       |  Supported | Doing  | Doing | layer/normalization
| [mindspore.nn.MatrixDiagPart](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.MatrixDiagPart)                       |  Supported | Doing  | Doing | layer/normalization
| [mindspore.nn.MatrixSetDiag](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.MatrixSetDiag)                       |  Supported | Doing  | Doing | layer/normalization
| [mindspore.nn.LinSpace](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.LinSpace)                       |  Supported | Doing  | Doing | layer/normalization
| [mindspore.nn.MaxPool2d](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.MaxPool2d)                                          |  Supported |  Supported |   Supported |layer/pooling
| [mindspore.nn.AvgPool2d](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.AvgPool2d)                                          |  Supported |  Supported | Doing |layer/pooling
| [mindspore.nn.L1Loss](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.L1Loss)                                                |Supported |Supported | Doing |loss/loss
| [mindspore.nn.MSELoss](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.MSELoss)                                              |  Supported |Doing | Doing |loss/loss
| [mindspore.nn.SmoothL1Loss](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.SmoothL1Loss)                                    | Supported |Doing | Doing |loss/loss
| [mindspore.nn.SoftmaxCrossEntropyWithLogits](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.SoftmaxCrossEntropyWithLogits)  |  Supported |  Supported |   Doing |loss/loss
| [mindspore.nn.SoftmaxCrossEntropyExpand](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.SoftmaxCrossEntropyExpand)          |  Supported |Supported | Doing |loss/loss
| [mindspore.nn.CosineEmbeddingLoss](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.CosineEmbeddingLoss)                                                |Supported |Supported | Doing |loss/loss
| [mindspore.nn.ProximalAdagrad](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.ProximalAdagrad)                              |  Supported |Doing | Doing |optim/ProximalAdagrad
| [mindspore.nn.LazyAdam](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.LazyAdam)                                            |  Supported |Doing | Doing |optim/lazyadam
| [mindspore.nn.Adam](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.Adam)                                                    |  Supported |Doing | Doing |optim/adam
| [mindspore.nn.AdamWeightDecay](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.AdamWeightDecay)                              |  Supported | Supported | Doing |optim/adam
| [mindspore.nn.Lamb](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.Lamb)                                                    |  Supported | Supported | Doing |optim/lamb
| [mindspore.nn.LARS](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.LARS)                                                    |Supported |Doing | Doing |optim/lars
| [mindspore.nn.Momentum](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.Momentum)                                            |  Supported |  Supported |   Supported |optim/momentum
| [mindspore.nn.Optimizer](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.Optimizer)                                          |  Supported |  Supported | Doing |optim/optimizer
| [mindspore.nn.RMSProp](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.RMSProp)                                          |  Supported | Supported | Doing |optim/optimizer
| [mindspore.nn.SGD](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.SGD)                                                      |Supported |Supported | Doing |optim/sgd
| [mindspore.nn.WithLossCell](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.WithLossCell)                                    |  Supported |  Supported | Doing |wrap/cell_wrapper
| [mindspore.nn.WithGradCell](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.WithGradCell)                                    |  Supported | Supported | Doing |wrap/cell_wrapper
| [mindspore.nn.TrainOneStepCell](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.TrainOneStepCell)                            |  Supported |  Supported | Doing |wrap/cell_wrapper
| [mindspore.nn.DataWrapper](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.DataWrapper)                                      |Doing | Supported | Doing |wrap/cell_wrapper
| [mindspore.nn.GetNextSingleOp](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.GetNextSingleOp)                              |Doing |  Supported | Doing |wrap/cell_wrapper
| [mindspore.nn.WithEvalCell](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.WithEvalCell)                                    |  Supported |  Supported | Doing |wrap/cell_wrapper
| [mindspore.nn.ParameterUpdate](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.ParameterUpdate)                              |  Supported |Doing | Doing |wrap/cell_wrapper
| [mindspore.nn.DistributedGradReducer](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.DistributedGradReducer)                |  Supported |Doing | Doing |wrap/grad_reducer
| [mindspore.nn.DynamicLossScaleUpdateCell](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.DynamicLossScaleUpdateCell)        | Doing |Doing | Doing |wrap/loss_scale
| [mindspore.nn.FixedLossScaleUpdateCell](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.FixedLossScaleUpdateCell)            | Doing |Doing | Doing |wrap/loss_scale
| [mindspore.nn.TrainOneStepWithLossScaleCell](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.TrainOneStepWithLossScaleCell)  | Doing |Doing | Doing |wrap/loss_scale
| [mindspore.nn.Cell](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.nn.html#mindspore.nn.Cell)                                                    |  Supported |  Supported |   Supported |cell

## mindspore.ops.operations

| 操作名                                       | Ascend | GPU | CPU  |算子类别
| :-----------                                 |:------   |:------  |:-----|:---
| [mindspore.ops.operations.Flatten](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Flatten)                             |  Supported | Supported    |Supported | nn_ops
| [mindspore.ops.operations.Softmax](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Softmax)                             |    Supported | Supported    |  Supported | nn_ops
| [mindspore.ops.operations.Acosh](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Acosh)                                 |  Doing | Doing | Doing | nn_ops
| [mindspore.ops.operations.FloorMod](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FloorMod)                           |  Supported | Doing | Doing | nn_ops
| [mindspore.ops.operations.Elu](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Elu)                                     |  Supported | Doing | Doing | nn_ops
| [mindspore.ops.operations.MirrorPad](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.MirrorPad)                                     |  Supported | Supported | Doing | nn_ops
| [mindspore.ops.operations.Unpack](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Unpack)                                     |  Supported | Doing | Doing | nn_ops
| [mindspore.ops.operations.Pack](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Pack)                                     |  Supported| Doing | Doing | nn_ops
| [mindspore.ops.operations.L2Loss](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.L2Loss)                               |    Supported | Doing | Doing | nn_ops
| [mindspore.ops.operations.CTCLoss](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.CTCLoss)                               |  Supported | Doing | Doing | nn_ops
| [mindspore.ops.operations.RNNTLoss](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.RNNTLoss)                               |  Supported | Doing | Doing | nn_ops
| [mindspore.ops.operations.LogSoftmax](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LogSoftmax)                       |    Supported | Supported  |Doing | nn_ops
| [mindspore.ops.operations.Softplus](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Softplus)                       |    Supported | Doing  |Doing | nn_ops
| [mindspore.ops.operations.ReLU](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReLU)                                   |    Supported | Supported    |  Supported | nn_ops
| [mindspore.ops.operations.ReLU6](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReLU6)                                 |  Supported | Supported    |Supported | nn_ops
| [mindspore.ops.operations.HSwish](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.HSwish)                                 |  Doing | Supported    |Doing | nn_ops
| [mindspore.ops.operations.HSigmoid](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.HSigmoid)                                 |  Doing | Supported    |Doing | nn_ops
| [mindspore.ops.operations.Sigmoid](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Sigmoid)                             |    Supported | Supported  |Doing | nn_ops
| [mindspore.ops.operations.Tanh](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Tanh)                                   |    Supported | Supported  |Doing | nn_ops
| [mindspore.ops.operations.BatchNorm](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BatchNorm)                         |    Supported | Doing  |Doing | nn_ops
| [mindspore.ops.operations.LRN](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LRN)                         |    Supported | Doing  |Doing | nn_ops
| [mindspore.ops.operations.Conv2D](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Conv2D)                               |    Supported | Supported    |  Supported | nn_ops
| [mindspore.ops.operations.DepthwiseConv2dNative](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.DepthwiseConv2dNative) |  Supported | Doing  |Doing | nn_ops
| [mindspore.ops.operations.DepthwiseConv2dNativeBackpropInput](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.DepthwiseConv2dNativeBackpropInput) |  Supported | Doing  |Doing | nn_ops
| [mindspore.ops.operations.DepthwiseConv2dNativeiBackpropFilter](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.DepthwiseConv2dNativeBackpropFilter) |  Supported | Doing  |Doing | nn_ops
| [mindspore.ops.operations.MaxPoolWithArgmax](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.MaxPoolWithArgmax)         |    Supported | Doing  |Doing | nn_ops
| [mindspore.ops.operations.MaxPool](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.MaxPool)                             |    Supported | Supported    |  Supported | nn_ops
| [mindspore.ops.operations.AvgPool](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.AvgPool)                             |  Supported | Supported    |Doing | nn_ops
| [mindspore.ops.operations.Conv2DBackpropInput](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Conv2DBackpropInput)     |  Supported | Supported    |Doing | nn_ops
| [mindspore.ops.operations.BiasAdd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BiasAdd)                             |    Supported | Supported    |  Supported | nn_ops
| [mindspore.ops.operations.TopK](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.TopK)                                   |    Supported | Supported  |Doing | nn_ops
| [mindspore.ops.operations.SoftmaxCrossEntropyWithLogits](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SoftmaxCrossEntropyWithLogits) |  Supported | Supported  |Doing | nn_ops
| [mindspore.ops.operations.SparseSoftmaxCrossEntropyWithLogits](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseSoftmaxCrossEntropyWithLogits) |  Doing   | Supported  |  Supported | nn_ops
| [mindspore.ops.operations.ApplyMomentum](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyMomentum)                 |    Supported  | Supported    |   Supported | nn_ops
| [mindspore.ops.operations.ApplyAddSign](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyAddSign)                 |    Supported  | Doing    |   Doing | nn_ops
| [mindspore.ops.operations.ApplyPowerSign](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyPowerSign)                 |    Supported  | Doing    |   Doing | nn_ops
| [mindspore.ops.operations.ApplyGradientDescent](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyGradientDescent)                 |    Supported  | Doing    |   Doing | nn_ops
| [mindspore.ops.operations.ApplyProximalGradientDescent](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyProximalGradientDescent)                 |    Supported  | Doing    |   Doing | nn_ops
| [mindspore.ops.operations.ApplyRMSProp](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyRMSProp)                 |    Supported  | Supported    |   Doing | nn_ops
| [mindspore.ops.operations.ApplyCenteredRMSProp](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyCenteredRMSProp)                 |    Supported  | Supported    |   Doing | nn_ops
| [mindspore.ops.operations.SparseApplyAdagrad](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyAdagrad)                 |    Supported  | Doing    |   Doing | nn_ops
| [mindspore.ops.operations.SparseApplyAdagradV2](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyAdagradV2)                 |    Supported  | Doing    |   Doing | nn_ops
| [mindspore.ops.operations.SparseApplyProximalAdagrad](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyProximalAdagrad)                 |    Supported  | Doing    |   Doing | nn_ops
| [mindspore.ops.operations.FusedSparseProximalAdagrad](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FusedSparseProximalAdagrad)                 |    Doing  | Doing    |   Supported | nn_ops
| [mindspore.ops.operations.ApplyProximalAdagrad](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyProximalAdagrad)                 |    Supported  | Doing    |   Doing | nn_ops
| [mindspore.ops.operations.FusedSparseLazyAdam](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FusedSparseLazyAdam)     |  Doing |  Doing  |  Supported | nn_ops
| [mindspore.ops.operations.FusedSparseAdam](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FusedSparseAdam)             |  Doing |  Doing  |  Supported | nn_ops
| [mindspore.ops.operations.SmoothL1Loss](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SmoothL1Loss)                   |  Supported  | Supported  | Doing | nn_ops
| [mindspore.ops.operations.SGD](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SGD)                                     |  Supported  | Doing  | Doing | nn_ops
| [mindspore.ops.operations.LayerNorm](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LayerNorm)                         |    Supported  | Supported  | Doing | nn_ops
| [mindspore.ops.operations.L2Normalize](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.L2Normalize)                     |  Supported  | Doing  | Doing | nn_ops
| [mindspore.ops.operations.DropoutGenMask](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.DropoutGenMask)               |  Supported  | Doing  | Doing | nn_ops
| [mindspore.ops.operations.DropoutDoMask](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.DropoutDoMask)                 |    Supported  | Doing  | Doing | nn_ops
| [mindspore.ops.operations.ResizeBilinear](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ResizeBilinear)               |  Supported  | Doing  | Doing | nn_ops
| [mindspore.ops.operations.OneHot](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.OneHot)                               |    Supported  |   Supported  | Supported | nn_ops
| [mindspore.ops.operations.Gelu](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Gelu)                                   |    Supported  | Supported  | Doing | nn_ops
| [mindspore.ops.operations.GetNext](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.GetNext)                             |    Supported  | Supported    | Doing | nn_ops
| [mindspore.ops.operations.PReLU](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.PReLU)                                 |  Supported  | Doing  | Doing | nn_ops
| [mindspore.ops.operations.LSTM](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LSTM)                                   |  Doing  | Supported | Supported | nn_ops
| [mindspore.ops.operations.BasicLSTMCell](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BasicLSTMCell)                                   |  Doing  | Doing | Doing | nn_ops
| [mindspore.ops.operations.SigmoidCrossEntropyWithLogits](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SigmoidCrossEntropyWithLogits) |  Supported | Supported  | Doing | nn_ops
| [mindspore.ops.operations.Pad](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Pad)                                     |    Supported | Supported  | Doing | nn_ops
| [mindspore.ops.operations.ROIAlign](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ROIAlign)                           |  Supported  | Supported  | Doing | nn_ops
| [mindspore.ops.operations.Adam](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Adam)                                   |    Supported | Supported  | Doing | nn_ops
| [mindspore.ops.operations.BinaryCrossEntropy](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BinaryCrossEntropy)       |  Supported | Supported  | Doing | nn_ops
| [mindspore.ops.operations.KLDivLoss](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.KLDivLoss)       |  Doing | Supported  | Doing | nn_ops
| [mindspore.ops.operations.LARSUpdate](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LARSUpdate)                       |  Supported | Doing  | Doing | nn_ops
| [mindspore.ops.operations.Softsign](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Softsign)                       |  Supported | Doing  | Doing | nn_ops
| [mindspore.ops.operations.TensorAdd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.TensorAdd)                         |    Supported   | Supported  | Supported | math_ops
| [mindspore.ops.operations.AssignAdd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.AssignAdd)                         |    Supported   | Supported  | Supported | math_ops
| [mindspore.ops.operations.AssignSub](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.AssignSub)                         |   Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.ReduceMean](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReduceMean)                       |    Supported   | Supported  | Supported | math_ops
| [mindspore.ops.operations.ReduceSum](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReduceSum)                         |    Supported | Supported    | Supported | math_ops
| [mindspore.ops.operations.ReduceAll](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReduceAll)                         |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.ReduceMax](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReduceMax)                         |    Supported   | Supported  | Supported | math_ops
| [mindspore.ops.operations.ReduceMin](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReduceMin)                         |  Supported | Supported  | Doing | math_ops 
| [mindspore.ops.operations.ReduceProd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReduceProd)                       |  Supported | Doing  | Doing | math_ops 
| [mindspore.ops.operations.CumProd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.CumProd)                             |  Supported | Doing  | Doing | math_ops 
| [mindspore.ops.operations.MatMul](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.MatMul)                               |    Supported   | Supported  |   Supported | math_ops
| [mindspore.ops.operations.BatchMatMul](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BatchMatMul)                     |    Supported | Supported  | Doing | math_ops
| [mindspore.ops.operations.CumSum](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.CumSum)                               |  Supported   | Supported| Doing | math_ops
| [mindspore.ops.operations.AddN](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.AddN)                                   |    Supported   | Supported  | Supported | math_ops
| [mindspore.ops.operations.Neg](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Neg)                                     |    Supported   | Supported  | Doing | math_ops
| [mindspore.ops.operations.Sub](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Sub)                                     |    Supported   | Supported | Doing | math_ops   
| [mindspore.ops.operations.Mul](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Mul)                                     |    Supported   | Supported  |   Supported | math_ops
| [mindspore.ops.operations.Square](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Square)                               |    Supported | Supported  | Doing | math_ops   
| [mindspore.ops.operations.SquareSumAll](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SquareSumAll)                               |    Supported | Doing  | Doing | math_ops   
| [mindspore.ops.operations.Rsqrt](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Rsqrt)                                 |    Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.Sqrt](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Sqrt)                                   |    Supported | Doing | Doing | math_ops
| [mindspore.ops.operations.Reciprocal](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Reciprocal)                       |    Supported   | Supported  | Doing | math_ops
| [mindspore.ops.operations.Pow](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Pow)                                     |    Supported | Supported  | Doing | math_ops
| [mindspore.ops.operations.Exp](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Exp)                                     |    Supported   | Supported  | Doing | math_ops
| [mindspore.ops.operations.Log](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Log)                                     |    Supported   | Supported  | Doing | math_ops
| [mindspore.ops.operations.Log1p](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Log1p)                                     |    Supported   | Doing  | Doing | math_ops
| [mindspore.ops.operations.Minimum](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Minimum)                             |    Supported | Supported  | Doing | math_ops 
| [mindspore.ops.operations.Maximum](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Maximum)                             |    Supported | Supported  | Doing | math_ops
| [mindspore.ops.operations.RealDiv](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.RealDiv)                             |    Supported   | Supported  | Doing | math_ops
| [mindspore.ops.operations.Div](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Div)                                     |    Supported | Supported  | Doing | math_ops
| [mindspore.ops.operations.DivNoNan](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.DivNoNan)                                     |    Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.FloorDiv](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FloorDiv)                           |    Supported | Supported  | Doing | math_ops
| [mindspore.ops.operations.Floor](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Floor)                                 |  Supported | Supported  | Doing | math_ops
| [mindspore.ops.operations.Equal](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Equal)                                 |    Supported | Supported    | Doing | math_ops
| [mindspore.ops.operations.EqualCount](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.EqualCount)                       |  Doing | Supported    |   Supported | math_ops
| [mindspore.ops.operations.NotEqual](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.NotEqual)                           |  Supported | Supported  | Doing | math_ops
| [mindspore.ops.operations.Greater](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Greater)                             |    Supported | Supported  | Doing | math_ops
| [mindspore.ops.operations.GreaterEqual](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.GreaterEqual)                   |  Supported | Supported  | Doing | math_ops
| [mindspore.ops.operations.Less](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Less)                                   |    Supported | Supported  | Doing | math_ops
| [mindspore.ops.operations.Atan2](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Atan2)                                   |    Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.LessEqual](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LessEqual)                         |    Supported | Supported  | Doing | math_ops
| [mindspore.ops.operations.LogicalNot](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LogicalNot)                       |  Supported | Supported  | Doing | math_ops
| [mindspore.ops.operations.LogicalAnd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LogicalAnd)                       |  Supported | Supported  | Doing | math_ops
| [mindspore.ops.operations.LogicalOr](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LogicalOr)                         |  Supported | Supported  | Doing | math_ops
| [mindspore.ops.operations.BitwiseAnd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BitwiseAnd)                       |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.BitwiseOr](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BitwiseOr)                       |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.BitwiseXor](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BitwiseXor)                       |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.Ceil](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Ceil)                       |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.Inv](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Inv)                       |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.Invert](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Invert)                       |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.NPUAllocFloatStatus](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.NPUAllocFloatStatus)     |    Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.NPUGetFloatStatus](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.NPUGetFloatStatus)         |    Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.NPUClearFloatStatus](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.NPUClearFloatStatus)     |    Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.FloatStatus](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FloatStatus)     |    Doing | Supported  | Doing | math_ops
| [mindspore.ops.operations.Cos](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Cos)                                     |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.Cosh](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Cosh)                                     |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.ACos](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ACos)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.BesselI0e](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BesselI0e)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.BesselI1e](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BesselI1e)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.TruncateDiv](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.TruncateDiv)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.TruncateMod](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.TruncateMod)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.Tan](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Tan)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.Asin](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Asin)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.Asinh](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Asinh)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.Erf](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Erf)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.Erfc](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Erfc)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.Sin](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Sin)                                     |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.Sinh](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Sinh)                                     |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.Expm1](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Expm1)                                     |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.NMSWithMask](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.NMSWithMask)                     |  Supported | Supported  | Doing | math_ops
| [mindspore.ops.operations.Abs](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Abs)                                     |  Supported | Supported  | Doing | math_ops
| [mindspore.ops.operations.Sign](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Sign)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.Round](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Round)                                 |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.ApproximateEqual](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApproximateEqual)                                 |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.InplaceAdd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.InplaceAdd)                                 |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.InplaceSub](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.InplaceSub)                                 |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.Mod](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Mod)                                 |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.operations.ExpandDims](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ExpandDims)                       |    Supported |   Supported  | Supported | array_ops
| [mindspore.ops.operations.DType](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.DType)                                 |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.operations.SameTypeShape](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SameTypeShape)                 |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.operations.Cast](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Cast)                                   |    Supported |   Supported  | Doing | array_ops
| [mindspore.ops.operations.IsSubClass](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.IsSubClass)                       |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.operations.IsInstance](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.IsInstance)                       |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.operations.Reshape](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Reshape)                             |    Supported | Supported    |   Supported | array_ops
| [mindspore.ops.operations.Shape](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Shape)                                 |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.operations.Squeeze](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Squeeze)                             |  Supported | Supported    | Doing | array_ops
| [mindspore.ops.operations.Transpose](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Transpose)                         |    Supported | Supported    | Supported | array_ops
| [mindspore.ops.operations.GatherV2](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.GatherV2)                           |    Supported | Supported    | Supported | array_ops
| [mindspore.ops.operations.Split](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Split)                                 |    Supported | Supported  | Doing | array_ops
| [mindspore.ops.operations.Rank](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Rank)                                   |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.operations.TruncatedNormal](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.TruncatedNormal)             |  Doing | Doing  | Doing | array_ops
| [mindspore.ops.operations.Size](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Size)                                   |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.operations.Fill](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Fill)                                   |   Supported |  Supported  |  Supported | array_ops
| [mindspore.ops.operations.OnesLike](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.OnesLike)                           |   Supported | Supported  | Doing | array_ops
| [mindspore.ops.operations.ZerosLike](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ZerosLike)                         |    Supported |   Supported  | Doing | array_ops
| [mindspore.ops.operations.TupleToArray](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.TupleToArray)                   |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.operations.ScalarToArray](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScalarToArray)                 |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.operations.ScalarToTensor](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScalarToTensor)               |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.operations.InvertPermutation](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.InvertPermutation)         |    Supported |   Supported  |   Doing | array_ops
| [mindspore.ops.operations.Argmax](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Argmax)                               |  Supported | Supported    |   Supported | array_ops
| [mindspore.ops.operations.Argmin](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Argmin)                               |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.ArgMaxWithValue](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ArgMaxWithValue)             |  Supported | Supported  | Doing | array_ops
| [mindspore.ops.operations.ArgMinWithValue](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ArgMinWithValue)             |   Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.Tile](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Tile)                                   |  Supported | Supported  | Doing | array_ops
| [mindspore.ops.operations.UnsortedSegmentSum](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.UnsortedSegmentSum)       |    Supported | Supported  | Doing | array_ops
| [mindspore.ops.operations.UnsortedSegmentMin](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.UnsortedSegmentMin)       |    Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.UnsortedSegmentProd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.UnsortedSegmentProd)       |    Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.Concat](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Concat)                               |    Supported |   Supported  | Supported | array_ops
| [mindspore.ops.operations.ParallelConcat](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ParallelConcat)                               |    Supported |   Doing  | Doing | array_ops
| [mindspore.ops.operations.Slice](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Slice)                                 |    Supported | Supported    | Supported | array_ops
| [mindspore.ops.operations.Select](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Select)                               |    Supported | Supported  | Doing | array_ops
| [mindspore.ops.operations.StridedSlice](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.StridedSlice)                   |    Supported | Supported    | Supported | array_ops
| [mindspore.ops.operations.Diag](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Diag)                                   |   Doing |  Doing  |  Doing | array_ops
| [mindspore.ops.operations.DiagPart](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.DiagPart)                                   |   Doing |  Doing  |  Doing | array_ops
| [mindspore.ops.operations.Eye](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Eye)                                     |   Supported |  Supported  |  Supported | array_ops 
| [mindspore.ops.operations.ScatterNd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterNd)                         |    Supported | Supported  | Doing | array_ops
| [mindspore.ops.operations.ResizeNearestNeighbor](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ResizeNearestNeighbor) |  Supported | Supported  | Doing | array_ops
| [mindspore.ops.operations.GatherNd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.GatherNd)                           |  Supported | Supported  | Doing | array_ops
| [mindspore.ops.operations.ApplyFtrl](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyFtrl)                           |  Supported | Supported  | Doing | array_ops
| [mindspore.ops.operations.SparseApplyFtrl](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyFtrl)                           |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.FusedSparseFtrl](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FusedSparseFtrl)                           |  Doing | Doing  | Supported | array_ops
| [mindspore.ops.operations.SparseApplyFtrlV2](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyFtrlV2)                           |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.ScatterNdUpdate](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterNdUpdate)             |  Supported | Doing  | Supported | array_ops
| [mindspore.ops.operations.ScatterUpdate](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterUpdate)             |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.ScatterMul](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterMul)             |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.ScatterDiv](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterDiv)             |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.SpaceToDepth](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SpaceToDepth)                   |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.DepthToSpace](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.DepthToSpace)                   |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.SpaceToBatch](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SpaceToBatch)                   |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.SpaceToBatchND](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SpaceToBatchND)                   |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.BatchToSpace](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BatchToSpace)                   |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.BatchToSpaceND](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BatchToSpaceND)                   |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.IsFinite](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.IsFinite)                           |  Supported | Supported  | Doing | array_ops
| [mindspore.ops.operations.InplaceUpdate](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.InplaceUpdate)                           |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.ScatterSub](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterSub)                           |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.ScatterMax](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterMax)                           |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.ScatterMin](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterMin)                           |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.ScatterNdAdd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterNdAdd)                           |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.ScatterNdSub](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterNdSub)                           |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.ScatterNonAliasingAdd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterNonAliasingAdd)         |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.Rint](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Rint)         |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.ReverseV2](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReverseV2)         |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.operations.ReduceOp](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReduceOp)                           |    Supported |   Supported  | Doing | comm_ops
| [mindspore.ops.operations.AllReduce](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.AllReduce)                         |  Supported |   Supported  | Doing | comm_ops
| [mindspore.ops.operations.AllGather](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.AllGather)                         |  Supported |   Supported  | Doing | comm_ops
| [mindspore.ops.operations.ReduceScatter](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReduceScatter)                 |  Doing |   Supported  | Doing | comm_ops
| [mindspore.ops.operations.Broadcast](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Broadcast)                         |  Supported | Doing  | Doing | comm_ops
| [mindspore.ops.operations.ControlDepend](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ControlDepend)                 |    Supported |   Supported  |   Supported | control_ops
| [mindspore.ops.operations.GeSwitch](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.GeSwitch)                           |  Doing | Doing  | Doing | control_ops
| [mindspore.ops.operations.Merge](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Merge)                                 |  Doing | Doing  | Doing | control_ops
| [mindspore.ops.operations.ScalarSummary](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScalarSummary)                 |  Supported |   Supported  | Supported | debug_ops
| [mindspore.ops.operations.ImageSummary](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ImageSummary)                   |  Supported |   Supported  | Supported | debug_ops
| [mindspore.ops.operations.TensorSummary](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.TensorSummary)                 |  Supported |   Supported  | Supported | debug_ops
| [mindspore.ops.operations.HistogramSummary](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.HistogramSummary)                 |  Supported |   Supported  | Supported | debug_ops
| [mindspore.ops.operations.InsertGradientOf](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.InsertGradientOf)           |    Supported |   Supported  |   Supported | debug_ops
| [mindspore.ops.operations.Print](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Print)                                 |    Supported | Doing  | Doing | debug_ops
| [mindspore.ops.operations.Assign](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Assign)                               |    Supported | Supported  | Doing | other_ops
| [mindspore.ops.operations.BoundingBoxEncode](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BoundingBoxEncode)         |  Supported | Supported  | Doing | other_ops
| [mindspore.ops.operations.BoundingBoxDecode](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BoundingBoxDecode)         |  Supported | Supported  | Doing | other_ops
| [mindspore.ops.operations.PopulationCount](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.PopulationCount)         |  Supported | Doing  | Doing | other_ops
| [mindspore.ops.operations.CheckValid](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.CheckValid)                       |  Supported | Supported  | Doing | other_ops
| [mindspore.ops.operations.IOU](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.IOU)                                     |  Supported | Supported  | Doing | other_ops
| [mindspore.ops.operations.MakeRefKey](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.MakeRefKey)                       |    Supported |   Supported  |   Supported | other_ops
| [mindspore.ops.operations.InTopK](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.InTopK)                       |    Supported |   Doing |   Doing | other_ops
| [mindspore.ops.operations.StandardNormal](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.StandardNormal)   |  Supported | Supported   | Doing | random_ops
| [mindspore.ops.operations.Gamma](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Gamma)   |  Supported | Doing   | Doing | random_ops
| [mindspore.ops.operations.Poisson](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Poisson)   |  Supported | Doing   | Doing | random_ops
| [mindspore.ops.operations.UniformInt](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.UniformInt)   |  Supported | Supported   | Doing | random_ops
| [mindspore.ops.operations.UniformReal](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.UniformReal)   |  Supported | Supported   | Doing | random_ops
| [mindspore.ops.operations.RandomChoiceWithMask](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.RandomChoiceWithMask)   |  Doing| Supported   | Doing | random_ops
| [mindspore.ops.operations.RandomCategorical](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.RandomCategorical)   |  Supported| Doing   | Doing | random_ops
| [mindspore.ops.operations.ScalarCast](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScalarCast)                       |    Supported |   Supported  |   Supported | inner_ops
| [mindspore.ops.operations.ReverseSequence](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReverseSequence)             |    Supported  | Doing  | Doing | array_ops
| [mindspore.ops.operations.CropAndResize](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.CropAndResize)                 |    Supported  | Doing  | Doing | image_ops
| [mindspore.ops.operations.SquaredDifference](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SquaredDifference)  |  Supported  | Doing  | Doing | math_ops
| [mindspore.ops.operations.Xdivy](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Xdivy)                         |  Supported  | Doing  | Doing | math_ops
| [mindspore.ops.operations.Xlogy](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Xlogy)                         |  Supported  | Doing  | Doing | math_ops

## mindspore.ops.functional

| 操作名                | 对应functional算子
| :-----------         | :-----------
| [mindspore.ops.operations.Pack](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Pack)    |  pack
| [mindspore.ops.operations.TensorAdd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.TensorAdd)    |  tensor_add
| [mindspore.ops.operations.AssignSub](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.AssignSub)    |  assign_sub
| [mindspore.ops.operations.AddN](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.AddN)    |  addn
| [mindspore.ops.operations.Square](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Square)    |  square
| [mindspore.ops.operations.Sqrt](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Sqrt)    |  sqrt
| [mindspore.ops.operations.Equal](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Equal)    |  equal
| [mindspore.ops.operations.NotEqual](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.NotEqual)    |  not_equal
| [mindspore.ops.operations.LogicalNot](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LogicalNot)    |  logical_not
| [mindspore.ops.operations.LogicalAnd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LogicalAnd)    |  logical_and
| [mindspore.ops.operations.LogicalOr](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LogicalOr)    |  logical_or
| [mindspore.ops.operations.ExpandDims](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ExpandDims)    |  expand_dims
| [mindspore.ops.operations.DType](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.DType)    |  dtype
| [mindspore.ops.operations.Cast](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Cast)    |  cast
| [mindspore.ops.operations.Reshape](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Reshape)    |  reshape
| [mindspore.ops.operations.Shape](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Shape)    |  shape
| [mindspore.ops.operations.GatherV2](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.GatherV2)    |  gather
| [mindspore.ops.operations.Rank](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Rank)    |  rank
| [mindspore.ops.operations.Size](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Size)    |  size
| [mindspore.ops.operations.Fill](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Fill)    |  fill
| [mindspore.ops.operations.OnesLike](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.OnesLike)    |  ones_like
| [mindspore.ops.operations.Tile](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Tile)    |  tile
| [mindspore.ops.operations.Select](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Select)    |  select
| [mindspore.ops.operations.ScatterNd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterNd)    |  scatter_nd
| [mindspore.ops.operations.GatherNd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.GatherNd)    |  gather_nd
| [mindspore.ops.operations.ControlDepend](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ControlDepend)    |  control_depend
| [mindspore.ops.operations.Print](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Print)    |  print
| [mindspore.ops.operations.Assign](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Assign)    |  assign
| [mindspore.ops.operations.Pow](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Pow)    |  tensor_pow

> 当前functional支持了一部分没有属性的算子，后续会进一步补齐完整。

## 隐式类型转换
### 转换规则
* 标量与Tensor运算：运算时，将标量自动转为Tensor，数据类型和参与运算的Tensor数据类型保持一致；
而当Tensor是bool数据类型，标量是int或float时，将标量和Tensor都转为数据类型为int32或float32的Tensor。
* 不同数据类型Tensor运算：数据类型优先级排序为bool < uint8 < int8 < int16 < int32 < int64 < float16 < float32 < float64，
运算时，先确定参与运算的Tensor中优先级相对最高的数据类型，然后将低优先级数据类型Tensor转换为相对最高优先级数据类型；
而当int8和uint8数据类型的Tensor进行运算时，将其都转为int16的Tensor。
* 不支持对Parameter进行数据类型转换：如果按照转换规则推导，需要对网络中定义的Parameter进行数据类型转换时，会抛出RuntimeError异常。

### 参与转换的数据类型
* bool
* int8
* uint8
* int16
* int32
* int64
* float16
* float32
* float64
   
### 支持算子

| 算子名
| :-----------
| [mindspore.ops.operations.Assign](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Assign)
| [mindspore.ops.operations.AssignSub](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.AssignSub)
| [mindspore.ops.operations.ApplyMomentum](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyMomentum)
| [mindspore.ops.operations.FusedSparseAdam](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FusedSparseAdam)
| [mindspore.ops.operations.FusedSparseLazyAdam](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FusedSparseLazyAdam)
| [mindspore.ops.operations.FusedSparseFtrl](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FusedSparseFtrl)
| [mindspore.ops.operations.FusedSparseProximalAdagrad](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FusedSparseProximalAdagrad)
| [mindspore.ops.operations.ApplyAdaMax](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyAdaMax)
| [mindspore.ops.operations.ApplyAdadelta](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyAdadelta)
| [mindspore.ops.operations.ApplyAdagrad](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyAdagrad)
| [mindspore.ops.operations.ApplyAdagradV2](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyAdagradV2)
| [mindspore.ops.operations.SparseApplyAdagrad](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyAdagrad)
| [mindspore.ops.operations.SparseApplyAdagradV2](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyAdagradV2)
| [mindspore.ops.operations.ApplyProximalAdagrad](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyProximalAdagrad)
| [mindspore.ops.operations.SparseApplyProximalAdagrad](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyProximalAdagrad)
| [mindspore.ops.operations.ApplyAddSign](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyAddSign)
| [mindspore.ops.operations.ApplyPowerSign](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyPowerSign)
| [mindspore.ops.operations.ApplyGradientDescent](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyGradientDescent)
| [mindspore.ops.operations.ApplyProximalGradientDescent](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyProximalGradientDescent)
| [mindspore.ops.operations.SparseApplyFtrl](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyFtrl)
| [mindspore.ops.operations.SparseApplyFtrlV2](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyFtrlV2)
| [mindspore.ops.operations.BitwiseAnd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BitwiseAnd)
| [mindspore.ops.operations.BitwiseOr](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BitwiseOr)
| [mindspore.ops.operations.BitwiseXor](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BitwiseXor)
| [mindspore.ops.operations.TensorAdd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.TensorAdd)
| [mindspore.ops.operations.Sub](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Sub)
| [mindspore.ops.operations.Mul](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Mul)
| [mindspore.ops.operations.Pow](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Pow)
| [mindspore.ops.operations.Minimum](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Minimum)
| [mindspore.ops.operations.Maximum](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Maximum)
| [mindspore.ops.operations.RealDiv](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.RealDiv)
| [mindspore.ops.operations.Div](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Div)
| [mindspore.ops.operations.DivNoNan](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.DivNoNan)
| [mindspore.ops.operations.FloorDiv](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FloorDiv)
| [mindspore.ops.operations.TruncateDiv](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.TruncateDiv)
| [mindspore.ops.operations.TruncateMod](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.TruncateMod)
| [mindspore.ops.operations.Mod](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Mod)
| [mindspore.ops.operations.FloorMod](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FloorMod)
| [mindspore.ops.operations.Atan2](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Atan2)
| [mindspore.ops.operations.SquaredDifference](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SquaredDifference)
| [mindspore.ops.operations.Xdivy](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Xdivy)
| [mindspore.ops.operations.Xlogy](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Xlogy)
| [mindspore.ops.operations.Equal](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Equal)
| [mindspore.ops.operations.ApproximateEqual](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApproximateEqual)
| [mindspore.ops.operations.NotEqual](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.NotEqual)
| [mindspore.ops.operations.Greater](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Greater)
| [mindspore.ops.operations.GreaterEqual](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.GreaterEqual)
| [mindspore.ops.operations.Less](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Less)
| [mindspore.ops.operations.LessEqual](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LessEqual)
| [mindspore.ops.operations.LogicalAnd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LogicalAnd)
| [mindspore.ops.operations.LogicalOr](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LogicalOr)
| [mindspore.ops.operations.ScatterNdUpdate](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterNdUpdate)
| [mindspore.ops.operations.ScatterNdAdd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterNdAdd)
| [mindspore.ops.operations.ScatterNdSub](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterNdSub)
| [mindspore.ops.operations.ScatterNonAliasingAdd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterNonAliasingAdd)
| [mindspore.ops.operations.ScatterUpdate](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterUpdate)
| [mindspore.ops.operations.ScatterMax](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterMax)
| [mindspore.ops.operations.ScatterMin](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterMin)
| [mindspore.ops.operations.ScatterAdd](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterAdd)
| [mindspore.ops.operations.ScatterSub](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterSub)
| [mindspore.ops.operations.ScatterMul](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterMul)
| [mindspore.ops.operations.ScatterDiv](https://www.mindspore.cn/api/zh-CN/r0.7/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterDiv)

