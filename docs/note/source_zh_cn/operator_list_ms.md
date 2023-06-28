# MindSpore算子支持

`Linux` `Ascend` `GPU` `CPU` `模型开发` `初级` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.0/docs/note/source_zh_cn/operator_list_ms.md" target="_blank"><img src="./_static/logo_source.png"></a>

## mindspore.nn

| 操作名                                       | Ascend | GPU | CPU |算子类别
| :-----------                               |:------   |:------  |:-----|:---
| [mindspore.nn.Softmax](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Softmax)                                              |  Supported |  Supported |   Supported |layer/activation
| [mindspore.nn.LogSoftmax](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.LogSoftmax)                                        |  Supported | Supported | Doing |layer/activation
| [mindspore.nn.ReLU](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.ReLU)                                                    |  Supported |  Supported |   Supported |layer/activation
| [mindspore.nn.ReLU6](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.ReLU6)                                                  |Supported  |  Supported | Supported |layer/activation
| [mindspore.nn.HSwish](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.HSwish)                                                    |  Doing |  Supported |   Doing |layer/activation
| [mindspore.nn.HSigmoid](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.HSigmoid)                                                    |  Doing |  Supported |   Doing |layer/activation
| [mindspore.nn.LeakyReLU](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.LeakyReLU)                                          |  Supported |Supported | Doing |layer/activation
| [mindspore.nn.Tanh](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Tanh)                                                    |  Supported | Supported | Doing |layer/activation
| [mindspore.nn.GELU](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.GELU)                                                    |  Supported | Supported | Doing |layer/activation
| [mindspore.nn.Sigmoid](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Sigmoid)                                              |  Supported |Supported | Doing |layer/activation
| [mindspore.nn.PReLU](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.PReLU)                                                  | Supported |Doing | Doing |layer/activation
| [mindspore.nn.Dropout](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Dropout)                                              |Supported | Supported | Supported |layer/basic
| [mindspore.nn.Flatten](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Flatten)                                              |Supported |  Supported | Supported |layer/basic
| [mindspore.nn.Dense](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Dense)                                                  |Supported |  Supported | Supported |layer/basic
| [mindspore.nn.ClipByNorm](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.ClipByNorm)                                        |Supported | Supported | Doing |layer/basic
| [mindspore.nn.Norm](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Norm)                                                    |Supported | Supported | Doing |layer/basic
| [mindspore.nn.OneHot](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.OneHot)                                                |  Supported |  Supported | Supported |layer/basic
| [mindspore.nn.Range](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Range)                                                |  Supported |  Doing | Doing |layer/basic
| [mindspore.nn.SequentialCell](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.SequentialCell)                                |Supported |  Supported | Doing |layer/container
| [mindspore.nn.CellList](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.CellList)                                            |  Supported |  Supported | Doing |layer/container
| [mindspore.nn.Conv2d](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Conv2d)                                                |  Supported |  Supported |   Supported |layer/conv
| [mindspore.nn.Conv2dTranspose](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Conv2dTranspose)                              |  Supported |  Supported | Doing |layer/conv
| [mindspore.nn.Conv1d](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Conv1d)                              |  Supported |  Supported | Doing |layer/conv
| [mindspore.nn.Conv1dTranspose](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Conv1dTranspose)                              |  Supported |  Supported | Doing |layer/conv
| [mindspore.nn.Embedding](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Embedding)                                          |Supported |  Supported | Doing |layer/embedding
| [mindspore.nn.ImageGradients](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.ImageGradients)                                | Supported |Supported | Doing |layer/image
| [mindspore.nn.SSIM](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.SSIM)                                                    | Supported | Supported | Doing |layer/image
| [mindspore.nn.PSNR](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.PSNR)                                                    | Supported |Supported | Doing |layer/image
| [mindspore.nn.CentralCrop](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.CentralCrop)                                                    | Supported |Supported | Doing |layer/image
| [mindspore.nn.LSTM](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.LSTM)                                                    | Doing | Supported | Doing |layer/lstm
| [mindspore.nn.LSTMCell](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.LSTMCell)                                            | Doing | Supported | Supported |layer/lstm
| [mindspore.nn.GlobalBatchNorm](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.GlobalBatchNorm)                                      |  Supported |Doing | Doing |layer/normalization
| [mindspore.nn.BatchNorm1d](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.BatchNorm1d)                                      |  Supported |Doing | Doing |layer/normalization
| [mindspore.nn.BatchNorm2d](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.BatchNorm2d)                                      |  Supported |  Supported | Doing |layer/normalization
| [mindspore.nn.GroupNorm](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.GroupNorm)                                          |  Supported |  Doing | Doing |layer/normalization
| [mindspore.nn.LayerNorm](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.LayerNorm)                                          |  Supported | Supported | Doing |layer/normalization
| [mindspore.nn.MatrixDiag](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.MatrixDiag)                       |  Supported | Doing  | Doing | layer/normalization
| [mindspore.nn.MatrixDiagPart](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.MatrixDiagPart)                       |  Supported | Doing  | Doing | layer/normalization
| [mindspore.nn.MatrixSetDiag](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.MatrixSetDiag)                       |  Supported | Doing  | Doing | layer/normalization
| [mindspore.nn.LinSpace](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.LinSpace)                       |  Supported | Doing  | Doing | layer/normalization
| [mindspore.nn.MaxPool2d](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.MaxPool2d)                                          |  Supported |  Supported |   Supported |layer/pooling
| [mindspore.nn.AvgPool2d](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.AvgPool2d)                                          |  Supported |  Supported | Doing |layer/pooling
| [mindspore.nn.DenseBnAct](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.DenseBnAct)                                                  |Supported |  Supported | Supported |layer/quant
| [mindspore.nn.Conv2dBnAct](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Conv2dBnAct)                              |  Supported |  Supported | Supported |layer/quant
| [mindspore.nn.FakeQuantWithMinMax](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.FakeQuantWithMinMax)                              |  Supported |  Supported | Doing |layer/quant
| [mindspore.nn.Conv2dBnFoldQuant](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Conv2dBnFoldQuant)                              |  Supported |  Supported | Doing |layer/quant
| [mindspore.nn.Conv2dBnWithoutFoldQuant](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Conv2dBnWithoutFoldQuant)                              |  Supported |  Supported | Doing |layer/quant
| [mindspore.nn.Conv2dQuant](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Conv2dQuant)                              |  Supported |  Supported | Doing |layer/quant
| [mindspore.nn.DenseQuant](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.DenseQuant)                              |  Supported |  Supported | Doing |layer/quant
| [mindspore.nn.ActQuant](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.ActQuant)                              |  Supported |  Supported | Doing |layer/quant
| [mindspore.nn.LeakyReLUQuant](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.LeakyReLUQuant)                              |  Supported |  Supported | Doing |layer/quant
| [mindspore.nn.HSwishQuant](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.HSwishQuant)                              |  Doing |  Supported | Doing |layer/quant
| [mindspore.nn.HSigmoidQuant](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.HSigmoidQuant)                              |  Doing |  Supported | Doing |layer/quant
| [mindspore.nn.TensorAddQuant](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.TensorAddQuant)                              |  Supported |  Supported | Doing |layer/quant
| [mindspore.nn.MulQuant](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.MulQuant)                              |  Supported |  Supported | Doing |layer/quant
| [mindspore.nn.L1Loss](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.L1Loss)                                                |Supported |Supported | Doing |loss/loss
| [mindspore.nn.MSELoss](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.MSELoss)                                              |  Supported |Supported | Doing |loss/loss
| [mindspore.nn.SmoothL1Loss](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.SmoothL1Loss)                                    | Supported |Doing | Doing |loss/loss
| [mindspore.nn.SoftmaxCrossEntropyWithLogits](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.SoftmaxCrossEntropyWithLogits)  |  Supported |  Supported |   Supported |loss/loss
| [mindspore.nn.CosineEmbeddingLoss](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.CosineEmbeddingLoss)                                                |Supported |Supported | Doing |loss/loss
| [mindspore.nn.ProximalAdagrad](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.ProximalAdagrad)                              |  Supported |Doing | Doing |optim/ProximalAdagrad
| [mindspore.nn.LazyAdam](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.LazyAdam)                                            |  Supported |Doing | Doing |optim/lazyadam
| [mindspore.nn.Adam](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Adam)                                                    |  Supported |Doing | Doing |optim/adam
| [mindspore.nn.AdamWeightDecay](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.AdamWeightDecay)                              |  Supported | Supported | Doing |optim/adam
| [mindspore.nn.Lamb](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Lamb)                                                    |  Supported | Supported | Doing |optim/lamb
| [mindspore.nn.LARS](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.LARS)                                                    |Supported |Doing | Doing |optim/lars
| [mindspore.nn.Momentum](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Momentum)                                            |  Supported |  Supported |   Supported |optim/momentum
| [mindspore.nn.Optimizer](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Optimizer)                                          |  Supported |  Supported | Doing |optim/optimizer
| [mindspore.nn.RMSProp](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.RMSProp)                                          |  Supported | Supported | Doing |optim/optimizer
| [mindspore.nn.SGD](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.SGD)                                                      |Supported |Supported | Doing |optim/sgd
| [mindspore.nn.WithLossCell](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.WithLossCell)                                    |  Supported |  Supported | Doing |wrap/cell_wrapper
| [mindspore.nn.WithGradCell](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.WithGradCell)                                    |  Supported | Supported | Doing |wrap/cell_wrapper
| [mindspore.nn.TrainOneStepCell](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.TrainOneStepCell)                            |  Supported |  Supported | Doing |wrap/cell_wrapper
| [mindspore.nn.GetNextSingleOp](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.GetNextSingleOp)                              |Doing |  Supported | Doing |wrap/cell_wrapper
| [mindspore.nn.WithEvalCell](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.WithEvalCell)                                    |  Supported |  Supported | Doing |wrap/cell_wrapper
| [mindspore.nn.ParameterUpdate](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.ParameterUpdate)                              |  Supported |Doing | Doing |wrap/cell_wrapper
| [mindspore.nn.DistributedGradReducer](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.DistributedGradReducer)                |  Supported |Doing | Doing |wrap/grad_reducer
| [mindspore.nn.DynamicLossScaleUpdateCell](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.DynamicLossScaleUpdateCell)        | Supported |Supported | Doing |wrap/loss_scale
| [mindspore.nn.FixedLossScaleUpdateCell](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.FixedLossScaleUpdateCell)            | Supported |Supported | Doing |wrap/loss_scale
| [mindspore.nn.TrainOneStepWithLossScaleCell](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.TrainOneStepWithLossScaleCell)  | Supported |Supported | Doing |wrap/loss_scale
| [mindspore.nn.Cell](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Cell)                                                    |  Supported |  Supported |   Supported |cell
| [mindspore.nn.EmbeddingLookup](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.EmbeddingLookup)                                          |Supported |  Supported | Supported |layer/embedding
| [mindspore.nn.Pad](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Pad)                                        |Supported | Supported | Doing |layer/basic
| [mindspore.nn.MatMul](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.MatMul)                                        |Supported | Doing | Doing |layer/math
| [mindspore.nn.LGamma](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.LGamma)                                        |Supported | Doing | Doing |layer/math
| [mindspore.nn.ReduceLogSumExp](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.ReduceLogSumExp)                                        |Supported | Supported | Doing |layer/math
| [mindspore.nn.MSSSIM](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.MSSSIM)                                | Supported |Doing | Doing |layer/image

## mindspore.ops.operations

| 操作名                                       | Ascend | GPU | CPU  |算子类别
| :-----------                                 |:------   |:------  |:-----|:---
| [mindspore.ops.Flatten](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Flatten)                             |  Supported | Supported    |Supported | nn_ops
| [mindspore.ops.Softmax](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Softmax)                             |    Supported | Supported    |  Supported | nn_ops
| [mindspore.ops.Acosh](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Acosh)                                 |  Supported | Doing | Doing | nn_ops
| [mindspore.ops.FloorMod](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FloorMod)                           |  Supported | Doing | Doing | nn_ops
| [mindspore.ops.Elu](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Elu)                                     |  Supported | Doing | Doing | nn_ops
| [mindspore.ops.MirrorPad](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.MirrorPad)                                     |  Supported | Supported | Doing | nn_ops
| [mindspore.ops.Unpack](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Unpack)                                     |  Supported | Doing | Doing | nn_ops
| [mindspore.ops.Pack](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Pack)                                     |  Supported| Doing | Doing | nn_ops
| [mindspore.ops.L2Loss](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.L2Loss)                               |    Supported | Doing | Doing | nn_ops
| [mindspore.ops.CTCLoss](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.CTCLoss)                               |  Supported | Supported | Doing | nn_ops
| [mindspore.ops.RNNTLoss](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.RNNTLoss)                               |  Supported | Doing | Doing | nn_ops
| [mindspore.ops.LogSoftmax](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LogSoftmax)                       |    Supported | Supported  |Doing | nn_ops
| [mindspore.ops.Softplus](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Softplus)                       |    Supported | Doing  |Doing | nn_ops
| [mindspore.ops.ReLU](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReLU)                                   |    Supported | Supported    |  Supported | nn_ops
| [mindspore.ops.ReLU6](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReLU6)                                 |  Supported | Supported    |Supported | nn_ops
| [mindspore.ops.HSwish](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.HSwish)                                 |  Doing | Supported    |Doing | nn_ops
| [mindspore.ops.HSigmoid](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.HSigmoid)                                 |  Doing | Supported    |Doing | nn_ops
| [mindspore.ops.Sigmoid](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Sigmoid)                             |    Supported | Supported  |Doing | nn_ops
| [mindspore.ops.Tanh](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Tanh)                                   |    Supported | Supported  |Doing | nn_ops
| [mindspore.ops.BatchNorm](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BatchNorm)                         |    Supported | Doing  |Doing | nn_ops
| [mindspore.ops.LRN](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LRN)                         |    Supported | Doing  |Doing | nn_ops
| [mindspore.ops.Conv2D](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Conv2D)                               |    Supported | Supported    |  Supported | nn_ops
| [mindspore.ops.DepthwiseConv2dNative](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.DepthwiseConv2dNative) |  Supported | Doing  |Doing | nn_ops
| [mindspore.ops.DepthwiseConv2dNativeBackpropInput](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.DepthwiseConv2dNativeBackpropInput) |  Supported | Doing  |Doing | nn_ops
| [mindspore.ops.DepthwiseConv2dNativeiBackpropFilter](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.DepthwiseConv2dNativeBackpropFilter) |  Supported | Doing  |Doing | nn_ops
| [mindspore.ops.MaxPoolWithArgmax](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.MaxPoolWithArgmax)         |    Supported | Doing  |Doing | nn_ops
| [mindspore.ops.MaxPool](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.MaxPool)                             |    Supported | Supported    |  Supported | nn_ops
| [mindspore.ops.AvgPool](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.AvgPool)                             |  Supported | Supported    |Doing | nn_ops
| [mindspore.ops.Conv2DBackpropInput](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Conv2DBackpropInput)     |  Supported | Supported    |Doing | nn_ops
| [mindspore.ops.BiasAdd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BiasAdd)                             |    Supported | Supported    |  Supported | nn_ops
| [mindspore.ops.TopK](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.TopK)                                   |    Supported | Supported  |Doing | nn_ops
| [mindspore.ops.SoftmaxCrossEntropyWithLogits](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SoftmaxCrossEntropyWithLogits) |  Supported | Supported  |Doing | nn_ops
| [mindspore.ops.SparseSoftmaxCrossEntropyWithLogits](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SparseSoftmaxCrossEntropyWithLogits) |  Doing   | Supported  |  Supported | nn_ops
| [mindspore.ops.ApplyMomentum](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyMomentum)                 |    Supported  | Supported    |   Supported | nn_ops
| [mindspore.ops.ApplyAddSign](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyAddSign)                 |    Supported  | Doing    |   Doing | nn_ops
| [mindspore.ops.ApplyPowerSign](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyPowerSign)                 |    Supported  | Doing    |   Doing | nn_ops
| [mindspore.ops.ApplyGradientDescent](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyGradientDescent)                 |    Supported  | Doing    |   Doing | nn_ops
| [mindspore.ops.ApplyProximalGradientDescent](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyProximalGradientDescent)                 |    Supported  | Doing    |   Doing | nn_ops
| [mindspore.ops.ApplyRMSProp](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyRMSProp)                 |    Supported  | Supported    |   Doing | nn_ops
| [mindspore.ops.ApplyCenteredRMSProp](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyCenteredRMSProp)                 |    Supported  | Supported    |   Doing | nn_ops
| [mindspore.ops.SparseApplyAdagrad](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SparseApplyAdagrad)                 |    Supported  | Doing    |   Doing | nn_ops
| [mindspore.ops.SparseApplyAdagradV2](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SparseApplyAdagradV2)                 |    Supported  | Doing    |   Doing | nn_ops
| [mindspore.ops.SparseApplyProximalAdagrad](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SparseApplyProximalAdagrad)                 |    Supported  | Doing    |   Doing | nn_ops
| [mindspore.ops.FusedSparseProximalAdagrad](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FusedSparseProximalAdagrad)                 |    Doing  | Doing    |   Supported | nn_ops
| [mindspore.ops.ApplyProximalAdagrad](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyProximalAdagrad)                 |    Supported  | Doing    |   Doing | nn_ops
| [mindspore.ops.FusedSparseLazyAdam](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FusedSparseLazyAdam)     |  Doing |  Doing  |  Supported | nn_ops
| [mindspore.ops.FusedSparseAdam](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FusedSparseAdam)             |  Doing |  Doing  |  Supported | nn_ops
| [mindspore.ops.SmoothL1Loss](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SmoothL1Loss)                   |  Supported  | Supported  | Doing | nn_ops
| [mindspore.ops.SGD](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SGD)                                     |  Supported  | Doing  | Doing | nn_ops
| [mindspore.ops.LayerNorm](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LayerNorm)                         |    Supported  | Supported  | Doing | nn_ops
| [mindspore.ops.L2Normalize](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.L2Normalize)                     |  Supported  | Doing  | Doing | nn_ops
| [mindspore.ops.DropoutGenMask](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.DropoutGenMask)               |  Supported  | Doing  | Doing | nn_ops
| [mindspore.ops.DropoutDoMask](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.DropoutDoMask)                 |    Supported  | Doing  | Doing | nn_ops
| [mindspore.ops.ResizeBilinear](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ResizeBilinear)               |  Supported  | Doing  | Doing | nn_ops
| [mindspore.ops.OneHot](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.OneHot)                               |    Supported  |   Supported  | Supported | nn_ops
| [mindspore.ops.Gelu](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Gelu)                                   |    Supported  | Supported  | Doing | nn_ops
| [mindspore.ops.GetNext](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.GetNext)                             |    Supported  | Supported    | Doing | nn_ops
| [mindspore.ops.PReLU](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.PReLU)                                 |  Supported  | Doing  | Doing | nn_ops
| [mindspore.ops.LSTM](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LSTM)                                   |  Doing  | Supported | Supported | nn_ops
| [mindspore.ops.BasicLSTMCell](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BasicLSTMCell)                                   |  Supported  | Doing | Doing | nn_ops
| [mindspore.ops.SigmoidCrossEntropyWithLogits](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SigmoidCrossEntropyWithLogits) |  Supported | Supported  | Doing | nn_ops
| [mindspore.ops.Pad](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Pad)                                     |    Supported | Supported  | Doing | nn_ops
| [mindspore.ops.ROIAlign](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ROIAlign)                           |  Supported  | Supported  | Doing | nn_ops
| [mindspore.ops.Adam](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Adam)                                   |    Supported | Supported  | Doing | nn_ops
| [mindspore.ops.BinaryCrossEntropy](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BinaryCrossEntropy)       |  Supported | Supported  | Doing | nn_ops
| [mindspore.ops.KLDivLoss](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.KLDivLoss)       |  Doing | Supported  | Doing | nn_ops
| [mindspore.ops.LARSUpdate](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LARSUpdate)                       |  Supported | Doing  | Doing | nn_ops
| [mindspore.ops.Softsign](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Softsign)                       |  Supported | Doing  | Doing | nn_ops
| [mindspore.ops.TensorAdd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.TensorAdd)                         |    Supported   | Supported  | Supported | math_ops
| [mindspore.ops.AssignAdd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.AssignAdd)                         |    Supported   | Supported  | Supported | math_ops
| [mindspore.ops.AssignSub](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.AssignSub)                         |   Supported | Doing  | Doing | math_ops
| [mindspore.ops.ReduceMean](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReduceMean)                       |    Supported   | Supported  | Supported | math_ops
| [mindspore.ops.ReduceSum](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReduceSum)                         |    Supported | Supported    | Supported | math_ops
| [mindspore.ops.ReduceAll](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReduceAll)                         |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.ReduceMax](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReduceMax)                         |    Supported   | Supported  | Supported | math_ops
| [mindspore.ops.ReduceMin](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReduceMin)                         |  Supported | Supported  | Doing | math_ops
| [mindspore.ops.ReduceProd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReduceProd)                       |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.CumProd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.CumProd)                             |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.MatMul](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.MatMul)                               |    Supported   | Supported  |   Supported | math_ops
| [mindspore.ops.BatchMatMul](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BatchMatMul)                     |    Supported | Supported  | Doing | math_ops
| [mindspore.ops.CumSum](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.CumSum)                               |  Supported   | Supported| Doing | math_ops
| [mindspore.ops.AddN](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.AddN)                                   |    Supported   | Supported  | Supported | math_ops
| [mindspore.ops.Neg](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Neg)                                     |    Supported   | Supported  | Doing | math_ops
| [mindspore.ops.Sub](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Sub)                                     |    Supported   | Supported | Supported | math_ops
| [mindspore.ops.Mul](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Mul)                                     |    Supported   | Supported  |   Supported | math_ops
| [mindspore.ops.Square](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Square)                               |    Supported | Supported  | Supported | math_ops
| [mindspore.ops.SquareSumAll](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SquareSumAll)                               |    Supported | Doing  | Doing | math_ops
| [mindspore.ops.Rsqrt](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Rsqrt)                                 |    Supported | Supported  | Doing | math_ops
| [mindspore.ops.Sqrt](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Sqrt)                                   |    Supported | Supported | Doing | math_ops
| [mindspore.ops.Reciprocal](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Reciprocal)                       |    Supported   | Supported  | Doing | math_ops
| [mindspore.ops.Pow](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Pow)                                     |    Supported | Supported  | Doing | math_ops
| [mindspore.ops.Exp](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Exp)                                     |    Supported   | Supported  | Doing | math_ops
| [mindspore.ops.Log](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Log)                                     |    Supported   | Supported  | Doing | math_ops
| [mindspore.ops.Log1p](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Log1p)                                     |    Supported   | Doing  | Doing | math_ops
| [mindspore.ops.Minimum](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Minimum)                             |    Supported | Supported  | Doing | math_ops
| [mindspore.ops.Maximum](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Maximum)                             |    Supported | Supported  | Doing | math_ops
| [mindspore.ops.RealDiv](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.RealDiv)                             |    Supported   | Supported  | Doing | math_ops
| [mindspore.ops.Div](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Div)                                     |    Supported | Supported  | Doing | math_ops
| [mindspore.ops.DivNoNan](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.DivNoNan)                                     |    Supported | Doing  | Doing | math_ops
| [mindspore.ops.FloorDiv](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FloorDiv)                           |    Supported | Supported  | Doing | math_ops
| [mindspore.ops.Floor](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Floor)                                 |  Supported | Supported  | Doing | math_ops
| [mindspore.ops.Equal](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Equal)                                 |    Supported | Supported    | Doing | math_ops
| [mindspore.ops.EqualCount](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.EqualCount)                       |  Doing | Supported    |   Supported | math_ops
| [mindspore.ops.NotEqual](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.NotEqual)                           |  Supported | Supported  | Doing | math_ops
| [mindspore.ops.Greater](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Greater)                             |    Supported | Supported  | Doing | math_ops
| [mindspore.ops.GreaterEqual](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.GreaterEqual)                   |  Supported | Supported  | Doing | math_ops
| [mindspore.ops.Less](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Less)                                   |    Supported | Supported  | Doing | math_ops
| [mindspore.ops.Atan2](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Atan2)                                   |    Supported | Doing  | Doing | math_ops
| [mindspore.ops.LessEqual](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LessEqual)                         |    Supported | Supported  | Doing | math_ops
| [mindspore.ops.LogicalNot](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LogicalNot)                       |  Supported | Supported  | Doing | math_ops
| [mindspore.ops.LogicalAnd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LogicalAnd)                       |  Supported | Supported  | Doing | math_ops
| [mindspore.ops.LogicalOr](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LogicalOr)                         |  Supported | Supported  | Doing | math_ops
| [mindspore.ops.BitwiseAnd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BitwiseAnd)                       |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.BitwiseOr](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BitwiseOr)                       |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.BitwiseXor](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BitwiseXor)                       |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.Ceil](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Ceil)                       |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.Inv](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Inv)                       |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.Invert](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Invert)                       |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.NPUAllocFloatStatus](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.NPUAllocFloatStatus)     |    Supported | Doing  | Doing | math_ops
| [mindspore.ops.NPUGetFloatStatus](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.NPUGetFloatStatus)         |    Supported | Doing  | Doing | math_ops
| [mindspore.ops.NPUClearFloatStatus](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.NPUClearFloatStatus)     |    Supported | Doing  | Doing | math_ops
| [mindspore.ops.FloatStatus](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FloatStatus)     |    Doing | Supported  | Doing | math_ops
| [mindspore.ops.Cos](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Cos)                                     |  Supported | Supported  | Doing | math_ops
| [mindspore.ops.Cosh](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Cosh)                                     |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.ACos](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ACos)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.BesselI0e](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BesselI0e)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.BesselI1e](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BesselI1e)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.TruncateDiv](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.TruncateDiv)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.TruncateMod](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.TruncateMod)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.Tan](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Tan)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.Asin](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Asin)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.Asinh](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Asinh)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.Erf](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Erf)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.Erfc](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Erfc)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.Sin](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Sin)                                     |  Supported | Supported  | Doing | math_ops
| [mindspore.ops.Sinh](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Sinh)                                     |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.Expm1](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Expm1)                                     |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.NMSWithMask](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.NMSWithMask)                     |  Supported | Supported  | Doing | math_ops
| [mindspore.ops.Abs](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Abs)                                     |  Supported | Supported  | Doing | math_ops
| [mindspore.ops.Sign](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Sign)                                   |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.Round](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Round)                                 |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.ApproximateEqual](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApproximateEqual)                                 |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.InplaceAdd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.InplaceAdd)                                 |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.InplaceSub](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.InplaceSub)                                 |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.Mod](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Mod)                                 |  Supported | Doing  | Doing | math_ops
| [mindspore.ops.ExpandDims](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ExpandDims)                       |    Supported |   Supported  | Supported | array_ops
| [mindspore.ops.DType](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.DType)                                 |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.SameTypeShape](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SameTypeShape)                 |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.Cast](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Cast)                                   |    Supported |   Supported  | Doing | array_ops
| [mindspore.ops.IsSubClass](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.IsSubClass)                       |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.IsInstance](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.IsInstance)                       |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.Reshape](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Reshape)                             |    Supported | Supported    |   Supported | array_ops
| [mindspore.ops.Shape](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Shape)                                 |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.Squeeze](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Squeeze)                             |  Supported | Supported    | Doing | array_ops
| [mindspore.ops.Transpose](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Transpose)                         |    Supported | Supported    | Supported | array_ops
| [mindspore.ops.GatherV2](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.GatherV2)                           |    Supported | Supported    | Supported | array_ops
| [mindspore.ops.Split](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Split)                                 |    Supported | Supported  | Doing | array_ops
| [mindspore.ops.Rank](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Rank)                                   |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.Size](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Size)                                   |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.Fill](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Fill)                                   |   Supported |  Supported  |  Supported | array_ops
| [mindspore.ops.OnesLike](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.OnesLike)                           |   Supported | Supported  | Doing | array_ops
| [mindspore.ops.ZerosLike](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ZerosLike)                         |    Supported |   Supported  | Doing | array_ops
| [mindspore.ops.TupleToArray](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.TupleToArray)                   |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.ScalarToArray](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScalarToArray)                 |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.ScalarToTensor](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScalarToTensor)               |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.InvertPermutation](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.InvertPermutation)         |    Supported |   Supported  |   Supported | array_ops
| [mindspore.ops.Argmax](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Argmax)                               |  Supported | Supported    |   Supported | array_ops
| [mindspore.ops.Argmin](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Argmin)                               |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.ArgMaxWithValue](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ArgMaxWithValue)             |  Supported | Supported  | Doing | array_ops
| [mindspore.ops.ArgMinWithValue](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ArgMinWithValue)             |   Supported | Doing  | Doing | array_ops
| [mindspore.ops.Tile](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Tile)                                   |  Supported | Supported  | Doing | array_ops
| [mindspore.ops.UnsortedSegmentSum](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.UnsortedSegmentSum)       |    Supported | Supported  | Doing | array_ops
| [mindspore.ops.UnsortedSegmentMin](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.UnsortedSegmentMin)       |    Supported | Doing  | Doing | array_ops
| [mindspore.ops.UnsortedSegmentProd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.UnsortedSegmentProd)       |    Supported | Doing  | Doing | array_ops
| [mindspore.ops.Concat](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Concat)                               |    Supported |   Supported  | Supported | array_ops
| [mindspore.ops.ParallelConcat](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ParallelConcat)                               |    Supported |   Doing  | Doing | array_ops
| [mindspore.ops.Slice](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Slice)                                 |    Supported | Supported    | Supported | array_ops
| [mindspore.ops.Select](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Select)                               |    Supported | Supported  | Doing | array_ops
| [mindspore.ops.StridedSlice](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.StridedSlice)                   |    Supported | Supported    | Supported | array_ops
| [mindspore.ops.Diag](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Diag)                                   |   Doing |  Doing  |  Doing | array_ops
| [mindspore.ops.DiagPart](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.DiagPart)                                   |   Doing |  Doing  |  Doing | array_ops
| [mindspore.ops.Eye](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Eye)                                     |   Supported |  Supported  |  Supported | array_ops
| [mindspore.ops.ScatterNd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterNd)                         |    Supported | Supported  | Doing | array_ops
| [mindspore.ops.ResizeNearestNeighbor](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ResizeNearestNeighbor) |  Supported | Supported  | Doing | array_ops
| [mindspore.ops.GatherNd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.GatherNd)                           |  Supported | Supported  | Doing | array_ops
| [mindspore.ops.ApplyFtrl](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyFtrl)                           |  Supported | Supported  | Doing | array_ops
| [mindspore.ops.SparseApplyFtrl](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SparseApplyFtrl)                           |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.FusedSparseFtrl](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FusedSparseFtrl)                           |  Doing | Doing  | Supported | array_ops
| [mindspore.ops.SparseApplyFtrlV2](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SparseApplyFtrlV2)                           |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.ScatterNdUpdate](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterNdUpdate)             |  Supported | Doing  | Supported | array_ops
| [mindspore.ops.ScatterUpdate](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterUpdate)             |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.ScatterMul](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterMul)             |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.ScatterDiv](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterDiv)             |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.SpaceToDepth](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SpaceToDepth)                   |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.DepthToSpace](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.DepthToSpace)                   |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.SpaceToBatch](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SpaceToBatch)                   |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.SpaceToBatchND](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SpaceToBatchND)                   |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.BatchToSpace](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BatchToSpace)                   |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.BatchToSpaceND](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BatchToSpaceND)                   |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.IsFinite](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.IsFinite)                           |  Supported | Supported  | Doing | array_ops
| [mindspore.ops.InplaceUpdate](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.InplaceUpdate)                           |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.ScatterSub](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterSub)                           |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.ScatterMax](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterMax)                           |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.ScatterMin](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterMin)                           |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.ScatterNdAdd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterNdAdd)                           |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.ScatterNdSub](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterNdSub)                           |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.ScatterNonAliasingAdd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterNonAliasingAdd)         |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.Rint](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Rint)         |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.ReverseV2](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReverseV2)         |  Supported | Doing  | Doing | array_ops
| [mindspore.ops.ReduceOp](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReduceOp)                           |    Supported |   Supported  | Doing | comm_ops
| [mindspore.ops.AllReduce](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.AllReduce)                         |  Supported |   Supported  | Doing | comm_ops
| [mindspore.ops.AllGather](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.AllGather)                         |  Supported |   Supported  | Doing | comm_ops
| [mindspore.ops.ReduceScatter](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReduceScatter)                 |  Doing |   Supported  | Doing | comm_ops
| [mindspore.ops.Broadcast](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Broadcast)                         |  Supported | Doing  | Doing | comm_ops
| [mindspore.ops.ControlDepend](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ControlDepend)                 |    Supported |   Supported  |   Supported | control_ops
| [mindspore.ops.GeSwitch](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.GeSwitch)                           |  Doing | Doing  | Doing | control_ops
| [mindspore.ops.Merge](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Merge)                                 |  Doing | Doing  | Doing | control_ops
| [mindspore.ops.ScalarSummary](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScalarSummary)                 |  Supported |   Supported  | Supported | debug_ops
| [mindspore.ops.ImageSummary](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ImageSummary)                   |  Supported |   Supported  | Supported | debug_ops
| [mindspore.ops.TensorSummary](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.TensorSummary)                 |  Supported |   Supported  | Supported | debug_ops
| [mindspore.ops.HistogramSummary](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.HistogramSummary)                 |  Supported |   Supported  | Supported | debug_ops
| [mindspore.ops.InsertGradientOf](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.InsertGradientOf)           |    Supported |   Supported  |   Supported | debug_ops
| [mindspore.ops.Print](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Print)                                 |    Supported | Doing  | Doing | debug_ops
| [mindspore.ops.Assign](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Assign)                               |    Supported | Supported  | Doing | other_ops
| [mindspore.ops.BoundingBoxEncode](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BoundingBoxEncode)         |  Supported | Supported  | Doing | other_ops
| [mindspore.ops.BoundingBoxDecode](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BoundingBoxDecode)         |  Supported | Supported  | Doing | other_ops
| [mindspore.ops.PopulationCount](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.PopulationCount)         |  Supported | Doing  | Doing | other_ops
| [mindspore.ops.CheckValid](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.CheckValid)                       |  Supported | Supported  | Doing | other_ops
| [mindspore.ops.IOU](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.IOU)                                     |  Supported | Supported  | Doing | other_ops
| [mindspore.ops.MakeRefKey](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.MakeRefKey)                       |    Supported |   Supported  |   Supported | other_ops
| [mindspore.ops.InTopK](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.InTopK)                       |    Supported |   Doing |   Doing | other_ops
| [mindspore.ops.StandardNormal](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.StandardNormal)   |  Supported | Supported   | Doing | random_ops
| [mindspore.ops.Gamma](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Gamma)   |  Supported | Doing   | Doing | random_ops
| [mindspore.ops.Poisson](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Poisson)   |  Supported | Doing   | Doing | random_ops
| [mindspore.ops.UniformInt](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.UniformInt)   |  Supported | Supported   | Doing | random_ops
| [mindspore.ops.UniformReal](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.UniformReal)   |  Supported | Supported   | Doing | random_ops
| [mindspore.ops.RandomChoiceWithMask](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.RandomChoiceWithMask)   |  Supported| Supported   | Doing | random_ops
| [mindspore.ops.RandomCategorical](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.RandomCategorical)   |  Supported| Doing   | Doing | random_ops
| [mindspore.ops.ScalarCast](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScalarCast)                       |    Supported |   Supported  |   Supported | inner_ops
| [mindspore.ops.ReverseSequence](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReverseSequence)             |    Supported  | Doing  | Doing | array_ops
| [mindspore.ops.CropAndResize](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.CropAndResize)                 |    Supported  | Doing  | Doing | image_ops
| [mindspore.ops.SquaredDifference](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SquaredDifference)  |  Supported  | Doing  | Doing | math_ops
| [mindspore.ops.Xdivy](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Xdivy)                         |  Supported  | Doing  | Doing | math_ops
| [mindspore.ops.Xlogy](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Xlogy)                         |  Supported  | Doing  | Doing | math_ops
| [mindspore.ops.HistogramFixedWidth](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.HistogramFixedWidth)  |  Supported  | Doing  | Doing | math_ops
| [mindspore.ops.Eps](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Eps)                         |  Supported  | Supported  | Doing | math_ops
| [mindspore.ops.ReLUV2](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReLUV2)       |  Supported | Supported  | Supported | nn_ops
| [mindspore.ops.BNTrainingReduce](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BNTrainingReduce)       |  Supported | Doing  | Doing | nn_ops
| [mindspore.ops.BNTrainingUpdate](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BNTrainingUpdate)       |  Supported | Doing  | Doing | nn_ops
| [mindspore.ops.AccumulateNV2](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.AccumulateNV2)                         |  Supported  | Doing  | Doing | math_ops
| [mindspore.ops.ScatterUpdate](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterUpdate)             |    Supported  | Doing  | Doing | array_ops
| [mindspore.ops.TensorScatterUpdate](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.TensorScatterUpdate)             |    Supported  | Doing  | Doing | array_ops
| [mindspore.ops.IFMR](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.IFMR)                         |  Supported  | Doing  | Doing | math_ops
| [mindspore.ops.DynamicShape](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.DynamicShape)             |    Supported  | Supported  | Supported | array_ops
| [mindspore.ops.Unique](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Unique)             |    Doing  | Doing  | Doing | array_ops

## mindspore.ops.functional

| 操作名                | 对应functional算子
| :-----------         | :-----------
| [mindspore.ops.Pack](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Pack)    |  pack
| [mindspore.ops.TensorAdd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.TensorAdd)    |  tensor_add
| [mindspore.ops.AssignSub](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.AssignSub)    |  assign_sub
| [mindspore.ops.AddN](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.AddN)    |  addn
| [mindspore.ops.Square](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Square)    |  square
| [mindspore.ops.Sqrt](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Sqrt)    |  sqrt
| [mindspore.ops.Equal](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Equal)    |  equal
| [mindspore.ops.NotEqual](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.NotEqual)    |  not_equal
| [mindspore.ops.LogicalNot](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LogicalNot)    |  logical_not
| [mindspore.ops.LogicalAnd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LogicalAnd)    |  logical_and
| [mindspore.ops.LogicalOr](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LogicalOr)    |  logical_or
| [mindspore.ops.ExpandDims](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ExpandDims)    |  expand_dims
| [mindspore.ops.DType](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.DType)    |  dtype
| [mindspore.ops.Cast](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Cast)    |  cast
| [mindspore.ops.Reshape](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Reshape)    |  reshape
| [mindspore.ops.Shape](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Shape)    |  shape
| [mindspore.ops.GatherV2](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.GatherV2)    |  gather
| [mindspore.ops.Rank](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Rank)    |  rank
| [mindspore.ops.Size](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Size)    |  size
| [mindspore.ops.Fill](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Fill)    |  fill
| [mindspore.ops.OnesLike](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.OnesLike)    |  ones_like
| [mindspore.ops.Tile](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Tile)    |  tile
| [mindspore.ops.Select](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Select)    |  select
| [mindspore.ops.ScatterNd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterNd)    |  scatter_nd
| [mindspore.ops.GatherNd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.GatherNd)    |  gather_nd
| [mindspore.ops.ControlDepend](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ControlDepend)    |  control_depend
| [mindspore.ops.Print](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Print)    |  print
| [mindspore.ops.Assign](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Assign)    |  assign
| [mindspore.ops.Pow](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Pow)    |  tensor_pow

> 当前functional支持了一部分没有属性的算子，后续会进一步补齐完整。
