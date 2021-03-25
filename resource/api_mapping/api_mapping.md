# API 映射

由社区提供的PyTorch APIs和MindSpore APIs之间的映射。

| PyTorch APIs                                         | MindSpore APIs                                                 |  INFO  |
|------------------------------------------------------|----------------------------------------------------------------|--------|
| torch.abs                                            | mindspore.ops.Abs                                              | 功能一致 |
| torch.acos                                           | mindspore.ops.ACos                                             | 功能一致 |
| torch.add                                            | mindspore.ops.Add                                              | 功能一致 |
| torch.argmax                                         | mindspore.ops.Argmax                                           | 功能一致 |
| torch.argmin                                         | mindspore.ops.Argmin                                           | 功能一致 |
| torch.asin                                           | mindspore.ops.Asin                                             | 功能一致 |
| torch.atan                                           | mindspore.ops.Atan                                             | 功能一致 |
| torch.atan2                                          | mindspore.ops.Atan2                                            | 功能一致 |
| torch.bitwise_and                                    | mindspore.ops.BitwiseAnd                                       | 功能一致 |
| torch.bitwise_or                                     | mindspore.ops.BitwiseOr                                        | 功能一致 |
| torch.bmm                                            | mindspore.ops.BatchMatMul                                      | 功能一致 |
| torch.broadcast_tensors                              | mindspore.ops.BroadcastTo                                      |[差异对比](https://gitee.com/mindspore/docs/blob/r1.2/resource/api_mapping/BroadcastTo.md)|
| torch.cat                                            | mindspore.ops.Concat                                           | 功能一致 |
| torch.ceil                                           | mindspore.ops.Ceil                                             | 功能一致 |
| torch.chunk                                          | mindspore.ops.Split                                            | 功能一致 |
| torch.clamp                                          | mindspore.ops.clip_by_value                                    | 功能一致 |
| torch.cos                                            | mindspore.ops.Cos                                              | 功能一致 |
| torch.cosh                                           | mindspore.ops.Cosh                                             | 功能一致 |
| torch.cuda.device_count                              | mindspore.communication.get_group_size                         | 功能一致 |
| torch.cuda.set_device                                | mindspore.context.set_context                                  |[差异对比](https://gitee.com/mindspore/docs/blob/r1.2/resource/api_mapping/set_context.md)|
| torch.cumprod                                        | mindspore.ops.CumProd                                          | 功能一致 |
| torch.cumsum                                         | mindspore.ops.CumSum                                           | 功能一致 |
| torch.det                                            | mindspore.nn.MatDet                                            | 功能一致 |
| torch.diag                                           | mindspore.nn.MatrixDiag                                        |[差异对比](https://gitee.com/mindspore/docs/blob/r1.2/resource/api_mapping/MatrixDiag.md)|
| torch.digamma                                        | mindspore.nn.DiGamma                                           | 功能一致 |
| torch.distributed.all_gather                         | mindspore.ops.AllGather                                        | 功能一致 |
| torch.distributed.all_reduce                         | mindspore.ops.AllReduce                                        | 功能一致 |
| torch.distributions.gamma.Gamma                      | mindspore.ops.Gamma                                            | 功能一致 |
| torch.distributed.get_rank                           | mindspore.communication.get_rank                               | 功能一致 |
| torch.distributed.init_process_group                 | mindspore.communication.init                                   |[差异对比](https://gitee.com/mindspore/docs/blob/r1.2/resource/api_mapping/init.md)|
| torch.div                                            | mindspore.ops.Div                                              | 功能一致 |
| torch.dot                                            | mindspore.ops.tensor_dot                                       |[差异对比](https://gitee.com/mindspore/docs/blob/r1.2/resource/api_mapping/tensor_dot.md)|
| torch.eq                                             | mindspore.ops.Equal                                            | 功能一致 |
| torch.erfc                                           | mindspore.ops.Erfc                                             | 功能一致 |
| torch.exp                                            | mindspore.ops.Exp                                              | 功能一致 |
| torch.expm1                                          | mindspore.ops.Expm1                                            | 功能一致 |
| torch.eye                                            | mindspore.ops.Eye                                              | 功能一致 |
| torch.flatten                                        | mindspore.ops.Flatten                                          |[差异对比](https://gitee.com/mindspore/docs/blob/r1.2/resource/api_mapping/Flatten.md)|
| torch.flip                                           | mindspore.ops.ReverseV2                                        | 功能一致 |
| torch.floor                                          | mindspore.ops.Floor                                            | 功能一致 |
| torch.floor_divide                                   | mindspore.ops.FloorDiv                                         |[差异对比](https://gitee.com/mindspore/docs/blob/r1.2/resource/api_mapping/FloorDiv.md)|
| torch.fmod                                           | mindspore.ops.Mod                                              | 功能一致 |
| torch.gather                                         | mindspore.ops.GatherD                                          | 功能一致 |
| torch.histc                                          | mindspore.ops.HistogramFixedWidth                              | 功能一致 |
| torch.inverse                                        | mindspore.nn.MatInverse                                        | 功能一致 |
| torch.lgamma                                         | mindspore.nn.LGamma                                            | 功能一致 |
| torch.linspace                                       | mindspore.ops.LinSpace                                         | 功能一致 |
| torch.load                                           | mindspore.load_checkpoint                                      | 功能一致 |
| torch.log                                            | mindspore.ops.Log                                              | 功能一致 |
| torch.log1p                                          | mindspore.ops.Log1p                                            | 功能一致 |
| torch.logsumexp                                      | mindspore.nn.ReduceLogSumExp                                   | 功能一致 |
| torch.matmul                                         | mindspore.nn.MatMul                                            | 功能一致 |
| torch.max                                            | mindspore.ops.Maximum                                          | 功能一致 |
| torch.mean                                           | mindspore.ops.ReduceMean                                       | 功能一致 |
| torch.min                                            | mindspore.ops.Minimum                                          | 功能一致 |
| torch.mm                                             | mindspore.ops.MatMul                                           | 功能一致 |
| torch.mul                                            | mindspore.ops.Mul                                              | 功能一致 |
| torch.nn.AdaptiveAvgPool2d                           | mindspore.ops.ReduceMean                                       |[差异对比](https://gitee.com/mindspore/docs/blob/r1.2/resource/api_mapping/ReduceMean.md)|
| torch.nn.AvgPool1d                                   | mindspore.nn.AvgPool1d                                         | 功能一致 |
| torch.nn.AvgPool2d                                   | mindspore.nn.AvgPool2d                                         | 功能一致 |
| torch.nn.BatchNorm1d                                 | mindspore.nn.BatchNorm1d                                       | 功能一致 |
| torch.nn.BatchNorm2d                                 | mindspore.nn.BatchNorm2d                                       | 功能一致 |
| torch.nn.Conv2d                                      | mindspore.nn.Conv2d                                            | 功能一致 |
| torch.nn.ConvTranspose2d                             | mindspore.nn.Conv2dTranspose                                   | 功能一致 |
| torch.nn.CrossEntropyLoss                            | mindspore.nn.SoftmaxCrossEntropyWithLogits                     | 功能一致 |
| torch.nn.CTCLoss                                     | mindspore.ops.CTCLoss                                          | 功能一致 |
| torch.nn.Dropout                                     | mindspore.nn.Dropout                                           | 功能一致 |
| torch.nn.Embedding                                   | mindspore.nn.Embedding                                         | 功能一致 |
| torch.nn.Flatten                                     | mindspore.nn.Flatten                                           |[差异对比](https://gitee.com/mindspore/docs/blob/r1.2/resource/api_mapping/nn_Flatten.md)|
| torch.nn.functional.adaptive_avg_pool2d              | mindspore.nn.AvgPool2d                                         |[差异对比](https://gitee.com/mindspore/docs/blob/r1.2/resource/api_mapping/AvgPool2d.md)|
| torch.nn.functional.avg_pool2d                       | mindspore.ops.AvgPool                                          | 功能一致 |
| torch.nn.functional.binary_cross_entropy             | mindspore.ops.BinaryCrossEntropy                               | 功能一致 |
| torch.nn.functional.conv2d                           | mindspore.ops.Conv2D                                           | 功能一致 |
| torch.nn.functional.elu                              | mindspore.ops.Elu                                              | 功能一致 |
| torch.nn.functional.log_softmax                      | mindspore.nn.LogSoftmax                                        | 功能一致 |
| torch.nn.functional.normalize                        | mindspore.ops.L2Normalize                                      |[差异对比](https://gitee.com/mindspore/docs/blob/r1.2/resource/api_mapping/L2Normalize.md)|
| torch.nn.functional.one_hot                          | mindspore.ops.OneHot                                           | 功能一致 |
| torch.nn.functional.pad                              | mindspore.ops.Pad                                              | 功能一致 |
| torch.nn.functional.pixel_shuffle                    | mindspore.ops.DepthToSpace                                     | 功能一致 |
| torch.nn.functional.relu                             | mindspore.ops.ReLU                                             | 功能一致 |
| torch.nn.functional.softmax                          | mindspore.ops.Softmax                                          | 功能一致 |
| torch.nn.functional.softplus                         | mindspore.ops.Softplus                                         | 功能一致 |
| torch.nn.functional.softsign                         | mindspore.ops.Softsign                                         | 功能一致 |
| torch.nn.GELU                                        | mindspore.nn.GELU                                              | 功能一致 |
| torch.nn.GELU                                        | mindspore.nn.FastGelu                                          |[差异对比](https://gitee.com/mindspore/docs/blob/r1.2/resource/api_mapping/FastGelu.md)|
| torch.nn.GroupNorm                                   | mindspore.nn.GroupNorm                                         | 功能一致 |
| torch.nn.init.constant_                              | mindspore.common.initializer.Constant                          | 差异对比 |
| torch.nn.init.uniform_                               | mindspore.common.initializer.Uniform                           |[差异对比](https://gitee.com/mindspore/docs/blob/r1.2/resource/api_mapping/Uniform.md)|
| torch.nn.KLDivLoss                                   | mindspore.ops.KLDivLoss                                        | 功能一致 |
| torch.nn.L1Loss                                      | mindspore.nn.L1Loss                                            | 功能一致 |
| torch.nn.LayerNorm                                   | mindspore.nn.LayerNorm                                         | 功能一致 |
| torch.nn.LeakyReLU                                   | mindspore.nn.LeakyReLU                                         | 功能一致 |
| torch.nn.Linear                                      | mindspore.nn.Dense                                             | 差异对比 |
| torch.nn.LSTM                                        | mindspore.nn.LSTM                                              | 功能一致 |
| torch.nn.LSTMCell                                    | mindspore.nn.LSTMCell                                          | 功能一致 |
| torch.nn.MaxPool2d                                   | mindspore.nn.MaxPool2d                                         | 功能一致 |
| torch.nn.Module                                      | mindspore.nn.Cell                                              | 功能一致 |
| torch.nn.Module.load_state_dict                      | mindspore.load_param_into_net                                  | 功能一致 |
| torch.nn.ModuleList                                  | mindspore.nn.CellList                                          | 功能一致 |
| torch.nn.MSELoss                                     | mindspore.nn.MSELoss                                           | 功能一致 |
| torch.nn.Parameter                                   | mindspore.Parameter                                            | 功能一致 |
| torch.nn.ParameterList                               | mindspore.ParameterTuple                                       | 差异对比 |
| torch.nn.PixelShuffle                                | mindspore.ops.DepthToSpace                                     | 功能一致 |
| torch.nn.PReLU                                       | mindspore.nn.PReLU                                             | 功能一致 |
| torch.nn.ReLU                                        | mindspore.nn.ReLU                                              | 功能一致 |
| torch.nn.ReplicationPad2d                            | mindspore.nn.Pad                                               | 功能一致 |
| torch.nn.Sequential                                  | mindspore.nn.SequentialCell                                    | 功能一致 |
| torch.nn.Sigmoid                                     | mindspore.nn.Sigmoid                                           | 功能一致 |
| torch.nn.SmoothL1Loss                                | mindspore.nn.SmoothL1Loss                                      | 功能一致 |
| torch.nn.Softmax                                     | mindspore.nn.Softmax                                           | 功能一致 |
| torch.nn.SyncBatchNorm.convert_sync_batchnorm        | mindspore.nn.GlobalBatchNorm                                   | 功能一致 |
| torch.nn.Tanh                                        | mindspore.nn.Tanh                                              | 功能一致 |
| torch.nn.Unfold                                      | mindspore.nn.Unfold                                            | 功能一致 |
| torch.nn.Upsample                                    | mindspore.ops.ResizeBilinear                                   | 差异对比 |
| torch.norm                                           | mindspore.nn.Norm                                              | 差异对比 |
| torch.numel                                          | mindspore.ops.Size                                             | 功能一致 |
| torch.ones                                           | mindspore.ops.Ones                                             | 功能一致 |
| torch.ones_like                                      | mindspore.ops.OnesLike                                         | 功能一致 |
| torch.optim.Adadelta                                 | mindspore.ops.ApplyAdadelta                                    | 差异对比 |
| torch.optim.Adagrad                                  | mindspore.ops.ApplyAdagrad                                     | 差异对比 |
| torch.optim.Adam                                     | mindspore.nn.Adam                                              | 功能一致 |
| torch.optim.Adamax                                   | mindspore.ops.ApplyAdaMax                                      | 功能一致 |
| torch.optim.AdamW                                    | mindspore.nn.AdamWeightDecay                                   | 功能一致 |
| torch.optim.lr_scheduler.CosineAnnealingWarmRestarts | mindspore.nn.cosine_decay_lr                                   | 功能一致 |
| torch.optim.lr_scheduler.StepLR                      | mindspore.nn.piecewise_constant_lr                             | 功能一致 |
| torch.optim.Optimizer.step                           | mindspore.nn.TrainOneStepCell                                  | 差异对比 |
| torch.optim.RMSprop                                  | mindspore.nn.RMSProp                                           | 功能一致 |
| torch.optim.SGD                                      | mindspore.nn.SGD                                               | 功能一致 |
| torch.pow                                            | mindspore.ops.Pow                                              | 功能一致 |
| torch.prod                                           | mindspore.ops.ReduceProd                                       | 功能一致 |
| torch.rand                                           | mindspore.ops.UniformReal                                      | 功能一致 |
| torch.randint                                        | mindspore.ops.UniformInt                                       | 功能一致 |
| torch.randn                                          | mindspore.ops.StandardNormal                                   | 功能一致 |
| torch.range                                          | mindspore.nn.Range                                             | 功能一致 |
| torch.round                                          | mindspore.ops.Rint                                             | 功能一致 |
| torch.save                                           | mindspore.save_checkpoint                                      | 功能一致 |
| torch.sigmoid                                        | mindspore.ops.Sigmoid                                          | 功能一致 |
| torch.sin                                            | mindspore.ops.Sin                                              | 功能一致 |
| torch.sinh                                           | mindspore.ops.Sinh                                             | 功能一致 |
| torch.sparse.FloatTensor                             | mindspore.Tensor                                               | 差异对比 |
| torch.split                                          | mindspore.ops.Split                                            | 功能一致 |
| torch.sqrt                                           | mindspore.ops.Sqrt                                             | 功能一致 |
| torch.squeeze                                        | mindspore.ops.Squeeze                                          | 功能一致 |
| torch.stack                                          | mindspore.ops.Stack                                            | 功能一致 |
| torch.std_mean                                       | mindspore.ops.ReduceMean                                       | 差异对比 |
| torch.sum                                            | mindspore.ops.ReduceSum                                        | 功能一致 |
| torch.tan                                            | mindspore.ops.Tan                                              | 功能一致 |
| torch.tanh                                           | mindspore.ops.Tanh                                             | 功能一致 |
| torch.tensor                                         | mindspore.Tensor                                               | 功能一致 |
| torch.Tensor                                         | mindspore.Tensor                                               | 功能一致 |
| torch.Tensor.chunk                                   | mindspore.ops.Split                                            | 功能一致 |
| torch.Tensor.expand                                  | mindspore.ops.BroadcastTo                                      | 功能一致 |
| torch.Tensor.fill_                                   | mindspore.ops.Fill                                             | 功能一致 |
| torch.Tensor.float                                   | mindspore.ops.Cast                                             |[差异对比](https://gitee.com/mindspore/docs/blob/r1.2/resource/api_mapping/Cast.md)|
| torch.Tensor.index_add                               | mindspore.ops.InplaceAdd                                       | 功能一致 |
| torch.Tensor.mm                                      | mindspore.ops.MatMul                                           | 功能一致 |
| torch.Tensor.mul                                     | mindspore.ops.Mul                                              | 功能一致 |
| torch.Tensor.pow                                     | mindspore.ops.Pow                                              | 功能一致 |
| torch.Tensor.repeat                                  | mindspore.ops.Tile                                             | 功能一致 |
| torch.repeat_interleave                              | mindspore.ops.repeat_elements                                  | 功能一致 |
| torch.Tensor.requires_grad_                          | mindspore.Parameter.requires_grad                              | 功能一致 |
| torch.Tensor.round                                   | mindspore.ops.Round                                            | 功能一致 |
| torch.Tensor.scatter                                 | mindspore.ops.ScatterNd                                        | 功能一致 |
| torch.Tensor.scatter_add_                            | mindspore.ops.ScatterNdAdd                                     | 差异对比 |
| torch.Tensor.sigmoid                                 | mindspore.nn.Sigmoid                                           | 功能一致 |
| torch.Tensor.sign                                    | mindspore.ops.Sign                                             | 功能一致 |
| torch.Tensor.size                                    | mindspore.ops.Shape                                            | 功能一致 |
| torch.Tensor.sqrt                                    | mindspore.ops.Sqrt                                             | 功能一致 |
| torch.Tensor.sub                                     | mindspore.ops.Sub                                              | 功能一致 |
| torch.Tensor.t                                       | mindspore.ops.Transpose                                        | 差异对比 |
| torch.Tensor.transpose                               | mindspore.ops.Transpose                                        | 功能一致 |
| torch.Tensor.unsqueeze                               | mindspore.ops.ExpandDims                                       | 功能一致 |
| torch.Tensor.view                                    | mindspore.ops.Reshape                                          | 功能一致 |
| torch.Tensor.zero_                                   | mindspore.ops.ZerosLike                                        | 功能一致 |
| torch.transpose                                      | mindspore.ops.Transpose                                        | 功能一致 |
| torch.tril                                           | mindspore.nn.Tril                                              | 功能一致 |
| torch.triu                                           | mindspore.nn.Triu                                              | 功能一致 |
| torch.unbind                                         | mindspore.ops.Unstack                                          | 功能一致 |
| torch.unique                                         | mindspore.ops.Unique                                           | 差异对比 |
| torch.unsqueeze                                      | mindspore.ops.ExpandDims                                       | 功能一致 |
| torch.utils.data.DataLoader                          | mindspore.DatasetHelper                                        | 功能一致 |
| torch.utils.data.Dataset                             | mindspore.dataset.MindDataset                                  | 功能一致 |
| torch.utils.data.distributed.DistributedSampler      | mindspore.dataset.DistributedSampler                           | 功能一致 |
| torch.zeros                                          | mindspore.ops.Zeros                                            | 功能一致 |
| torch.zeros_like                                     | mindspore.ops.ZerosLike                                        | 功能一致 |
| torchvision.datasets.ImageFolder                     | mindspore.dataset.ImageFolderDataset                           | 功能一致 |
| torchvision.ops.nms                                  | mindspore.ops.NMSWithMask                                      | 功能一致 |
| torchvision.ops.roi_align                            | mindspore.ops.ROIAlign                                         | 功能一致 |
| torchvision.transforms.CenterCrop                    | mindspore.dataset.vision.py_transforms.CenterCrop              | 功能一致 |
| torchvision.transforms.ColorJitter                   | mindspore.dataset.vision.py_transforms.RandomColorAdjust       | 功能一致 |
| torchvision.transforms.Compose                       | mindspore.dataset.transforms.py_transforms.Compose             | 功能一致 |
| torchvision.transforms.Normalize                     | mindspore.dataset.vision.py_transforms.Normalize               | 功能一致 |
| torchvision.transforms.RandomHorizontalFlip          | mindspore.dataset.vision.py_transforms.RandomHorizontalFlip    | 功能一致 |
| torchvision.transforms.Resize                        | mindspore.dataset.vision.py_transforms.Resize                  | 功能一致 |
| torchvision.transforms.ToTensor                      | mindspore.dataset.vision.py_transforms.ToTensor                | 功能一致 |
