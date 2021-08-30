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
| torch.broadcast_tensors                              | mindspore.ops.BroadcastTo                                      |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/BroadcastTo.md)|
| torch.cat                                            | mindspore.ops.Concat                                           | 功能一致 |
| torch.ceil                                           | mindspore.ops.Ceil                                             | 功能一致 |
| torch.chunk                                          | mindspore.ops.Split                                            | 功能一致 |
| torch.clamp                                          | mindspore.ops.clip_by_value                                    | 功能一致 |
| torch.cos                                            | mindspore.ops.Cos                                              | 功能一致 |
| torch.cosh                                           | mindspore.ops.Cosh                                             | 功能一致 |
| torch.cuda.device_count                              | mindspore.communication.get_group_size                         | 功能一致 |
| torch.cuda.set_device                                | mindspore.context.set_context                                  |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/set_context.md)|
| torch.cumprod                                        | mindspore.ops.CumProd                                          | 功能一致 |
| torch.cumsum                                         | mindspore.ops.CumSum                                           | 功能一致 |
| torch.det                                            | mindspore.nn.MatDet                                            | 功能一致 |
| torch.diag                                           | mindspore.nn.MatrixDiag                                        |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/MatrixDiag.md)|
| torch.digamma                                        | mindspore.nn.DiGamma                                           | 功能一致 |
| torch.distributed.all_gather                         | mindspore.ops.AllGather                                        | 功能一致 |
| torch.distributed.all_reduce                         | mindspore.ops.AllReduce                                        | 功能一致 |
| torch.distributions.gamma.Gamma                      | mindspore.ops.Gamma                                            | 功能一致 |
| torch.distributed.get_rank                           | mindspore.communication.get_rank                               | 功能一致 |
| torch.distributed.init_process_group                 | mindspore.communication.init                                   |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/init.md)|
| torch.distributed.new_group                          | mindspore.communication.create_group                           |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/create_group.md)|
| torch.div                                            | mindspore.ops.Div                                              | 功能一致 |
| torch.dot                                            | mindspore.ops.tensor_dot                                       |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/tensor_dot.md)|
| torch.eq                                             | mindspore.ops.Equal                                            | 功能一致 |
| torch.erfc                                           | mindspore.ops.Erfc                                             | 功能一致 |
| torch.exp                                            | mindspore.ops.Exp                                              | 功能一致 |
| torch.expm1                                          | mindspore.ops.Expm1                                            | 功能一致 |
| torch.eye                                            | mindspore.ops.Eye                                              | 功能一致 |
| torch.flatten                                        | mindspore.ops.Flatten                                          |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/Flatten.md)|
| torch.flip                                           | mindspore.ops.ReverseV2                                        | 功能一致 |
| torch.floor                                          | mindspore.ops.Floor                                            | 功能一致 |
| torch.floor_divide                                   | mindspore.ops.FloorDiv                                         |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/FloorDiv.md)|
| torch.fmod                                           | mindspore.ops.Mod                                              | 功能一致 |
| torch.gather                                         | mindspore.ops.GatherD                                          | 功能一致 |
| torch.gt                                             | mindspore.ops.Greater                                          | 功能一致 |
| torch.histc                                          | mindspore.ops.HistogramFixedWidth                              | 功能一致 |
| torch.inverse                                        | mindspore.nn.MatInverse                                        | 功能一致 |
| torch.lgamma                                         | mindspore.nn.LGamma                                            | 功能一致 |
| torch.linspace                                       | mindspore.ops.LinSpace                                         | 功能一致 |
| torch.load                                           | mindspore.load_checkpoint                                      | 功能一致 |
| torch.log                                            | mindspore.ops.Log                                              | 功能一致 |
| torch.log1p                                          | mindspore.ops.Log1p                                            | 功能一致 |
| torch.logsumexp                                      | mindspore.nn.ReduceLogSumExp                                   | 功能一致 |
| torch.matmul                                         | mindspore.nn.MatMul                                            | 功能一致 |
| torch.max                                            | mindspore.ops.ArgMaxWithValue                                  |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/ArgMaxWithValue.md)|
| torch.maximum                                        | mindspore.ops.Maximum                                          | 功能一致 |
| torch.mean                                           | mindspore.ops.ReduceMean                                       | 功能一致 |
| torch.min                                            | mindspore.ops.ArgMinWithValue                                  |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/ArgMinWithValue.md)|
| torch.minimum                                        | mindspore.ops.Minimum                                          | 功能一致 |
| torch.mm                                             | mindspore.ops.MatMul                                           | 功能一致 |
| torch.mul                                            | mindspore.ops.Mul                                              | 功能一致 |
| torch.nn.AdaptiveAvgPool2d                           | mindspore.ops.AdaptiveAvgPool2D                                | 功能一致 |
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
| torch.nn.Flatten                                     | mindspore.nn.Flatten                                           |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/nn_Flatten.md)|
| torch.nn.functional.adaptive_avg_pool2d              | mindspore.ops.AdaptiveAvgPool2D                                | 功能一致 |
| torch.nn.functional.avg_pool2d                       | mindspore.ops.AvgPool                                          | 功能一致 |
| torch.nn.functional.binary_cross_entropy             | mindspore.ops.BinaryCrossEntropy                               | 功能一致 |
| torch.nn.functional.conv2d                           | mindspore.ops.Conv2D                                           | 功能一致 |
| torch.nn.functional.elu                              | mindspore.ops.Elu                                              | 功能一致 |
| torch.nn.functional.log_softmax                      | mindspore.nn.LogSoftmax                                        | 功能一致 |
| torch.nn.functional.normalize                        | mindspore.ops.L2Normalize                                      |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/L2Normalize.md)|
| torch.nn.functional.one_hot                          | mindspore.ops.OneHot                                           | 功能一致 |
| torch.nn.functional.pad                              | mindspore.ops.Pad                                              | 功能一致 |
| torch.nn.functional.pixel_shuffle                    | mindspore.ops.DepthToSpace                                     | 功能一致 |
| torch.nn.functional.relu                             | mindspore.ops.ReLU                                             | 功能一致 |
| torch.nn.functional.softmax                          | mindspore.ops.Softmax                                          | 功能一致 |
| torch.nn.functional.softplus                         | mindspore.ops.Softplus                                         | 功能一致 |
| torch.nn.functional.softsign                         | mindspore.ops.Softsign                                         | 功能一致 |
| torch.nn.GELU                                        | mindspore.nn.GELU                                              | 功能一致 |
| torch.nn.GELU                                        | mindspore.nn.FastGelu                                          |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/FastGelu.md)|
| torch.nn.GroupNorm                                   | mindspore.nn.GroupNorm                                         | 功能一致 |
| torch.nn.init.constant_                              | mindspore.common.initializer.Constant                          |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/Constant.md)|
| torch.nn.init.uniform_                               | mindspore.common.initializer.Uniform                           |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/Uniform.md)|
| torch.nn.KLDivLoss                                   | mindspore.ops.KLDivLoss                                        | 功能一致 |
| torch.nn.L1Loss                                      | mindspore.nn.L1Loss                                            | 功能一致 |
| torch.nn.LayerNorm                                   | mindspore.nn.LayerNorm                                         | 功能一致 |
| torch.nn.LeakyReLU                                   | mindspore.nn.LeakyReLU                                         | 功能一致 |
| torch.nn.Linear                                      | mindspore.nn.Dense                                             |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/Dense.md)|
| torch.nn.LSTM                                        | mindspore.nn.LSTM                                              | 功能一致 |
| torch.nn.LSTMCell                                    | mindspore.nn.LSTMCell                                          | 功能一致 |
| torch.nn.MaxPool2d                                   | mindspore.nn.MaxPool2d                                         | 功能一致 |
| torch.nn.Module                                      | mindspore.nn.Cell                                              | 功能一致 |
| torch.nn.Module.add_module                                      | mindspore.nn.Cell.insert_child_to_cell                          | 功能一致 |
| torch.nn.Module.buffers                     | mindspore.nn.Cell.untrainable_params                              |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/UnTrainableParams.md)|
| torch.nn.Module.children                                      | mindspore.nn.Cell.cells                          |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/Cells.md)|
| torch.nn.Module.load_state_dict                      | mindspore.load_param_into_net                                  | 功能一致 |
| torch.nn.Module.named_children                     | mindspore.nn.Cell.name_cells                              [差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/NameCells.md)|
| torch.nn.Module.named_modules                      | mindspore.nn.Cell.cells_and_names                              | 功能一致 |
| torch.nn.Module.parameters                     | mindspore.nn.Cell.trainable_params                              |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/trainableParams.md)|
| torch.nn.Module.state_dict                      | mindspore.nn.Cell.parameters_dict                              | 功能一致 |
| torch.nn.Module.train                      | mindspore.nn.Cell.set_train                              | 功能一致 |
| torch.nn.ModuleList                                  | mindspore.nn.CellList                                          | 功能一致 |
| torch.nn.MSELoss                                     | mindspore.nn.MSELoss                                           | 功能一致 |
| torch.nn.Parameter                                   | mindspore.Parameter                                            | 功能一致 |
| torch.nn.Parameter.clone                                  | mindspore.Parameter.clone                                            | 功能一致 |
| torch.nn.Parameter.data                                  | mindspore.Parameter.data                                            |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/ParamsData.md)|
| torch.nn.ParameterList                               | mindspore.ParameterTuple                                       |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/ParameterTuple.md)|
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
| torch.nn.Upsample                                    | mindspore.ops.ResizeBilinear                                   |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/ResizeBilinear.md)|
| torch.norm                                           | mindspore.nn.Norm                                              |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/Norm.md)|
| torch.numel                                          | mindspore.ops.Size                                             | 功能一致 |
| torch.ones                                           | mindspore.ops.Ones                                             | 功能一致 |
| torch.ones_like                                      | mindspore.ops.OnesLike                                         | 功能一致 |
| torch.optim.Adadelta                                 | mindspore.ops.ApplyAdadelta                                    |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/ApplyAdadelta.md)|
| torch.optim.Adagrad                                  | mindspore.nn.Adagrad                                           |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/Adagrad.md)|
| torch.optim.Adam                                     | mindspore.nn.Adam                                              | 功能一致 |
| torch.optim.Adamax                                   | mindspore.ops.ApplyAdaMax                                      | 功能一致 |
| torch.optim.AdamW                                    | mindspore.nn.AdamWeightDecay                                   | 功能一致 |
| torch.optim.lr_scheduler.CosineAnnealingWarmRestarts | mindspore.nn.cosine_decay_lr                                   | 功能一致 |
| torch.optim.lr_scheduler.ExponentialLR        | mindspore.nn.exponential_decay_lr                                    |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/ExponentialDecayLR.md)|
| torch.optim.lr_scheduler.MultiStepLR        | mindspore.nn.piecewise_constant_lr                                  |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/PiecewiseConstantLR.md)|
| torch.optim.lr_scheduler.StepLR                      | mindspore.nn.piecewise_constant_lr                             | 功能一致 |
| torch.optim.Optimizer  | mindspore.nn.Optimizer                                    |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/Optimizer.md)|
| torch.optim.Optimizer.step                           | mindspore.nn.TrainOneStepCell                                  |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/TrainOneStepCell.md)|
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
| torch.split                                          | mindspore.ops.Split                                            | 功能一致 |
| torch.sqrt                                           | mindspore.ops.Sqrt                                             | 功能一致 |
| torch.squeeze                                        | mindspore.ops.Squeeze                                          | 功能一致 |
| torch.stack                                          | mindspore.ops.Stack                                            | 功能一致 |
| torch.std_mean                                       | mindspore.ops.ReduceMean                                       |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/ReduceMean&std_mean.md)|
| torch.sum                                            | mindspore.ops.ReduceSum                                        | 功能一致 |
| torch.tan                                            | mindspore.ops.Tan                                              | 功能一致 |
| torch.tanh                                           | mindspore.ops.Tanh                                             | 功能一致 |
| torch.tensor                                         | mindspore.Tensor                                               | 功能一致 |
| torch.Tensor                                         | mindspore.Tensor                                               | 功能一致 |
| torch.Tensor.chunk                                   | mindspore.ops.Split                                            | 功能一致 |
| torch.Tensor.expand                                  | mindspore.ops.BroadcastTo                                      | 功能一致 |
| torch.Tensor.fill_                                   | mindspore.ops.Fill                                             | 功能一致 |
| torch.Tensor.float                                   | mindspore.ops.Cast                                             |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/Cast.md)|
| torch.Tensor.index_add                               | mindspore.ops.InplaceAdd                                       | 功能一致 |
| torch.Tensor.mm                                      | mindspore.ops.MatMul                                           | 功能一致 |
| torch.Tensor.mul                                     | mindspore.ops.Mul                                              | 功能一致 |
| torch.Tensor.pow                                     | mindspore.ops.Pow                                              | 功能一致 |
| torch.Tensor.repeat                                  | mindspore.ops.Tile                                             | 功能一致 |
| torch.repeat_interleave                              | mindspore.ops.repeat_elements                                  | 功能一致 |
| torch.Tensor.requires_grad_                          | mindspore.Parameter.requires_grad                              | 功能一致 |
| torch.Tensor.round                                   | mindspore.ops.Round                                            | 功能一致 |
| torch.Tensor.scatter                                 | mindspore.ops.ScatterNd                                        | 功能一致 |
| torch.Tensor.scatter_add_                            | mindspore.ops.ScatterNdAdd                                     |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/ScatterNdAdd.md)|
| torch.Tensor.sigmoid                                 | mindspore.nn.Sigmoid                                           | 功能一致 |
| torch.Tensor.sign                                    | mindspore.ops.Sign                                             | 功能一致 |
| torch.Tensor.size                                    | mindspore.ops.Shape                                            | 功能一致 |
| torch.Tensor.sqrt                                    | mindspore.ops.Sqrt                                             | 功能一致 |
| torch.Tensor.sub                                     | mindspore.ops.Sub                                              | 功能一致 |
| torch.Tensor.t                                       | mindspore.ops.Transpose                                        |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/Transpose.md)|
| torch.Tensor.transpose                               | mindspore.ops.Transpose                                        | 功能一致 |
| torch.Tensor.unsqueeze                               | mindspore.ops.ExpandDims                                       | 功能一致 |
| torch.Tensor.view                                    | mindspore.ops.Reshape                                          | 功能一致 |
| torch.Tensor.zero_                                   | mindspore.ops.ZerosLike                                        | 功能一致 |
| torch.topk                                           | mindspore.ops.TopK                                             |[差异对比]|(https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/TopK.md)|
| torch.transpose                                      | mindspore.ops.Transpose                                        | 功能一致 |
| torch.tril                                           | mindspore.nn.Tril                                              | 功能一致 |
| torch.triu                                           | mindspore.nn.Triu                                              | 功能一致 |
| torch.unbind                                         | mindspore.ops.Unstack                                          | 功能一致 |
| torch.unique                                         | mindspore.ops.Unique                                           |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/Unique.md)|
| torch.unsqueeze                                      | mindspore.ops.ExpandDims                                       | 功能一致 |
| torch.utils.data.DataLoader                          | None                                                           |[差异对比](https://gitee.com/mindspore/docs/blob/master/resource/api_mapping/DataLoader.md)|
| torch.utils.data.Dataset                             | mindspore.dataset.GeneratorDataset                             | 差异对比 |
| torch.utils.data.distributed.DistributedSampler      | mindspore.dataset.DistributedSampler                           | 功能一致 |
| torch.utils.data.RandomSampler                       | mindspore.dataset.RandomSampler                                | 功能一致 |
| torch.utils.data.SequentialSampler                   | mindspore.dataset.SequentialSampler                            | 功能一致 |
| torch.utils.data.SubsetRandomSampler                 | mindspore.dataset.SubsetRandomSampler                          | 功能一致 |
| torch.utils.data.WeightedRandomSampler               | mindspore.dataset.WeightedRandomSampler                        | 功能一致 |
| torch.zeros                                          | mindspore.ops.Zeros                                            | 功能一致 |
| torch.zeros_like                                     | mindspore.ops.ZerosLike                                        | 功能一致 |
| torchtext.data.functional.custom_replace             | mindspore.dataset.text.transforms.RegexReplace                 | 功能一致 |
| torchtext.data.functional.load_sp_model              | mindspore.dataset.text.transforms.SentencePieceTokenizer       | 差异对比 |
| torchtext.data.functional.numericalize_tokens_from_iterator | mindspore.dataset.text.transforms.Lookup                | 差异对比 |
| torchtext.data.functional.sentencepiece_numericalizer | mindspore.dataset.text.transforms.SentencePieceTokenizer      | 差异对比 |
| torchtext.data.functional.sentencepiece_tokenizer    | mindspore.dataset.text.transforms.SentencePieceTokenizer       | 差异对比 |
| torchtext.data.functional.simple_space_split         | mindspore.dataset.text.transforms.WhitespaceTokenizer          | 功能一致 |
| torchtext.data.utils.ngrams_iterator                 | mindspore.dataset.text.transforms.Ngram                        | 功能一致 |
| torchvision.datasets.CelebA                          | mindspore.dataset.CelebADataset                                | 功能一致 |
| torchvision.datasets.CIFAR10                         | mindspore.dataset.Cifar10Dataset                               | 功能一致 |
| torchvision.datasets.CIFAR100                        | mindspore.dataset.Cifar100Dataset                              | 功能一致 |
| torchvision.datasets.CocoDetection                   | mindspore.dataset.CocoDataset                                  | 差异对比 |
| torchvision.datasets.ImageFolder                     | mindspore.dataset.ImageFolderDataset                           | 功能一致 |
| torchvision.datasets.MNIST                           | mindspore.dataset.MnistDataset                                 | 功能一致 |
| torchvision.datasets.VOCDetection                    | mindspore.dataset.VOCDataset                                   | 差异对比 |
| torchvision.datasets.VOCSegmentation                 | mindspore.dataset.VOCDataset                                   | 差异对比 |
| torchvision.ops.nms                                  | mindspore.ops.NMSWithMask                                      | 功能一致 |
| torchvision.ops.roi_align                            | mindspore.ops.ROIAlign                                         | 功能一致 |
| torchvision.transforms.CenterCrop                    | mindspore.dataset.vision.c_transforms.CenterCrop               | 功能一致 |
| torchvision.transforms.ColorJitter                   | mindspore.dataset.vision.c_transforms.RandomColorAdjust        | 功能一致 |
| torchvision.transforms.Compose                       | mindspore.dataset.transforms.c_transforms.Compose              | 功能一致 |
| torchvision.transforms.ConvertImageDtype             | mindspore.dataset.vision.py_transforms.ToType                  | 差异对比 |
| torchvision.transforms.FiveCrop                      | mindspore.dataset.vision.py_transforms.FiveCrop                | 功能一致 |
| torchvision.transforms.GaussianBlur                  | mindspore.dataset.vision.c_transforms.GaussianBlur             | 功能一致 |
| torchvision.transforms.Grayscale                     | mindspore.dataset.vision.py_transforms.Grayscale               | 功能一致 |
| torchvision.transforms.LinearTransformation          | mindspore.dataset.vision.py_transforms.LinearTransformation    | 功能一致 |
| torchvision.transforms.Normalize                     | mindspore.dataset.vision.c_transforms.Normalize                | 功能一致 |
| torchvision.transforms.Pad                           | mindspore.dataset.vision.c_transforms.Pad                      | 功能一致 |
| torchvision.transforms.RandomAffine                  | mindspore.dataset.vision.c_transforms.RandomAffine             | 功能一致 |
| torchvision.transforms.RandomApply                   | mindspore.dataset.transforms.c_transforms.RandomApply          | 功能一致 |
| torchvision.transforms.RandomChoice                  | mindspore.dataset.transforms.c_transforms.RandomChoice         | 功能一致 |
| torchvision.transforms.RandomCrop                    | mindspore.dataset.vision.c_transforms.RandomCrop               | 功能一致 |
| torchvision.transforms.RandomErasing                 | mindspore.dataset.vision.py_transforms.RandomErasing           | 功能一致 |
| torchvision.transforms.RandomGrayscale               | mindspore.dataset.vision.py_transforms.RandomGrayscale         | 功能一致 |
| torchvision.transforms.RandomHorizontalFlip          | mindspore.dataset.vision.c_transforms.RandomHorizontalFlip     | 功能一致 |
| torchvision.transforms.RandomOrder                   | mindspore.dataset.transforms.py_transforms.RandomOrder         | 功能一致 |
| torchvision.transforms.RandomPerspective             | mindspore.dataset.vision.py_transforms.RandomPerspective       | 功能一致 |
| torchvision.transforms.RandomPosterize               | mindspore.dataset.vision.c_transforms.RandomPosterize          | 功能一致 |
| torchvision.transforms.RandomResizedCrop             | mindspore.dataset.vision.c_transforms.RandomResizedCrop        | 功能一致 |
| torchvision.transforms.RandomRotation                | mindspore.dataset.vision.c_transforms.RandomRotation           | 功能一致 |
| torchvision.transforms.RandomSolarize                | mindspore.dataset.vision.c_transforms.RandomSolarize           | 差异对比 |
| torchvision.transforms.RandomVerticalFlip            | mindspore.dataset.vision.c_transforms.RandomVerticalFlip       | 功能一致 |
| torchvision.transforms.Resize                        | mindspore.dataset.vision.c_transforms.Resize                   | 功能一致 |
| torchvision.transforms.TenCrop                       | mindspore.dataset.vision.py_transforms.TenCrop                 | 功能一致 |
| torchvision.transforms.ToPILImage                    | mindspore.dataset.vision.py_transforms.ToPIL                   | 差异对比 |
| torchvision.transforms.ToTensor                      | mindspore.dataset.vision.py_transforms.ToTensor                | 差异对比 |
