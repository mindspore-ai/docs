# MindSpore Lite支持的TensorFlow Lite算子列表

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.11.0/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/r2.11.0/docs/lite/docs/source_zh_cn/reference/operator_list_lite_for_tflite.md)

| MindSpore Lite算子名称 | 对应的TensorFlow Lite算子                                    | 算子规格 |
| ---------------------- | ------------------------------------------------------------ | -------- |
| Abs                    | Abs                                                          | -        |
| Activation             | Activation, Elu, ReLU, ReLU6, PReLU, <br/>LeakyReLU, Tanh, HardSwish, Logistic | Elu、ReLU、ReLU6、PReLU、LeakyReLU、Tanh、HardSwish、Logistic不支持int类型。 |
| AddFusion              | Add                                                          | -        |
| AddN                   | AddN                                                         | -        |
| ArgmaxFusion           | Argmax                                                       | -        |
| ArgminFusion           | Argmin                                                       | -        |
| AvgPoolFusion          | MeanPooling                                                  | -        |
| BatchToSpace           | BatchToSpace                                                 | -        |
| BatchToSpaceND         | BatchToSpaceND                                               | -        |
| BinaryCrossEntropy     | BinaryCrossEntropy                                           | -        |
| Cast                   | Cast,<br/>QUANTIZE,<br/>DEQUANTIZE                           | -        |
| Ceil                   | Ceil                                                         | -        |
| Clip                   | Clip                                                         | -        |
| Concat                 | Concat                                                       | -        |
| Conv2DFusion           | Conv2D                                                       | -        |
| Conv2dTransposeFusion  | DeConv2D                                                     | -        |
| Cos                    | Cos                                                          | -        |
| CustomExtractFeatures  | ExtractFeatures                                              | TFLite CUSTOM算子。 |
| CustomNormalize        | Normalize                                                    | TFLite CUSTOM算子。 |
| CustomPredict          | Predict                                                      | TFLite CUSTOM算子。 |
| DepthToSpace           | DepthToSpace                                                 | -        |
| DetectionPostProcess   | Custom                                                       | -        |
| DivFusion              | Div                                                          | -        |
| Equal                  | Equal                                                        | -        |
| Erf                    | Erf                                                          | -        |
| ExpFusion              | Exp                                                          | -        |
| ExpandDims             | ExpandDims                                                   | -        |
| Fill                   | Fill                                                         | -        |
| Floor                  | Floor                                                        | -        |
| FloorDiv               | FloorDiv                                                     | -        |
| FloorMod               | FloorMod                                                     | -        |
| FullConnection         | FullyConnected                                               | -        |
| FusedBatchNorm         | FusedBatchNorm                                               | -        |
| GatherNd               | GatherND                                                     | -        |
| Gather                 | Gather                                                       | -        |
| Greater                | Greater                                                      | -        |
| GreaterEqual           | GreaterEqual                                                 | -        |
| HashtableLookup        | HashtableLookup                                              | -        |
| LeakyReLU              | LeakyRelu                                                    | -        |
| Less                   | Less                                                         | -        |
| LessEqual              | LessEqual                                                    | -        |
| LRN                    | LocalResponseNorm                                            | -        |
| Log                    | Log                                                          | -        |
| LogicalAnd             | LogicalAnd                                                   | -        |
| LogicalNot             | LogicalNot                                                   | -        |
| LogicalOr              | LogicalOr                                                    | -        |
| LogSoftmax             | LogSoftmax                                                   | -        |
| LshProjection          | LshProjection                                                | -        |
| MatMulFusion           | BatchMatMul                                                  | -        |
| Maximum                | Maximum                                                      | -        |
| MaxPoolFusion          | MaxPooling                                                   | -        |
| Minimum                | Minimum                                                      | -        |
| MulFusion              | Mul                                                          | -        |
| Neg                    | Neg                                                          | -        |
| NotEqual               | NotEqual                                                     | -        |
| OneHot                 | OneHot                                                       | -        |
| PadFusion              | Pad, MirrorPad, PadV2                                        | -        |
| PowFusion              | Pow                                                          | -        |
| PReLUFusion            | PRELU                                                        | -        |
| Range                  | Range                                                        | -        |
| Rank                   | Rank                                                         | -        |
| ReduceFusion           | Sum, Mean, ReduceMax, ReduceMin, ReduceProd                  | Sum、Mean、ReduceMax、ReduceMin、ReduceProd不支持bool类型。 |
| Reshape                | Reshape                                                      | -        |
| Resize                 | ResizeBilinear,<br/>ResizeNearestNeighbor                    | ResizeNearestNeighbor不支持fp16类型。 |
| ReverseV2              | ReverseV2                                                    | -        |
| ReverseSequence        | ReverseSequence                                              | -        |
| Round                  | Round                                                        | -        |
| Rsqrt                  | Rsqrt                                                        | -        |
| ScatterNd              | ScatterNd                                                    | -        |
| Shape                  | Shape                                                        | -        |
| Sin                    | Sin                                                          | -        |
| SliceFusion            | Slice                                                        | -        |
| SkipGram               | SkipGram                                                     | -        |
| Softmax                | Softmax                                                      | -        |
| SpaceToBatchND         | SpaceToBatchND                                               | -        |
| SpaceToDepth           | SpaceToDepth                                                 | -        |
| SparseToDense          | SparseToDense                                                 | -        |
| Split                  | Split, SplitV                                                | -        |
| Sqrt                   | Sqrt                                                         | -        |
| Square                 | Square                                                       | -        |
| SquaredDifference      | SquaredDifference                                            | -        |
| Squeeze                | Squeeze                                                      | -        |
| StridedSlice           | StridedSlice                                                 | -        |
| Stack                  | Pack                                                         | -        |
| SubFusion              | Sub                                                          | -        |
| TileFusion             | Tile                                                         | -        |
| TopKFusion             | TopKV2                                                       | -        |
| Transpose              | Transpose                                                    | -        |
| Unique                 | Unique                                                       | -        |
| Unstack                | Unpack                                                       | -        |
| Where                  | Where                                                        | -        |
| ZerosLike              | ZerosLike                                                    | -        |
