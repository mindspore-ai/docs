# List of TensorFlow Operators Supported by MindSpore Lite

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/lite/docs/source_en/reference/operator_list_lite_for_tensorflow.md)

| MindSpore Lite Operator Names | Corresponding TensorFlow Operators                                    |
| ---------------------- | ------------------------------------------------------------ |
| Abs                    | Abs                                                          |
| Activation             | Activation, Elu, Relu, Relu6, Sigmoid, Tanh, Selu, LeakyRelu, Softplus |
| Adam                   | Adam                                                         |
| AddFusion              | Add, AddV2                                                   |
| All                    | All                                                          |
| ApplyMomentum          | ApplyMomentum                                                |
| Assert                 | Assert                                                       |
| Assign                 | Assign                                                       |
| ArgmaxFusion           | ArgMax                                                       |
| ArgminFusion           | ArgMin                                                       |
| AvgPoolFusion          | AvgPool                                                      |
| BatchToSpace           | BatchToSpace                                                 |
| BatchToSpaceND         | BatchToSpaceND                                               |
| BiasAdd                | BiasAdd                                                      |
| BinaryCrossEntropy     | BinaryCrossEntropy                                           |
| BroadcastTo            | BroadcastTo                                                  |
| Cast                   | Cast                                                         |
| Ceil                   | Ceil                                                         |
| Clip                   | Clip                                                         |
| Concat                 | ConcatV2                                                     |
| Conv2DFusion           | Conv2D                                                       |
| Conv2dTransposeFusion  | Conv2DBackpropInput                                          |
| Cos                    | Cos                                                          |
| CropAndResize          | CropAndResize                                                |
| CumSum                 | Cumsum                                                       |
| DepthToSpace           | DepthToSpace                                                 |
| DivFusion              | Div, RealDiv                                                 |
| Dropout                | Dropout                                                      |
| Elu                    | NonMaxSuppressionV3                                          |
| Equal                  | Equal                                                        |
| Erf                    | Erf                                                          |
| ExpFusion              | Exp                                                          |
| ExpandDims             | ExpandDims                                                   |
| Fill                   | Fill                                                         |
| Floor                  | Floor                                                        |
| FloorDiv               | FloorDiv                                                     |
| FloorMod               | FloorMod                                                     |
| FusedBatchNorm         | FusedBatchNorm,<br/>FusedBatchNormV3                         |
| GatherNd               | GatherNd                                                     |
| Gather                 | GatherV2                                                     |
| Greater                | Greater                                                      |
| GreaterEqual           | GreaterEqual                                                 |
| InvertPermutation      | InvertPermutation                                            |
| IsFinite               | IsFinite                                                     |
| LeakyReLU              | LeakyRelu                                                    |
| Less                   | Less                                                         |
| LessEqual              | LessEqual                                                    |
| Log                    | Log                                                          |
| Log1p                  | Log1p                                                        |
| LogicalAnd             | LogicalAnd                                                   |
| LogicalNot             | LogicalNot                                                   |
| LogicalOr              | LogicalOr                                                    |
| MatMulFusion           | MatMul,<br/>BatchMatMul,<br/>BatchMatMulV2                   |
| Maximum                | Maximum                                                      |
| MaxPoolFusion          | MaxPool                                                      |
| Merge                  | Merge                                                        |
| Minimum                | Minimum                                                      |
| Mod                    | Mod                                                          |
| MulFusion              | Mul                                                          |
| Neg                    | Neg                                                          |
| NotEqual               | NotEqual                                                     |
| NonMaxSuppression     | NonMaxSuppression                                           |
| NonZero                | NonZero                                                      |
| OneHot                 | OneHot                                                       |
| OnesLike               | OnesLike                                                     |
| PadFusion              | MirrorPad, Pad, PadV2                                        |
| PowFusion              | Pow                                                          |
| RaggedRange            | RaggedRange                                                  |
| RandomNormal           | RandomNormal                                                 |
| RandomStandardNormal   | RandomStandardNormal                                         |
| Range                  | Range                                                        |
| Rank                   | Rank                                                         |
| ReduceFusion           | Sum, Max, Min, Mean, Prod, All                               |
| Reshape                | Reshape                                                      |
| Resize                 | ResizeBilinear,<br/>ResizeBicubic,<br/>ResizeNearestNeighbor |
| ReverseV2              | ReverseV2                                                    |
| ReverseSequence        | ReverseSequence                                              |
| Round                  | Round                                                        |
| Rsqrt                  | Rsqrt                                                        |
| Select                 | Select                                                       |
| Selu                   | Selu                                                         |
| SGD                    | SGD                                                          |
| Shape                  | Shape                                                        |
| Sin                    | Sin                                                          |
| Size                   | Size                                                         |
| SliceFusion            | Slice                                                        |
| Softmax                | Softmax                                                      |
| Softplus               | Softplus                                                     |
| SpaceToBatchND         | SpaceToBatchND                                               |
| Split                  | Split, SplitV                                                |
| Sqrt                   | Sqrt                                                         |
| Square                 | Square                                                       |
| SquaredDifference      | SquaredDifference                                            |
| Squeeze                | Squeeze                                                      |
| StridedSlice           | StridedSlice                                                 |
| Stack                  | Pack                                                         |
| SubFusion              | Sub                                                          |
| Switch                 | Switch                                                       |
| TensorListFromTensor   | TensorListFromTensor                                         |
| TensorListGetItem      | TensorListGetItem                                            |
| TensorListReserve      | TensorListReserve                                            |
| TensorListSetItem      | TensorListSetItem                                            |
| TensorListStack        | TensorListStack                                              |
| TensorScatterAdd       | TensorScatterAdd                                             |
| TileFusion             | Tile                                                         |
| TopKFusion             | TopKV2                                                       |
| Transpose              | Transpose                                                    |
| UnsortedSegmentSum     | UnsortedSegmentSum                                           |
| Where                  | Where                                                        |
| ZerosLike              | ZerosLike                                                    |
| Other operators supported by the conversion tool | Dropout, Enter,<br/>Exit, If, <br/>LinSpace,<br/>LoopCond,<br/>NextIteration,<br/>StatelessIf,<br/>StatelessWhile,<br/>TensorArrayGatherV3,<br/>TensorArrayReadV3,<br/>TensorArrayScatterV3,<br/>TensorArraySizeV3,<br/>TensorArrayV3,<br/>TensorArrayWriteV3,<br/>While |

> [Converter too](https://www.mindspore.cn/lite/docs/en/master/converter/converter_tool.html) supports operators that are not required to be explicitly implemented. Typically, such operators are optimized away in conversion tools—either fused or replaced with other operators.
