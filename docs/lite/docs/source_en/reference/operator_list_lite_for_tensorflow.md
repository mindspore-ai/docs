# List of TensorFlow Operators Supported by MindSpore Lite

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/lite/docs/source_en/reference/operator_list_lite_for_tensorflow.md)

| MindSpore Lite Operator Names | Corresponding TensorFlow Operators                                    | Operator Specifications |
| ---------------------- | ------------------------------------------------------------ | -------- |
| Abs                    | Abs                                                          | -        |
| Activation             | Activation, Elu, Relu, Relu6, Sigmoid, Tanh, LeakyRelu, Softplus | Selu is not supported. Elu, Relu, Relu6, Sigmoid, Tanh, and LeakyRelu do not support int type. |
| AddFusion              | Add, AddV2                                                   | -        |
| AddN                   | AddN                                                         | -        |
| All                    | All                                                          | -        |
| Assert                 | Assert                                                       | Assert does not produce outputs. Models without outputs fail to convert. |
| ArgmaxFusion           | ArgMax                                                       | -        |
| ArgminFusion           | ArgMin                                                       | -        |
| AvgPoolFusion          | AvgPool                                                      | -        |
| BatchToSpace           | BatchToSpace                                                 | -        |
| BatchToSpaceND         | BatchToSpaceND                                               | -        |
| BiasAdd                | BiasAdd                                                      | -        |
| BinaryCrossEntropy     | BinaryCrossEntropy                                           | -        |
| BroadcastTo            | BroadcastTo                                                  | -        |
| Cast                   | Cast                                                         | -        |
| Ceil                   | Ceil                                                         | -        |
| Clip                   | Clip                                                         | -        |
| Concat                 | ConcatV2                                                     | -        |
| Conv2DFusion           | Conv2D                                                       | -        |
| Conv2dTransposeFusion  | Conv2DBackpropInput                                          | -        |
| Cos                    | Cos                                                          | -        |
| CropAndResize          | CropAndResize                                                | -        |
| CumSum                 | Cumsum                                                       | -        |
| DepthToSpace           | DepthToSpace                                                 | -        |
| DivFusion              | Div, RealDiv                                                 | -        |
| Dropout                | Dropout                                                      | -        |
| Equal                  | Equal                                                        | -        |
| Erf                    | Erf                                                          | -        |
| ExpFusion              | Exp                                                          | -        |
| ExpandDims             | ExpandDims                                                   | May be optimized to a Reshape operator during graph construction. |
| Fill                   | Fill                                                         | -        |
| Floor                  | Floor                                                        | -        |
| FloorDiv               | FloorDiv                                                     | -        |
| FloorMod               | FloorMod                                                     | -        |
| FusedBatchNorm         | FusedBatchNorm,<br/>FusedBatchNormV3                         | -        |
| GatherNd               | GatherNd                                                     | -        |
| Gather                 | GatherV2                                                     | -        |
| Greater                | Greater                                                      | -        |
| GreaterEqual           | GreaterEqual                                                 | -        |
| InvertPermutation      | InvertPermutation                                            | -        |
| IsFinite               | IsFinite                                                     | -        |
| LeakyReLU              | LeakyRelu                                                    | Does not support int type. |
| Less                   | Less                                                         | -        |
| LessEqual              | LessEqual                                                    | -        |
| Log                    | Log                                                          | -        |
| Log1p                  | Log1p                                                        | -        |
| LogicalAnd             | LogicalAnd                                                   | -        |
| LogicalNot             | LogicalNot                                                   | -        |
| LogicalOr              | LogicalOr                                                    | -        |
| MatMulFusion           | MatMul,<br/>BatchMatMul,<br/>BatchMatMulV2                   | -        |
| Maximum                | Maximum                                                      | -        |
| MaxPoolFusion          | MaxPool                                                      | -        |
| Merge                  | Merge                                                        | -        |
| Minimum                | Minimum                                                      | -        |
| MulFusion              | Mul                                                          | -        |
| Neg                    | Neg                                                          | -        |
| NotEqual               | NotEqual                                                     | -        |
| NonMaxSuppression      | NonMaxSuppressionV3                                          | -        |
| OneHot                 | OneHot                                                       | -        |
| PadFusion              | MirrorPad, Pad, PadV2                                        | -        |
| PowFusion              | Pow                                                          | -        |
| RaggedRange            | RaggedRange                                                  | -        |
| RandomStandardNormal   | RandomStandardNormal                                         | -        |
| Range                  | Range                                                        | -        |
| Rank                   | Rank                                                         | May be optimized to a Const node during graph construction. |
| ReduceFusion           | Sum, Max, Min, Mean, Prod, All                               | Sum, Max, Min, and Prod do not support bool type. |
| Reshape                | Reshape                                                      | -        |
| Resize                 | ResizeBilinear,<br/>ResizeBicubic,<br/>ResizeNearestNeighbor | -        |
| ReverseV2              | ReverseV2                                                    | -        |
| ReverseSequence        | ReverseSequence                                              | -        |
| Round                  | Round                                                        | -        |
| Rsqrt                  | Rsqrt                                                        | -        |
| Select                 | Select                                                       | -        |
| Shape                  | Shape                                                        | May be optimized to a Const node during graph construction. |
| Sin                    | Sin                                                          | -        |
| Size                   | Size                                                         | May be optimized to a Const node during graph construction. |
| SliceFusion            | Slice                                                        | -        |
| Softmax                | Softmax                                                      | -        |
| Softplus               | Softplus                                                     | -        |
| SpaceToBatchND         | SpaceToBatchND                                               | -        |
| Split                  | Split, SplitV                                                | In multi-output scenarios, ensure that the number of model output names matches the actual outputs. |
| Sqrt                   | Sqrt                                                         | -        |
| Square                 | Square                                                       | -        |
| SquaredDifference      | SquaredDifference                                            | -        |
| Squeeze                | Squeeze                                                      | May be optimized to a Reshape operator during graph construction. |
| StridedSlice           | StridedSlice                                                 | -        |
| Stack                  | Pack                                                         | -        |
| SubFusion              | Sub                                                          | -        |
| Switch                 | Switch                                                       | -        |
| TensorListFromTensor   | TensorListFromTensor                                         | Not supported when TensorFlow outputs the variant type. |
| TensorListGetItem      | TensorListGetItem                                            | Not supported when the input is the variant type. |
| TensorListReserve      | TensorListReserve                                            | Not supported when TensorFlow outputs the variant type. |
| TensorListSetItem      | TensorListSetItem                                            | Not supported when the input is the variant type. |
| TensorListStack        | TensorListStack                                              | Not supported when the input is the variant type. |
| TileFusion             | Tile                                                         | -        |
| TopKFusion             | TopKV2                                                       | In multi-output scenarios, ensure that the number of model output names matches the actual outputs. |
| Transpose              | Transpose                                                    | -        |
| UnsortedSegmentSum     | UnsortedSegmentSum                                           | -        |
| Where                  | Where                                                        | -        |
| ZerosLike              | ZerosLike                                                    | May be constant-folded to a Const node during graph construction. |
| Other operators supported by the conversion tool | Enter,<br/>Exit, If,<br/>LinSpace,<br/>LoopCond,<br/>NextIteration,<br/>StatelessIf,<br/>StatelessWhile,<br/>TensorArrayGatherV3,<br/>TensorArrayReadV3,<br/>TensorArrayScatterV3,<br/>TensorArraySizeV3,<br/>TensorArrayV3,<br/>TensorArrayWriteV3,<br/>While | The [converter tool](https://www.mindspore.cn/lite/docs/en/master/converter/converter_tool.html) supports operators that do not require explicit implementation. Typically, such operators are optimized away in the converter tool, either fused or replaced with other operators. |
