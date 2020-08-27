# 算子支持

<a href="https://gitee.com/mindspore/docs/blob/master/lite/docs/source_zh_cn/operator_list.md" target="_blank"><img src="./_static/logo_source.png"></a>

> √勾选的项为MindSpore Lite所支持的算子。

| 操作名                   | CPU<br/>FP16 | CPU<br/>FP32 | CPU<br/>Int8 | CPU<br/>UInt8 | GPU<br/>FP16 | GPU<br/>FP32 | 算子类别 | 支持的Tensorflow<br/>Lite op | 支持的Caffe<br/>Lite op | 支持的Onnx<br/>Lite op |
|-----------------------|----------|----------|----------|-----------|----------|----------|------------------|----------|----------|----------|
| Abs                   |          | √        | √        | √         |          |          | math_ops         | Abs                             |               | Abs                |
| Add                   |          |          |          |           |          | √        |                  | Add                             |               | Add                |
| AddN                  |          | √        |          |           |          |          | math_ops         | AddN                            |               |                    |
| Argmax                |          | √        | √        | √         |          |          | array_ops        | Argmax                          | ArgMax        | ArgMax             |
| Argmin                |          | √        |          |           |          |          | array_ops        | Argmin                          |               |                    |
| Asin                  |          |          |          |           |          |          |                  |                                 |               | Asin               |
| Atan                  |          |          |          |           |          |          |                  |                                 |               | Atan               |
| AvgPool               |          | √        | √        | √         |          | √        | nn_ops           | MeanPooling                     | Pooling       | AveragePool        |
| BatchMatMul           | √        | √        | √        | √         |          |          | math_ops         |                                 |               |                    |
| BatchNorm             |          | √        |          |           |          | √        | nn_ops           |                                 | BatchNorm     | BatchNormalization |
| BatchToSpace          |          |          |          |           |          |          | array_ops        | BatchToSpace, BatchToSpaceND    |               |                    |
| BatchToSpaceND        |          |          |          |           |          |          |                  |                                 |               |                    |
| BiasAdd               |          | √        |          | √         |          | √        | nn_ops           |                                 |               | BiasAdd            |
| Broadcast             |          | √        |          |           |          |          | comm_ops         | BroadcastTo                     |               | Expand             |
| Cast                  |          | √        |          |           |          |          | array_ops        | Cast, DEQUANTIZE*               |               | Cast               |
| Ceil                  |          | √        |          | √         |          |          | math_ops         | Ceil                            |               | Ceil               |
| Concat                |          | √        | √        | √         |          | √        | array_ops        | Concat                          | Concat        | Concat             |
| Constant              |          |          |          |           |          |          |        |         |                                 | Constant      |
| Conv1dTranspose       |          |          |          | √         |          |          | layer/conv       |                                 |               |                    |
| Conv2d                | √        | √        | √        | √         |          | √        | layer/conv       | Conv2D                          | Convolution   | Conv               |
| Conv2dTranspose       |          | √        | √        | √         |          | √        | layer/conv       | DeConv2D                        | Deconvolution | ConvTranspose      |
| Cos                   |          | √        | √        | √         |          |          | math_ops         | Cos                             |               | Cos                |
| Crop                  |          |          |          |           |          |          |                  |                                 |  Crop         |                    |
| DeDepthwiseConv2D     |          |          |          |           |          |          |                  |                                 |  Deconvolution| ConvTranspose      |
| DepthToSpace          |          |          |          |           |          |          |                  | DepthToSpace                    |               | DepthToSpace       |
| DepthwiseConv2dNative | √        | √        | √        | √         |          | √        | nn_ops           | DepthwiseConv2D                 | Convolution   | Convolution        |
| Div                   |          | √        | √        | √         |          | √        | math_ops         | Div                             |               | Div                |
| Dropout               |          |          |          |           |          |          |                  |                                 |               | Dropout            |
| Eltwise               |          |          |          |           |          |          |                  |                                 |  Eltwise      |                    |
| Elu                   |          |          |          |           |          |          |                  |  Elu                            |               | Elu                |
| Equal                 |          | √        | √        | √         |          |          | math_ops         | Equal                           |               | Equal              |
| Exp                   |          | √        |          |           |          |          | math_ops         | Exp                             |               | Exp                |
| ExpandDims            |          | √        |          |           |          |          | array_ops        |                                 |               |                    |
| Fill                  |          | √        |          |           |          |          | array_ops        | Fill                            |               |                    |
| Flatten               |          |          |          |           |          |          |                  |                                 | Flatten       |                    |
| Floor                 |          | √        | √        | √         |          |          | math_ops         | flOOR                           |               | Floor              |
| FloorDiv              |          | √        |          |           |          |          | math_ops         | FloorDiv                        |               |                    |
| FloorMod              |          | √        |          |           |          |          | nn_ops           | FloorMod                        |               |                    |
| FullConnection        |          | √        |          |           |          |          | layer/basic      | FullyConnected                  | InnerProduct  |                    |
| GatherNd              |          | √        |          |           |          |          | array_ops        | GatherND                        |               |                    |
| GatherV2              |          | √        |          |           |          |          | array_ops        | Gather                          |               | Gather             |
| Greater               |          | √        | √        | √         |          |          | math_ops         | Greater                         |               | Greater            |
| GreaterEqual          |          | √        | √        | √         |          |          | math_ops         | GreaterEqual                    |               |                    |
| Hswish                |          |          |          |           |          |          |                  | HardSwish                       |               |                    |
| L2norm                |          |          |          |           |          |          |                  | L2_NORMALIZATION                |               |                    |
| LeakyReLU             |          | √        |          |           |          | √        | layer/activation | LeakyRelu                       |               | LeakyRelu          |
| Less                  |          | √        | √        | √         |          |          | math_ops         | Less                            |               | Less               |
| LessEqual             |          | √        | √        | √         |          |          | math_ops         | LessEqual                       |               |                    |
| LocalResponseNorm     |          |          |          |           |          |          |                  | LocalResponseNorm               |               | Lrn                |
| Log                   |          | √        | √        | √         |          |          | math_ops         | Log                             |               | Log                |
| LogicalAnd            |          | √        |          |           |          |          | math_ops         | LogicalAnd                      |               |                    |
| LogicalNot            |          | √        | √        | √         |          |          | math_ops         | LogicalNot                      |               |                    |
| LogicalOr             |          | √        |          |           |          |          | math_ops         | LogicalOr                       |               |                    |
| LSTM                  |          | √        |          |           |          |          | layer/lstm       |                                 |               |                    |
| MatMul                | √        | √        | √        | √         |          | √        | math_ops         |                                 |               | MatMul             |
| Maximum               |          |          |          |           |          |          | math_ops         | Maximum                         |               | Max                |
| MaxPool               |          | √        | √        | √         |          | √        | nn_ops           | MaxPooling                      | Pooling       | MaxPool            |
| Minimum               |          |          |          |           |          |          | math_ops         | Minimum                         |               | Min                |
| Mul                   |          | √        | √        | √         |          | √        | math_ops         | Mul                             |               | Mul                |
| Neg                   |          |          |          |           |          |         | math_ops         |                                 |               | Neg                |
| NotEqual              |          | √        | √        | √         |          |          | math_ops         | NotEqual                        |               |                    |
| OneHot                |          | √        |          |           |          |          | layer/basic      | OneHot                          |               |                    |
| Pack                  |          | √        |          |           |          |          | nn_ops           |                                 |               |                    |
| Pad                   |          | √        | √        | √         |          |          | nn_ops           | Pad                             |               | Pad                |
| Pow                   |          | √        | √        | √         |          |          | math_ops         | Pow                             | Power         | Power              |
| PReLU                 |          | √        | √        | √         |          | √        | layer/activation | Prelu                           | PReLU         | PRelu              |
| Range                 |          | √        |          |           |          |          | layer/basic      | Range                           |               |                    |
| Rank                  |          | √        |          |           |          |          | array_ops        | Rank                            |               |                    |
| RealDiv               |          | √        | √        | √         |          | √        | math_ops         | RealDiv                         |               |                    |
| ReduceMax             |          | √        | √        | √         |          |          | math_ops         | ReduceMax                       |               | ReduceMax          |
| ReduceMean            |          | √        | √        | √         |          |          | math_ops         | Mean                            |               | ReduceMean         |
| ReduceMin             |          | √        | √        | √         |          |          | math_ops         | ReduceMin                       |               | ReduceMin          |
| ReduceProd            |          | √        | √        | √         |          |          | math_ops         | ReduceProd                      |               |                    |
| ReduceSum             |          | √        | √        | √         |          |          | math_ops         | Sum                             |               | ReduceSum          |
| ReLU                  |          | √        | √        | √         |          | √        | layer/activation | Relu                            | ReLU          | Relu               |
| ReLU6                 |          | √        |          |           |          | √        | layer/activation | Relu6                           | ReLU6               | Clip*              |
| Reshape               |          | √        | √        | √         |          | √        | array_ops        | Reshape                         | Reshape       | Reshape,Flatten    |
| Resize                |          |          |          |           |          |          |                  | ResizeBilinear, NearestNeighbor | Interp        |                    |
| Reverse               |          |          |          |           |          |          |                  | reverse                         |               |                    |
| ReverseSequence       |          | √        |          |           |          |          | array_ops        | ReverseSequence                 |               |                    |
| Round                 |          | √        |          | √         |          |          | math_ops         | Round                           |               |                    |
| Rsqrt                 |          | √        | √        | √         |          |          | math_ops         | Rsqrt                           |               |                    |
| Scale                 |          |          |          |           |          |          |                  |                                 |  Scale        |                    |
| ScatterNd             |          | √        |          |           |          |          | array_ops        | ScatterNd                       |               |                    |
| Shape                 |          | √        |          | √         |          |          | array_ops        | Shape                           |               | Shape              |
| Sigmoid               |          | √        | √        | √         |          | √        | nn_ops           | Logistic                        | Sigmoid       | Sigmoid            |
| Sin                   |          |          |          |           |          |          |                  | Sin                             |               | Sin                |
| Slice                 |          | √        | √        | √         |          | √        | array_ops        | Slice                           |               | Slice              |
| Softmax               |          | √        | √        | √         |          | √        | layer/activation | Softmax                         | Softmax       | Softmax            |
| SpaceToBatchND        |          | √        |          |           |          |          | array_ops        | SpaceToBatchND                  |               |                    |
| SpareToDense          |          |          |          |           |          |          |                  |  SpareToDense                   |               |                    |
| SpaceToDepth          |          | √        |          |           |          |          | array_ops        | SpaceToDepth                    |               | SpaceToDepth       |
| Split                 |          | √        | √        | √         |          |          | array_ops        | Split, SplitV                   |               |                    |
| Sqrt                  |          | √        | √        | √         |          |          | math_ops         | Sqrt                            |               | Sqrt               |
| Square                |          | √        | √        | √         |          |          | math_ops         | Square                          |               |                    |
| SquaredDifference     |          |          |          |           |          |          |                  |  SquaredDifference              |               |                    |
| Squeeze               |          | √        | √        | √         |          |          | array_ops        | Squeeze                         |               | Squeeze            |
| StridedSlice          |          | √        | √        | √         |          |          | array_ops        | StridedSlice                    |               |                    |
| Stack                 |          |          |          |           |          |          |                  | Stack                           |               |                    |
| Sub                   |          | √        | √        | √         |          | √        | math_ops         | Sub                             |               |  Sub               |
| Tan                   |          |          |          |           |          |          |                  |                                 |               | Tan                |
| Tanh                  |          | √        |          |           |          |          | layer/activation | Tanh                            | TanH              |                    |
| TensorAdd             |          | √        | √        | √         |          | √        | math_ops         |                                 |               |                    |
| Tile                  |          | √        |          |           |          |          | array_ops        | Tile                            |               | Tile               |
| TopK                  |          | √        | √        | √         |          |          | nn_ops           | TopKV2                          |               |                    |
| Transpose             |          | √        | √        | √         |          | √        | array_ops        | Transpose                       | Permute       | Transpose          |
| Unique                |          |          |          |           |          |          |                  | Unique                          |               |                    |
| Unpack                |          | √        |          |           |          |          | nn_ops           |                                 |               |                    |
| Unsample              |          |          |          |           |          |          |                  |                                 |               | Unsample           |
| Unsqueeze             |          |          |          |           |          |          |                  |                                 |               | Unsqueeze          |
| Unstack               |          |          |          |           |          |          |                  | Unstack                         |               |                    |
| Where                 |          |          |          |           |          |          |                  |  Where                          |               |                    |
| ZerosLike             |          | √        |          |           |          |          | array_ops        | ZerosLike                       |               |                    |             

* Clip: only support convert clip(0, 6) to Relu6.
* DEQUANTIZE: only support to convert fp16 to fp32.
