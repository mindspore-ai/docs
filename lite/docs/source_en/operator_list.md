# Operator List

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r0.7/lite/docs/source_en/operator_list.md)

> √ The checked items are the operators supported by MindSpore Lite。

| Operation               | CPU<br/>FP16 | CPU<br/>FP32 | CPU<br/>Int8 | CPU<br/>UInt8 | GPU<br/>FP16 | GPU<br/>FP32 | Tensorflow <br/>Lite op supported | Caffe <br/>Lite op supported | Onnx <br/>Lite op supported |
|-----------------------|----------|----------|-----------|----------|----------|------------------|----------|----------|----------|
| Abs                   |          | √        | √        | √         |          |          | Abs        |               | Abs                |
| Add                   | √        | √        | √        | √         |          | √        | Add        |               | Add                |
| AddN                  |          | √        |          |           |          |          | AddN       |               |                    |
| Argmax                |          | √        | √        | √         |          |          | Argmax     | ArgMax        | ArgMax             |
| Argmin                |          | √        | √        | √         |          |          | Argmin     |               |                    |
| AvgPool               | √        | √        | √        | √         |          | √        | MeanPooling| Pooling       | AveragePool        |
| BatchNorm             | √        | √        | √        | √         |          | √        |            | BatchNorm     | BatchNormalization |
| BatchToSpace          |          | √        | √        | √         |          |          | BatchToSpace, BatchToSpaceND |  |               |
| BiasAdd               |          | √        | √        | √         |          | √         |           |                | BiasAdd            |
| Broadcast             |          | √        |          |           |          |          | BroadcastTo |               | Expand             |
| Cast                  | √        | √        |          | √         |          |          | Cast, DEQUANTIZE*  |        | Cast               |
| Ceil                  |          | √        | √        | √         |          |          | Ceil        |               | Ceil               |
| Concat                | √        | √        | √        | √         | √        | √        | Concat      | Concat        | Concat             |
| Conv2d                | √        | √        | √        | √         | √        | √        | Conv2D      | Convolution   | Conv               |
| Conv2dTranspose       | √        | √        | √        | √         | √        | √        | DeConv2D    | Deconvolution | ConvTranspose      |
| Cos                   |          | √        | √        | √         |          |          | Cos         |               | Cos                |
| Crop                  |          | √        | √        | √         |          |          |             |  Crop         |                    |
| DeDepthwiseConv2D     |          | √        | √        | √         |          |          |             |  Deconvolution| ConvTranspose      |
| DepthToSpace          |          | √        | √        | √         |          |          | DepthToSpace|               | DepthToSpace       |
| DepthwiseConv2dNative | √        | √        | √        | √         | √        | √        | DepthwiseConv2D | Convolution   | Convolution    |
| Div                   | √        | √        | √        | √         |          | √        | Div, RealDiv         |               | Div                |
| Eltwise               | √        | √        |          |           |          |          |             |  Eltwise      |                    |
| Elu                   |          | √        |          |           |          |          |  Elu        |               | Elu                |
| Equal                 | √        | √        | √        | √         |          |          | Equal       |               | Equal              |
| Exp                   |          | √        |          |           |          |          | Exp         |               | Exp                |
| ExpandDims            |          | √        |          |           |          |          |             |               |                    |
| Fill                  |          | √        |          |           |          |          | Fill        |               |                    |
| Flatten               |          | √        |          |           |          |          |             | Flatten       |                    |
| Floor                 |          | √        | √        | √         |          |          | flOOR       |               | Floor              |
| FloorDiv              | √        | √        |          |           |          |          | FloorDiv    |               |                    |
| FloorMod              | √        | √        |          |           |          |          | FloorMod    |               |                    |
| FullConnection        |          | √        | √        | √         |          |          | FullyConnected  | InnerProduct  |                |
| GatherNd              |          | √        | √        | √         |          |          | GatherND    |               |                    |
| GatherV2              |          | √        | √        | √         |          |          | Gather      |               | Gather             |
| Greater               | √        | √        | √        | √         |          |          | Greater     |               | Greater            |
| GreaterEqual          | √        | √        | √        | √         |          |          | GreaterEqual|               |                    |
| Hswish                | √        | √        | √        | √         |          |          | HardSwish   |               |                    |
| LeakyReLU             | √        | √        |          |           |          | √        | LeakyRelu   |               | LeakyRelu          |
| Less                  | √        | √        | √        | √         |          |          | Less        |               | Less               |
| LessEqual             | √        | √        | √        | √         |          |          | LessEqual   |               |                    |
| LRN     |          | √        |          |           |          |          | LocalResponseNorm  |        | Lrn                |
| Log                   |          | √        | √        | √         |          |          | Log         |               | Log                |
| LogicalAnd            | √        | √        |          |           |          |          | LogicalAnd  |               |                    |
| LogicalNot            |          | √        | √        | √         |          |          | LogicalNot  |               |                    |
| LogicalOr             | √        | √        |          |           |          |          | LogicalOr   |               |                    |
| LSTM                  |          | √        |          |           |          |          |             |               |                    |
| MatMul                |          | √        | √        | √         | √        | √        |             |               | MatMul             |
| Maximum               | √        | √        |          |           |          |          | Maximum     |               | Max                |
| MaxPool               | √        | √        | √        | √         |          | √        | MaxPooling  | Pooling       | MaxPool            |
| Minimum               | √        | √        |          |           |          |          | Minimum     |               | Min                |
| Mul                   | √        | √        | √        | √         |          | √        | Mul         |               | Mul                |
| NotEqual              | √        | √        | √        | √         |          |          | NotEqual    |               |                    |
| OneHot                |          | √        |          |           |          |          | OneHot      |               |                    |
| Pad                   |          | √        | √        | √         |          |          | Pad         |               | Pad                |
| Pow                   |          | √        | √        | √         |          |         | Pow          | Power         | Power              |
| PReLU                 |          | √        |          |          |          | √        |        | PReLU         |              |
| Range                 |          | √        |          |           |          |          | Range       |               |                    |
| Rank                  |          | √        |          |           |          |          | Rank        |               |                    |
| ReduceMax             | √        | √        | √        | √         |          |          | ReduceMax   |               | ReduceMax          |
| ReduceMean            | √        | √        | √        | √         |          |          | Mean        |               | ReduceMean         |
| ReduceMin             | √        | √        | √        | √         |          |          | ReduceMin   |               | ReduceMin          |
| ReduceProd            | √        | √        | √        | √         |          |          | ReduceProd  |               |                    |
| ReduceSum             | √        | √        | √        | √         |          |          | Sum         |               | ReduceSum          |
| ReduceSumSquare       | √        | √        | √        | √         |          |          |             |               |                    |
| ReLU                  | √        | √        | √        | √         |          | √        | Relu        | ReLU          | Relu               |
| ReLU6                 | √        | √        | √        | √         |          | √        | Relu6       | ReLU6         | Clip*              |
| Reshape               | √        | √        | √        | √         |          | √        | Reshape     | Reshape       | Reshape,Flatten    |
| Resize                |          | √        | √        | √         |          |          | ResizeBilinear, NearestNeighbor | Interp        |                    |
| Reverse               |          | √        |          |           |          |          | reverse     |               |                    |
| ReverseSequence       |          | √        |          |           |          |          | ReverseSequence  |          |                    |
| Round                 |          | √        | √        | √         |          |          | Round       |               |                    |
| Rsqrt                 |          | √        | √        | √         |          |          | Rsqrt       |               |                    |
| Scale                 |          | √        |          |           |          |          |             |  Scale        |                    |
| ScatterNd             |          | √        |          |           |          |          | ScatterNd   |               |                    |
| Shape                 |          | √        |          |          |          |          | Shape       |               | Shape              |
| Sigmoid               | √        | √        | √        | √         |          | √        | Logistic    | Sigmoid       | Sigmoid            |
| Sin                   |          | √        | √        | √         |          |          | Sin         |               | Sin                |
| Slice                 |          | √        | √        | √         | √        | √        | Slice       |               | Slice              |
| Softmax               | √        | √        | √        | √         |          | √        | Softmax     | Softmax       | Softmax            |
| SpaceToBatch          |          | √        |          |           |          |          |             |               |                    |
| SpaceToBatchND        |          | √        |          |           |          |          | SpaceToBatchND |            |                    |
| SpaceToDepth          |          | √        |          |           |          |          | SpaceToDepth   |            | SpaceToDepth       |
| SparseToDense         |          | √        |          |           |          |          |  SpareToDense  |            |                    |
| Split                 | √        | √        | √        | √         |          |          | Split, SplitV  |            |                    |
| Sqrt                  |          | √        | √        | √         |          |          | Sqrt        |               | Sqrt               |
| Square                |          | √        | √        | √         |          |          | Square      |               |                    |
| SquaredDifference     |          | √        |          |           |          |         |  SquaredDifference |         |                    |
| Squeeze               |          | √        | √        | √         |          |          | Squeeze     |               | Squeeze            |
| StridedSlice          |          | √        | √        | √         |          |          | StridedSlice|               |                    |
| Stack                 |          | √        |          |           |          |          | Stack       |               |                    |
| Sub                   | √        | √        | √        | √         |          | √        | Sub         |               |  Sub               |
| Tanh                  | √        | √        |          |           |          |          | Tanh        | TanH          |                    |
| Tile                  |          | √        |          |           |          |          | Tile        |               | Tile               |
| TopK                  |          | √        | √        | √         |          |          | TopKV2      |               |                    |
| Transpose             | √        | √        |          |           |          | √        | Transpose   | Permute       | Transpose          |
| Unique                |          | √        |          |           |          |          | Unique      |               |                    |
| Unsqueeze             |          | √        | √        | √         |          |          |             |               | Unsqueeze          |
| Unstack               |          | √        |          |           |          |          | Unstack     |               |                    |
| Where                 |          | √        |          |           |          |          |  Where      |               |                    |
| ZerosLike             |          | √        |          |           |          |          | ZerosLike   |               |               |             

* Clip: only support convert clip(0, 6) to Relu6.
* DEQUANTIZE: only support to convert fp16 to fp32.
