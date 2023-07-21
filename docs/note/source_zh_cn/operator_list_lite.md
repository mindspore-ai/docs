# MindSpore Lite算子支持

`Linux` `Ascend` `端侧` `推理应用` `初级` `中级` `高级`

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/docs/note/source_zh_cn/operator_list_lite.md)

| 操作名                   | CPU<br/>FP16 | CPU<br/>FP32 | CPU<br/>Int8 | CPU<br/>UInt8 | GPU<br/>FP16 | GPU<br/>FP32 | 支持的Tensorflow<br/>Lite算子 | 支持的Caffe<br/>Lite算子 | 支持的Onnx<br/>Lite算子 |
|-----------------------|----------|----------|----------|-----------|----------|-------------------|----------|----------|---------|
| Abs                   |          | Supported        | Supported        | Supported         | Supported        | Supported        | Abs        |               | Abs                |
| Add                   | Supported        | Supported        | Supported        | Supported         | Supported        | Supported        | Add        |               | Add                |
| AddN                  |          | Supported        |          |           |          |          | AddN       |               |                    |
| Argmax                |          | Supported        | Supported        | Supported         |          |          | Argmax     | ArgMax        | ArgMax             |
| Argmin                |          | Supported        | Supported        | Supported         |          |          | Argmin     |               |                    |
| AvgPool               | Supported        | Supported        | Supported        | Supported         | Supported        | Supported        | MeanPooling| Pooling       | AveragePool        |
| BatchNorm             | Supported        | Supported        | Supported        | Supported         | Supported        | Supported        |            | BatchNorm     | BatchNormalization |
| BatchToSpace          |          | Supported        | Supported        | Supported         |          |          | BatchToSpace, BatchToSpaceND |  |               |
| BiasAdd               |          | Supported        | Supported        | Supported         | Supported        | Supported         |           |                | BiasAdd            |
| Broadcast             |          | Supported        |          |           |          |          | BroadcastTo |               | Expand             |
| Cast                  | Supported        | Supported        | Supported| Supported         | Supported        | Supported        | Cast, QUANTIZE, DEQUANTIZE  |        | Cast               |
| Ceil                  |          | Supported        | Supported        | Supported         | Supported        | Supported        | Ceil        |               | Ceil               |
| Concat                | Supported        | Supported        | Supported        | Supported         | Supported        | Supported        | Concat      | Concat        | Concat             |
| Conv2d                | Supported        | Supported        | Supported        | Supported         | Supported        | Supported        | Conv2D      | Convolution   | Conv               |
| Conv2dTranspose       | Supported        | Supported        | Supported        | Supported         | Supported        | Supported        | DeConv2D    | Deconvolution | ConvTranspose      |
| Cos                   |          | Supported        | Supported        | Supported         | Supported        | Supported        | Cos         |               | Cos                |
| Crop                  |          | Supported        | Supported        | Supported         |          |          |             |  Crop         |                    |
| DeDepthwiseConv2D     |          | Supported        | Supported        | Supported         |          |          |             |  Deconvolution| ConvTranspose      |
| DepthToSpace          |          | Supported        | Supported        | Supported         |          |          | DepthToSpace|               | DepthToSpace       |
| DepthwiseConv2dNative | Supported        | Supported        | Supported        | Supported         | Supported        | Supported        | DepthwiseConv2D | Convolution   | Convolution    |
| DetectionPostProcess  |          | Supported        |       |          |                   |          | DetectionPostProcess |           |                 |
| Div                   | Supported        | Supported        | Supported        | Supported         | Supported        | Supported        | Div, RealDiv         |               | Div                |
| Eltwise               | Supported        | Supported        |          |           |          |          |             |  Eltwise      |                    |
| Elu                   |          | Supported        |          |           |          |          |  Elu        |               | Elu                |
| Equal                 | Supported        | Supported        | Supported        | Supported         |          |          | Equal       |               | Equal              |
| Exp                   |          | Supported        |          |           | Supported        | Supported        | Exp         |  Exp             | Exp                |
| ExpandDims            |          | Supported        |          |           |          |          |ExpandDims             |               |                    |
| Fill                  |          | Supported        |          |           |          |          | Fill        |               |                    |
| Flatten               |          | Supported        |          |           |          |          |             | Flatten       |                    |
| Floor                 |          | Supported        | Supported        | Supported         | Supported        | Supported        | flOOR       |               | Floor              |
| FloorDiv              | Supported        | Supported        |          |           |          |          | FloorDiv    |               |                    |
| FloorMod              | Supported        | Supported        |          |           |          |          | FloorMod    |               |                    |
| FullConnection        | Supported        | Supported        | Supported        | Supported         | Supported        | Supported        | FullyConnected  | InnerProduct  |                |
| GatherNd              |          | Supported        | Supported        | Supported         |          |          | GatherND    |               |                    |
| GatherV2              |          | Supported        | Supported        | Supported         |          |          | Gather      |               | Gather             |
| Greater               | Supported        | Supported        | Supported        | Supported         |          |          | Greater     |               | Greater            |
| GreaterEqual          | Supported        | Supported        | Supported        | Supported         |          |          | GreaterEqual|               |                    |
| Hswish                | Supported        | Supported        | Supported        | Supported         |          |          | HardSwish   |               |                    |
| L2Norm                |         | Supported        |          |           |         |         | L2_NORMALIZATION   |               |           |
| LeakyReLU             | Supported        | Supported        |          |           | Supported        | Supported        | LeakyRelu   |               | LeakyRelu          |
| Less                  | Supported        | Supported        | Supported        | Supported         |          |          | Less        |               | Less               |
| LessEqual             | Supported        | Supported        | Supported        | Supported         |          |          | LessEqual   |               |                    |
| LRN     |          | Supported        |          |           |          |          | LocalResponseNorm  |        | Lrn, LRN                |
| Log                   |          | Supported        | Supported        | Supported         | Supported        | Supported        | Log         |               | Log                |
| LogicalAnd            | Supported        | Supported        |          |           |          |          | LogicalAnd  |               |                    |
| LogicalNot            |          | Supported        | Supported        | Supported         | Supported        | Supported        | LogicalNot  |               |                    |
| LogicalOr             | Supported        | Supported        |          |           |          |          | LogicalOr   |               |                    |
| LSTM                  |          | Supported        |          |           |          |          |             |               |                    |
| MatMul                |          | Supported        | Supported        | Supported         | Supported        | Supported        |             |               | MatMul             |
| Maximum               | Supported        | Supported        |          |           |          |          | Maximum     |               | Max                |
| MaxPool               | Supported        | Supported        | Supported        | Supported         | Supported        | Supported        | MaxPooling  | Pooling       | MaxPool            |
| Minimum               | Supported        | Supported        |          |           |          |          | Minimum     |               | Min                |
| Mul                   | Supported        | Supported        | Supported        | Supported         | Supported        | Supported        | Mul         |               | Mul                |
| Neg                   |          | Supported        |          |           |          |          |   Neg       |               | Neg                   |
| NotEqual              | Supported        | Supported        | Supported        | Supported         |          |          | NotEqual    |               |                    |
| OneHot                |          | Supported        |          |           |          |          | OneHot      |               |                    |
| Pad                   | Supported        | Supported        | Supported        | Supported         |          |          | Pad, MirrorPad         |               | Pad                |
| Pow                   |          | Supported        | Supported        | Supported         |          |         | Pow          | Power         | Power              |
| PReLU                 |          | Supported        |          |           | Supported        | Supported        |        | PReLU         |              |
| Range                 |          | Supported        |          |           |          |          | Range       |               |                    |
| Rank                  |          | Supported        |          |           |          |          | Rank        |               |                    |
| ReduceASum            |          | Supported        |          |           |          |          |          |   Reduction            |           |
| ReduceMax             | Supported        | Supported        | Supported        | Supported         |          |          | ReduceMax   |               | ReduceMax          |
| ReduceMean            | Supported        | Supported        | Supported        | Supported         |          |          | Mean        | Reduction              | ReduceMean         |
| ReduceMin             | Supported        | Supported        | Supported        | Supported         |          |          | ReduceMin   |               | ReduceMin          |
| ReduceProd            | Supported        | Supported        | Supported        | Supported         |          |          | ReduceProd  |               |                    |
| ReduceSum             | Supported        | Supported        | Supported        | Supported         |          |          | Sum         | Reduction              | ReduceSum          |
| ReduceSumSquare       | Supported        | Supported        | Supported        | Supported         |          |          |             |  Reduction             |                    |
| ReLU                  | Supported        | Supported        | Supported        | Supported         | Supported        | Supported        | Relu        | ReLU          | Relu               |
| ReLU6                 | Supported        | Supported        | Supported        | Supported         | Supported        | Supported        | Relu6       | ReLU6         | Clip*              |
| Reshape               | Supported        | Supported        | Supported        | Supported         | Supported        | Supported        | Reshape     | Reshape       | Reshape,Flatten    |
| Resize                |          | Supported        | Supported        | Supported         |          |          | ResizeBilinear, NearestNeighbor | Interp        |                    |
| Reverse               |          | Supported        |          |           |          |          | reverse     |               |                    |
| ReverseSequence       |          | Supported        |          |           |          |          | ReverseSequence  |          |                    |
| Round                 |          | Supported        | Supported        | Supported         | Supported        | Supported        | Round       |               |                    |
| Rsqrt                 |          | Supported        | Supported        | Supported         | Supported        | Supported        | Rsqrt       |               |                    |
| Scale                 |          | Supported        |          |           | Supported        | Supported        |             |  Scale        |                    |
| ScatterNd             |          | Supported        |          |           |          |          | ScatterNd   |               |                    |
| Shape                 |          | Supported        |          |          |          |          | Shape       |               | Shape              |
| Sigmoid               | Supported        | Supported        | Supported        | Supported         | Supported        | Supported        | Logistic    | Sigmoid       | Sigmoid            |
| Sin                   |          | Supported        | Supported        | Supported         | Supported        | Supported        | Sin         |               | Sin                |
| Slice                 |          | Supported        | Supported        | Supported         | Supported        | Supported        | Slice       | Slice              | Slice              |
| Softmax               | Supported        | Supported        | Supported        | Supported         | Supported        | Supported        | Softmax     | Softmax       | Softmax            |
| SpaceToBatch          |          | Supported        | Supported        |           |          |          |             |               |                    |
| SpaceToBatchND        |          | Supported        | Supported         |           |          |          | SpaceToBatchND |            |                    |
| SpaceToDepth          |          | Supported        |          |           |          |          | SpaceToDepth   |            | SpaceToDepth       |
| SparseToDense         |          | Supported        |          |           |          |          |  SpareToDense  |            |                    |
| Split                 | Supported        | Supported        | Supported        | Supported         |          |          | Split, SplitV  |            |                    |
| Sqrt                  |          | Supported        | Supported        | Supported         | Supported        | Supported        | Sqrt        |               | Sqrt               |
| Square                |          | Supported        | Supported        | Supported         | Supported        | Supported        | Square      |               |                    |
| SquaredDifference     |          | Supported        |          |           |          |          |  SquaredDifference |         |                    |
| Squeeze               |          | Supported        | Supported        | Supported         |          |          | Squeeze     |               | Squeeze            |
| StridedSlice          |          | Supported        | Supported        | Supported         |          |          | StridedSlice|               |                    |
| Stack                 |          | Supported        |          |           |          |          | Stack       |               |                    |
| Sub                   | Supported        | Supported        | Supported        | Supported         | Supported        | Supported        | Sub         |               |  Sub               |
| Tanh                  | Supported        | Supported        |          |           | Supported        | Supported        | Tanh        | TanH          |                    |
| Tile                  |          | Supported        |          |           |          |          | Tile        | Tile              | Tile               |
| TopK                  |          | Supported        | Supported        | Supported         |          |          | TopKV2      |               |                    |
| Transpose             | Supported        | Supported        |          |           | Supported        | Supported        | Transpose   | Permute       | Transpose          |
| Unique                |          | Supported        |          |           |          |          | Unique      |               |                    |
| Unsqueeze             |          | Supported        | Supported        | Supported         |          |          |             |               | Unsqueeze          |
| Unstack               |          | Supported        |          |           |          |          | Unstack     |               |                    |
| Where                 |          | Supported        |          |           |          |          |  Where      |               |                    |
| ZerosLike             |          | Supported        |          |           |          |          | ZerosLike   |               |               |

* Clip: 仅支持将clip(0, 6)转换为Relu6.
