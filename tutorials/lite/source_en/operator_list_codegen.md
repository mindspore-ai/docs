# Codegen Operator List

`Linux` `Ascend` `Device` `Inference` `Beginner` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/lite/source_en/operator_list_codegen.md" target="_blank"><img src="./_static/logo_source.png"></a>

This article lists the operators supported by MindSpore Lite Codegen.

| Operation <br/>&nbsp;   | CPU<br/>FP32 | CPU<br/>Int8 | CMSIS<br/>Int8  | TensorFlow<br/>Lite operators supported    | Caffe<br/>Lite operators supported  | Onnx<br/>Lite operators supported          |TensorFlow<br/>operators supported          |
|-----------------------|--------------|--------------|-----------------|---------------------------------|--------------------------|-------------------------------------------------|-------------------------------------------------|
| Abs                   | Supported    |              |                 | Abs                             |                          | Abs                                             |                                                 |
| Add                   | Supported    | Supported    | Supported       | Add                             |                          | Add, Int8Add                                    | Add, AddV2                                      |
| AddN                  | Supported    |              |                 | AddN                            |                          |                                                 |                                                 |
| AvgPool               | Supported    | Supported    | Supported       | MeanPooling                     | Pooling                  | AveragePool, GlobalAveragePool, Int8AveragePool |                                                 |
| BatchNorm             |              | Supported    | Supported       |                                 | BatchNorm                | BatchNormalization                              |                                                 |
| BiasAdd               | Supported    |              |                 |                                 |                          | BiasAdd                                         | BiasAdd                                         |
| Cast                  | Supported    | Supported    | Supported       | Cast, QUANTIZE, DEQUANTIZE      |                          | Cast                                            | Cast                                            |
| Ceil                  | Supported    |              |                 | Ceil                            |                          | Ceil                                            |                                                 |
| Concat                | Supported    | Supported    | Supported       | Concat                          | Concat                   | Concat                                          | ConcatV2                                        |
| Conv2d                | Supported    | Supported    | Supported       | Conv2D                          | Convolution              | Conv, Int8Conv, ConvRelu, Int8ConvRelu          | Conv2D                                          |
| Cos                   | Supported    |              |                 | Cos                             |                          | Cos                                             |                                                 |
| DetectionPostProcess  |              | Supported    |                 | Custom                          |                          |                                                 |                                                 |
| Div                   | Supported    | Supported    | Supported       | Div, RealDiv                    |                          | Div                                             | Div, RealDiv                                    |
| Eltwise               | Supported    |              |                 |                                 | Eltwise                  | Sum, Max<sup>[3]</sup>                          |                                                 |
| Equal                 | Supported    |              |                 | Equal                           |                          | Equal                                           | Equal                                           |
| ExpandDims            | Supported    |              |                 | ExpandDims                      |                          |                                                 | ExpandDims                                      |
| Floor                 | Supported    |              |                 | flOOR                           |                          | Floor                                           |                                                 |
| FloorDiv              | Supported    |              |                 | FloorDiv                        |                          |                                                 |                                                 |
| FloorMod              | Supported    |              |                 | FloorMod                        |                          |                                                 |                                                 |
| FullConnection        | Supported    | Supported    | Supported       | FullyConnected                  | InnerProduct             |                                                 |                                                 |
| Greater               | Supported    |              |                 | Greater                         |                          | Greater                                         | Greater                                         |
| GreaterEqual          | Supported    |              |                 | GreaterEqual                    |                          |                                                 | GreaterEqual                                    |
| Less                  | Supported    |              |                 | Less                            |                          | Less                                            | Less                                            |
| LessEqual             | Supported    |              |                 | LessEqual                       |                          |                                                 | LessEqual                                       |
| Log                   | Supported    |              |                 | Log                             |                          | Log                                             |                                                 |
| LogicalAnd            | Supported    |              |                 | LogicalAnd                      |                          | And                                             | LogicalAnd                                      |
| LogicalNot            | Supported    |              |                 | LogicalNot                      |                          | Not                                             |                                                 |
| LogicalOr             | Supported    |              |                 | LogicalOr                       |                          | Or                                              |                                                 |
| MatMul                | Supported    | Supported    |                 |                                 |                          | MatMul                                          | MatMul                                          |
| Maximum               | Supported    |              |                 | Maximum                         |                          |                                                 | Maximum                                         |
| MaxPool               | Supported    | Supported    | Supported       | MaxPooling                      | Pooling                  | MaxPool, GlobalMaxPool                          |                                                 |
| Minimum               | Supported    |              |                 | Minimum                         |                          | Min                                             | Minimum                                         |
| Mul                   | Supported    | Supported    | Supported       | Mul                             |                          | Mul                                             | Mul                                             |
| Neg                   | Supported    |              |                 | Neg                             |                          | Neg                                             |                                                 |
| NotEqual              | Supported    |              |                 | NotEqual                        |                          |                                                 |NotEqual                                         |
| ReLU                  | Supported    | Supported    | Supported       | Relu                            | ReLU                     | Relu                                            | Relu                                            |
| ReLU6                 | Supported    | Supported    | Supported       | Relu6                           | ReLU6                    | Clip<sup>[1]</sup>                              | Relu6                                           |
| Reshape               | Supported    | Supported    | Supported       | Reshape                         | Reshape                  | Reshape,Flatten                                 | Reshape                                         |
| Resize                |              | Supported    |                 | ResizeBilinear, NearestNeighbor | Interp                   |                                                 |                                                 |
| Round                 | Supported    |              |                 | Round                           |                          | Round                                           | Round                                           |
| Rsqrt                 | Supported    |              |                 | Rsqrt                           |                          |                                                 |                                                 |
| Sigmoid               | Supported    | Supported    | Supported       | Logistic                        | Sigmoid                  | Sigmoid                                         | Sigmoid                                         |
| Sin                   | Supported    |              |                 | Sin                             |                          | Sin                                             |                                                 |
| Softmax               | Supported    | Supported    | Supported       | Softmax                         | Softmax                  | Softmax                                         |                                                 |
| Sqrt                  | Supported    |              |                 | Sqrt                            |                          | Sqrt                                            |                                                 |
| Square                | Supported    |              |                 | Square                          |                          |                                                 |                                                 |
| SquaredDifference     | Supported    |              |                 | SquaredDifference               |                          |                                                 |                                                 |
| Squeeze               | Supported    |              |                 | Squeeze                         |                          | Squeeze                                         | Squeeze                                         |
| Sub                   | Supported    | Supported    | Supported       | Sub                             |                          | Sub                                             | Sub                                             |

[1] Clip: Only support converting clip(0, 6) to Relu6.

[2] Pow: Only support the form where the exponent is a single constant.

[3] Sum and Max: Only support 2 inputs.
