# Codegen Operator List

`Linux` `Ascend` `Device` `Inference` `Beginner` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r1.2/tutorials/lite/source_en/operator_list_codegen.md" target="_blank"><img src="./_static/logo_source.png"></a>

This article lists the operators supported by MindSpore Lite Codegen.

| Operation <br/>&nbsp;   | CPU<br/>FP32 | CPU<br/>Int8 | CMSIS<br/>Int8  | TensorFlow<br/>Lite operators supported    | Caffe Lite<br/>operators supported  | Onnx Lite<br/>operators supported          |TensorFlow<br/>operators supported          |
|-----------------------|:--------------:|:--------------:|:-----------------:|---------------------------------|--------------------------|-------------------------------------------------|-------------------------------------------------|
| Abs                   | ✅    |              |                 | Abs                             |                          | Abs                                             |                                                 |
| Add                   | ✅    | ✅    | ✅       | Add                             |                          | Add, Int8Add                                    | Add, AddV2                                      |
| AddN                  | ✅    |              |                 | AddN                            |                          |                                                 |                                                 |
| AvgPool               | ✅    | ✅    | ✅       | MeanPooling                     | Pooling                  | AveragePool,<br/>GlobalAveragePool,<br/>Int8AveragePool |                                                 |
| BatchNorm             |              | ✅    | ✅       |                                 | BatchNorm                | BatchNormalization                              |                                                 |
| BiasAdd               | ✅    |              |                 |                                 |                          | BiasAdd                                         | BiasAdd                                         |
| Cast                  | ✅    | ✅    | ✅       | Cast,<br/>QUANTIZE,<br/>DEQUANTIZE      |                          | Cast                                            | Cast                                            |
| Ceil                  | ✅    |              |                 | Ceil                            |                          | Ceil                                            |                                                 |
| Concat                | ✅    | ✅    | ✅       | Concat                          | Concat                   | Concat                                          | ConcatV2                                        |
| Conv2d                | ✅    | ✅    | ✅       | Conv2D                          | Convolution              | Conv, Int8Conv,<br/>ConvRelu,<br/>Int8ConvRelu          | Conv2D                                          |
| Cos                   | ✅    |              |                 | Cos                             |                          | Cos                                             |                                                 |
| DetectionPostProcess  |              | ✅    |                 | Custom                          |                          |                                                 |                                                 |
| Div                   | ✅    | ✅    | ✅       | Div,<br/>RealDiv                    |                          | Div                                             | Div,<br/>RealDiv                                    |
| Eltwise               | ✅    |              |                 |                                 | Eltwise                  | Sum, Max<sup>[3]</sup>                          |                                                 |
| Equal                 | ✅    |              |                 | Equal                           |                          | Equal                                           | Equal                                           |
| ExpandDims            | ✅    |              |                 | ExpandDims                      |                          |                                                 | ExpandDims                                      |
| Floor                 | ✅    |              |                 | flOOR                           |                          | Floor                                           |                                                 |
| FloorDiv              | ✅    |              |                 | FloorDiv                        |                          |                                                 |                                                 |
| FloorMod              | ✅    |              |                 | FloorMod                        |                          |                                                 |                                                 |
| FullConnection        | ✅    | ✅    | ✅       | FullyConnected                  | InnerProduct             |                                                 |                                                 |
| Greater               | ✅    |              |                 | Greater                         |                          | Greater                                         | Greater                                         |
| GreaterEqual          | ✅    |              |                 | GreaterEqual                    |                          |                                                 | GreaterEqual                                    |
| Less                  | ✅    |              |                 | Less                            |                          | Less                                            | Less                                            |
| LessEqual             | ✅    |              |                 | LessEqual                       |                          |                                                 | LessEqual                                       |
| Log                   | ✅    |              |                 | Log                             |                          | Log                                             |                                                 |
| LogicalAnd            | ✅    |              |                 | LogicalAnd                      |                          | And                                             | LogicalAnd                                      |
| LogicalNot            | ✅    |              |                 | LogicalNot                      |                          | Not                                             |                                                 |
| LogicalOr             | ✅    |              |                 | LogicalOr                       |                          | Or                                              |                                                 |
| MatMul                | ✅    | ✅    |                 |                                 |                          | MatMul                                          | MatMul                                          |
| Maximum               | ✅    |              |                 | Maximum                         |                          |                                                 | Maximum                                         |
| MaxPool               | ✅    | ✅    | ✅       | MaxPooling                      | Pooling                  | MaxPool,<br/>GlobalMaxPool                          |                                                 |
| Minimum               | ✅    |              |                 | Minimum                         |                          | Min                                             | Minimum                                         |
| Mul                   | ✅    | ✅    | ✅       | Mul                             |                          | Mul                                             | Mul                                             |
| Neg                   | ✅    |              |                 | Neg                             |                          | Neg                                             |                                                 |
| NotEqual              | ✅    |              |                 | NotEqual                        |                          |                                                 |NotEqual                                         |
| ReLU                  | ✅    | ✅    | ✅       | Relu                            | ReLU                     | Relu                                            | Relu                                            |
| ReLU6                 | ✅    | ✅    | ✅       | Relu6                           | ReLU6                    | Clip<sup>[1]</sup>                              | Relu6                                           |
| Reshape               | ✅    | ✅    | ✅       | Reshape                         | Reshape                  | Reshape,Flatten                                 | Reshape                                         |
| Resize                |              | ✅    |                 | ResizeBilinear,<br/>NearestNeighbor | Interp                   |                                                 |                                                 |
| Round                 | ✅    |              |                 | Round                           |                          | Round                                           | Round                                           |
| Rsqrt                 | ✅    |              |                 | Rsqrt                           |                          |                                                 |                                                 |
| Sigmoid               | ✅    | ✅    | ✅       | Logistic                        | Sigmoid                  | Sigmoid                                         | Sigmoid                                         |
| Sin                   | ✅    |              |                 | Sin                             |                          | Sin                                             |                                                 |
| Softmax               | ✅    | ✅    | ✅       | Softmax                         | Softmax                  | Softmax                                         |                                                 |
| Sqrt                  | ✅    |              |                 | Sqrt                            |                          | Sqrt                                            |                                                 |
| Square                | ✅    |              |                 | Square                          |                          |                                                 |                                                 |
| SquaredDifference     | ✅    |              |                 | SquaredDifference               |                          |                                                 |                                                 |
| Squeeze               | ✅    |              |                 | Squeeze                         |                          | Squeeze                                         | Squeeze                                         |
| Sub                   | ✅    | ✅    | ✅       | Sub                             |                          | Sub                                             | Sub                                             |

[1] Clip: Only support converting clip(0, 6) to Relu6.

[2] Pow: Only support the form where the exponent is a single constant.

[3] Sum and Max: Only support 2 inputs.
