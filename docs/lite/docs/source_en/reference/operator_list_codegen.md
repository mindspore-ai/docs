# Codegen Operator List

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/reference/operator_list_codegen.md)

This article lists the operators supported by MindSpore Lite Codegen.

| Operation <br/>&nbsp;   | CPU<br/>FP32 | CPU<br/>Int8 | CMSIS<br/>Int8  | TensorFlow Lite<br/>operators supported    | Caffe Lite<br/>operators supported  | Onnx Lite<br/>operators supported          |TensorFlow<br/>operators supported          |
|-----------------------|:--------------:|:--------------:|:-----------------:|---------------------------------|--------------------------|-------------------------------------------------|-------------------------------------------------|
| Abs                   | ✅    |              |                 | Abs                             |                          | Abs                                             |                                                 |
| Add                   | ✅    | ✅    | ✅       | Add                             |                          | Add, Int8Add                                    | Add, AddV2                                      |
| AddN                  | ✅    |              |                 | AddN                            |                          |                                                 |                                                 |
| AvgPool               | ✅    | ✅    | ✅       | MeanPooling                     | Pooling                  | AveragePool,<br/>GlobalAveragePool,<br/>Int8AveragePool |                                                 |
| BatchNorm             |              | ✅    | ✅       |                                 | BatchNorm                | BatchNormalization                              |                                                 |
| BiasAdd               | ✅    |              |                 |                                 |                          | BiasAdd                                         | BiasAdd                                         |
| Cast                  | ✅    | ✅    | ✅       | Cast, QUANTIZE,<br/>DEQUANTIZE      |                          | Cast                                            | Cast                                            |
| Ceil                  | ✅    |              |                 | Ceil                            |                          | Ceil                                            |                                                 |
| Concat                | ✅    | ✅    | ✅       | Concat                          | Concat                   | Concat                                          | ConcatV2                                        |
| Conv2d                | ✅    | ✅    | ✅       | Conv2D                          | Convolution              | Conv,<br/>Int8Conv,<br/>ConvRelu,<br/>Int8ConvRelu          | Conv2D                                          |
| Cos                   | ✅    |              |                 | Cos                             |                          | Cos                                             |                                                 |
| DetectionPostProcess  |              | ✅    |                 | Custom                          |                          |                                                 |                                                 |
| Div                   | ✅    | ✅    | ✅       | Div, RealDiv                    |                          | Div                                             | Div, RealDiv                                    |
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
