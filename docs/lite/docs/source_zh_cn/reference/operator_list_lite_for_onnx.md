# MindSpore Lite支持的ONNX算子列表

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.1/docs/lite/docs/source_zh_cn/reference/operator_list_lite_for_onnx.md)

> - 以下所有算子，均不支持int64类型输入。
> - 当前支持使用环境变量export KEEP_ORIGIN_DTYPE=1来保持数据类型为int64，当使用int32数据类型存在溢出时可以考虑使用该选项，但是目前仅为实验性选项，后续将移除。

| MindSpore Lite算子名称 | 算子功能                                                     | 对应ONNX算子名称                                             | 算子规格                                                     |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Abs                    | 逐元素计算绝对值                                             | Abs                                                          | 不支持uint8类型。不支持输入张量量化参数为空。                |
| Activation             | 激活函数                                                     | Relu、LeakyRelu、PRelu、Elu、Tanh、Sigmoid、HardSigmoid、Softplus、Gelu | -                                                            |
| AddFusion              | 逐元素计算加法                                               | Add、Int8Add                                                 | -                                                            |
| AdderFusion            | 基于加法的卷积运算                                                   | adder_f                                                      | -                                                            |
| ArgmaxFusion           | 求某一维度最大值                                             | ArgMax                                                       | 不支持uint8类型。不支持输入张量量化参数为空。                |
| ArgminFusion           | 求某一维度最小值                                             | ArgMin                                                       | -                                                            |
| AvgPoolFusion          | 平均池化                                                     | AveragePool、GlobalAveragePool、Int8AveragePool             | -                                                            |
| BatchNorm              | 批量归一化                                                   | BatchNormalization                                           | -                                                            |
| BiasAdd                | 将偏置向量（bias）添加到输入张量                             | BiasAdd                                                      | -                                                            |
| BroadcastTo            | 扩维                                                         | Expand                                                       | -                                                            |
| Cast                   | 数据类型转换                                                 | Cast                                                         | 不支持以下数值类型转换：fp32转int8、fp32转uint32、int32转int8、int32转uint32、int32转uint8、int8转bool、int8转uint8。 |
| Ceil                   | 向上取整                                                     | Ceil                                                         | -                                                            |
| Clip                   | 限制元素范围                                                 | Clip                                                         | 仅支持将clip(0,  6)转换为Relu6。                             |
| Concat                 | 拼接张量                                                     | Concat                                                       | -                                                            |
| ConstantOfShape        | 生成一个与输入形状相同的张量，并用指定常量填充               | ConstantOfShape                                              | -                                                            |
| Conv2DFusion           | 2D卷积                                                       | Conv、Int8Conv、ConvRelu、Int8ConvRelu                      | -                                                            |
| Conv2dTransposeFusion  | 执行转置卷积运算                                             | ConvTranspose                                                | -                                                            |
| Cos                    | 逐元素计算余弦                                               | Cos                                                          | -                                                            |
| CumSum                 | 累计元素和                                                   | CumSum                                                       | -                                                            |
| DepthToSpace           | 将深度数据重新排列到空间维度中                               | DepthToSpace                                                 | 不支持uint8类型。不支持未知维度输入。                        |
| DivFusion              | 逐元素除法                                                   | Div                                                          | 不支持除数为0。                                              |
| Dropout                | 随机将输入张量的部分元素置 0                                 | Dropout                                                      | -                                                            |
| DynamicQuant           | 动态将浮点张量量化为 uint8类型                               | DynamicQuantizeLinear                                        | -                                                            |
| Eltwise                | 元素级运算                                                   | Sum、Max                                                     | 仅支持输入个数为2。                                          |
| Elu                    | 激活函数，对负输入使用指数修正                               | Elu、NonMaxSuppression                                       | -                                                            |
| Equal                  | 判断输入是否相等                                             | Equal                                                        | 不支持uint8输入；int8输入不支持bool输出。                    |
| Erf                    | 误差函数                                                     | Erf                                                          | -                                                            |
| ExpFusion              | 逐元素取指数                                                 | Exp                                                          | -                                                            |
| Flatten                | 数据按维度展开                                               | Flatten                                                      | 不支持uint8类型。                                            |
| Floor                  | 向下取整                                                     | Floor                                                        | -                                                            |
| FusedBatchNorm         | 对输入做标准化                                               | BatchNormalization                                           | -                                                            |
| Gather                 | 沿单一维度收集指定索引位置的元素                             | Gather                                                       | 不支持uint8类型。不支持QuantType_QUANT_NONE量化类型。        |
| GatherD                | 将输入tensor中的元素根据索引tensor进行收集                   | GatherElements                                               | -                                                            |
| GatherNd               | 将输入张量的切片聚合成具有indices指定维度的新张量            | GatherND                                                     | -                                                            |
| Greater                | 逐元素比较两个张量，返回 A > B的逻辑结果（True/False）       | Greater                                                      | -                                                            |
| GreaterEqual           | 逐元素比较两个张量，返回 A ≥ B的逻辑结果                     | GreaterOrEqual                                               | -                                                            |
| InstanceNorm           | 实例归一化                                                   | InstanceNormalization                                        | -                                                            |
| LeakyReLU              | 带泄漏的 ReLU激活函数，对负输入给予微小斜率                  | LeakyRelu                                                    | -                                                            |
| Less                   | 逐元素比较两个张量，返回 A < B的逻辑结果。                   | Less                                                         | -                                                            |
| Log                    | 逐元素求对数                                                 | Log                                                          | 不支持负数输入。                                             |
| LogicalAnd             | 逐元素逻辑与（AND）运算                                      | And                                                          | -                                                            |
| LogicalNot             | 元素级逻辑非运算                                                 | Not                                                          | -                                                            |
| LogicalOr              | 逐元素逻辑或（OR）运算                                       | Or                                                           | -                                                            |
| LogSoftmax             | 对输入向量进行softmax操作，然后再对softmax结果取对数         | LogSoftmax                                                   | 不支持inf输入。                                              |
| LRN                    | 局部响应标准化，用于防止数据过度拟合                         | LRN                                                          | -                                                            |
| LSTM                   | 长短期记忆网络单元                                           | LSTM                                                         | -                                                            |
| MatMulFusion           | 对2个输入做矩阵乘法运算；使用输入张量、一组学习的权重计算内积，并添加偏差 | MatMul、Gemm                                                 | -                                                            |
| Maximum                | 取元素级最大值                                               | Max                                                          | -                                                            |
| MaxPoolFusion          | 最大池化                                                     | MaxPool、GlobalMaxPool                                       | -                                                            |
| Minimum                | 取元素级最小值                                               | Min                                                          | -                                                            |
| Mod                    | 返回除法元素的余数                                           | Mod                                                          | -                                                            |
| MulFusion              | 逐元素乘法                                                   | Mul                                                          | -                                                            |
| Neg                    | 逐元素求负数                                                 | Neg                                                          | -                                                            |
| NonMaxSuppression     | 非极大值抑制                                                 | NonMaxSuppression                                           | -                                                            |
| NonZero                | 返回输入张量中所有非零元素的索引                             | NonZero                                                      | -                                                            |
| OneHot                 | 将整数索引张量转换为独热编码（One-Hot）表示                  | OneHot                                                       | -                                                            |
| PadFusion              | 将输入张量加上指定的 padding，使其达到指定的大小             | Pad                                                          | 不支持int32类型。                                            |
| PowFusion              | 逐元素求幂                                                   | Pow                                                          | 仅支持指数为单个常数。                                       |
| PReLUFusion            | PRelu激活函数                                                | PRelu                                                        | -                                                            |
| RandomNormal           | 生成一个张量，其中的值从正态分布（高斯分布）中随机采样     | RandomNormal                                                 | -                                                            |
| Range                  | 生成某个区间内的元素                                         | Range                                                        | -                                                            |
| Reciprocal             | 返回倒数                                                     | Reciprocal                                                   | -                                                            |
| ReduceFusion           | 归约操作                                                     | ReduceMean、ReduceMax、ReduceMin、ReduceProd、ReduceSum、ReduceSumSquare、ReduceL2,ReduceL1、ReduceLogSum | -                                                            |
| Reshape                | 改变张量形状，总元素个数不变                                 | Reshape、Flatten                                             | -                                                            |
| Resize                 | 对输入张量进行上采样或调整大小                               | Resize、Upsample                                             | -                                                            |
| ReverseSequence        | 对输入张量的可变长度序列进行部分反转                         | ReverseSequence                                              | -                                                            |
| Round                  | 四舍五入到最接近的整数数值                                   | Round                                                        | -                                                            |
| ScatterNd              | 根据索引将更新张量中的值散射到输出张量的指定位置             | ScatterND                                                    | -                                                            |
| ScatterNdUpdate        | 使用给定值以及输入索引更新输入数据的值                       | ScatterNdUpdate                                              | -                                                            |
| Shape                  | 获得张量shape                                                | Shape                                                        | -                                                            |
| Sin                    | 逐元素计算正弦                                               | Sin                                                          | -                                                            |
| Size                   | 获取张量维度大小                                             | Size                                                         | -                                                            |
| SliceFusion            | 张量切片操作                                                 | Slice                                                        | -                                                            |
| Softmax                | 归一化操作                                                   | Softmax                                                      | -                                                            |
| SpaceToDepth           | 高度和宽度维度的值移至深度维度                               | SpaceToDepth                                                 | -                                                            |
| Splice                 | 沿指定轴连接输入张量的多个切片或范围。                       | Splice                                                       | -                                                            |
| Split                  | 将输入张量沿指定轴分割成多个较小的输出张量。                 | Split                                                        | -                                                            |
| Sqrt                   | 逐元素开根号                                                 | Sqrt                                                         | -                                                            |
| Squeeze                | 移除维度为1的维度                                            | Squeeze                                                      | -                                                            |
| StridedSlice           | Tensor切片                                                   | Slice、DynamicSlice                                          | -                                                            |
| SubFusion              | 逐元素相减                                                   | Sub                                                          | -                                                            |
| TileFusion             | 平铺给定矩阵                                                 | Tile                                                         | 不支持int8类型。                                             |
| TopKFusion             | 从输入张量中返回top K个元素                                  | TopK                                                         | -                                                            |
| Transpose              | Tensor转置                                                   | Transpose、Int8Transpose                                     | -                                                            |
| Tril                   | 下三角矩阵                                                   | Trilu（属性upper=0）                                         | -                                                            |
| Triu                   | 上三角矩阵                                                   | Trilu（属性upper=1）                                         | -                                                            |
| Unsqueeze              | 将输入张量添加一个新的维度                                   | Unsqueeze                                                    | -                                                            |
| Where                  | 元素选择                                                     | NonZero、Where                                               | -                                                            |
| 转换工具支持的其他算子 | -                                                            | Constant、Atan、Asin、Tan、Loop、Dropout、If、Identity、Int8GivenIntTensorFill、Int8GivenTensorFill、Int8Quantize、Int8Dequantize、LpNormalization | 转换工具支持，但不需要具体实现的算子，一般这类算子在转化工具中被优化而消失，如被融合掉或者使用其他算子代替。 |
