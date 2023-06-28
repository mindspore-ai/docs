# MindSpore分布式算子支持

`Linux` `Ascend` `GPU` `CPU` `模型开发` `初级` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.0/docs/note/source_zh_cn/operator_list_parallel.md" target="_blank"><img src="./_static/logo_source.png"></a>

## 分布式算子

| 操作名                | 约束
| :-----------         | :-----------
| [mindspore.ops.ACos](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ACos)     |  None
| [mindspore.ops.Cos](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Cos)    |  None
| [mindspore.ops.LogicalNot](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LogicalNot)    |  None
| [mindspore.ops.Log](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Log)    |  None
| [mindspore.ops.Exp](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Exp)    |  None
| [mindspore.ops.LogSoftmax](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LogSoftmax)    |  输入（logits）在轴（axis）对应的维度不可切分，切分后，在数学逻辑上和单机不等价
| [mindspore.ops.Softmax](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Softmax)    |  输入（logits）在轴（axis）对应的维度不可切分，切分后，在数学逻辑上和单机不等价
| [mindspore.ops.Tanh](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Tanh)    |  None
| [mindspore.ops.Gelu](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Gelu)    |  None
| [mindspore.ops.ReLU](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReLU)    |  None
| [mindspore.ops.Sqrt](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Sqrt)    |  None
| [mindspore.ops.Cast](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Cast)    |  None
| [mindspore.ops.Neg](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Neg)    |  None
| [mindspore.ops.ExpandDims](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ExpandDims)    |  None
| [mindspore.ops.Squeeze](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Squeeze)    |  None
| [mindspore.ops.Square](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Square)    |  None
| [mindspore.ops.Sigmoid](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Sigmoid)    |  None
| [mindspore.ops.Dropout](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Dropout)    |  不支持重复计算
| [mindspore.ops.Div](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Div)    |  None
| [mindspore.ops.TensorAdd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.TensorAdd)    |  None
| [mindspore.ops.RealDiv](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.RealDiv)    |  None
| [mindspore.ops.Mul](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Mul)    |  None
| [mindspore.ops.Sub](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Sub)    |  None
| [mindspore.ops.Pow](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Pow)    |  None
| [mindspore.ops.FloorDiv](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FloorDiv)    |  None
| [mindspore.ops.Greater](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Greater)    |  None
| [mindspore.ops.AssignSub](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.AssignSub)    |  None
| [mindspore.ops.SigmoidCrossEntropyWithLogits](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SigmoidCrossEntropyWithLogits)    |  None
| [mindspore.ops.Equal](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Equal)    | None
| [mindspore.ops.NotEqual](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.NotEqual)    |  None
| [mindspore.ops.Maximum](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Maximum)    |  None
| [mindspore.ops.Minimum](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Minimum)    |  None
| [mindspore.ops.BiasAdd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BiasAdd)    |  None
| [mindspore.ops.Concat](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Concat)    |  输入（input_x）在轴（axis）所对应的维度不能切分，切分后，在数学逻辑上和单机不等价
| [mindspore.ops.DropoutGenMask](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.DropoutGenMask)    |  需和`DropoutDoMask`联合使用
| [mindspore.ops.DropoutDoMask](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.DropoutDoMask)    |  需和`DropoutGenMask`联合使用，不支持配置切分策略
| [mindspore.ops.GatherV2](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.GatherV2)    |  仅支持1维和2维的input_params，并且input_params的最后一维要32字节对齐（出于性能考虑）；不支持标量input_indices；参数在轴（axis）所在维度切分时，不支持重复计算；不支持input_indices和input_params同时进行切分
| [mindspore.ops.SparseGatherV2](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SparseGatherV2)    |  同GatherV2
| [mindspore.ops.EmbeddingLookup](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.EmbeddingLookup)    |  同GatherV2
| [mindspore.ops.L2Normalize](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.L2Normalize)    | 输入（input_x）在轴（axis）对应的维度不能切，切分后，在数学逻辑上和单机不等价
| [mindspore.ops.SoftmaxCrossEntropyWithLogits](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SoftmaxCrossEntropyWithLogits)    | 输入（logits、labels）的最后一维不能切分；有两个输出，正向的loss只支持取[0]
| [mindspore.ops.MatMul](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.MatMul)    |  不支持`transpose_a=True`
| [mindspore.ops.BatchMatMul](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BatchMatMul)    |  不支持`transpore_a=True`
| [mindspore.ops.PReLU](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.PReLU)    |  weight的shape在非[1]的情况下，输入（input_x）的Channel维要和weight的切分方式一致
| [mindspore.ops.OneHot](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.OneHot)    |  仅支持输入（indices）是1维的Tensor，切分策略要配置输出的切分策略，以及第1和第2个输入的切分策略
| [mindspore.ops.ReduceSum](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReduceSum)    |  None
| [mindspore.ops.ReduceMax](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReduceMax)    |  输入在轴（axis）的维度进行切分时，分布式结果可能会和单机不一致
| [mindspore.ops.ReduceMin](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReduceMin)    |  输入在轴（axis）的维度进行切分时，分布式结果可能会和单机不一致
| [mindspore.ops.ArgMinWithValue](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ArgMinWithValue)    |  输入在轴（axis）的维度进行切分时，分布式结果可能会和单机不一致
| [mindspore.ops.ArgMaxWithValue](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ArgMaxWithValue)    |  输入在轴（axis）的维度进行切分时，分布式结果可能会和单机不一致
| [mindspore.ops.ReduceMean](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReduceMean)    |  None
| [mindspore.ops.Reshape](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Reshape)    |  不支持配置切分策略
| [mindspore.ops.StridedSlice](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.StridedSlice)    |  仅支持值为全0的mask；需要切分的维度必须全部提取；输入在strides不为1对应的维度不支持切分
| [mindspore.ops.Tile](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Tile)    |  仅支持对multiples配置切分策略
| [mindspore.ops.Transpose](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Transpose)    |  None

> 重复计算是指，机器没有用满，比如：集群有8张卡跑分布式训练，切分策略只对输入切成了4份。这种情况下会发生重复计算。
