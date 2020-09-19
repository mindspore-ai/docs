# MindSpore分布式算子支持

`Linux` `Ascend` `GPU` `CPU` `模型开发` `初级` `中级` `高级`

<!-- TOC -->

- [MindSpore分布式算子支持](#mindspore分布式算子支持)
    - [分布式算子](#分布式算子)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.0/docs/note/source_zh_cn/operator_list_parallel.md" target="_blank"><img src="./_static/logo_source.png"></a>

## 分布式算子

| 操作名                | 约束
| :-----------         | :-----------
| [mindspore.ops.operations.ACos](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ACos)     |  None
| [mindspore.ops.operations.Cos](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Cos)    |  None
| [mindspore.ops.operations.LogicalNot](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LogicalNot)    |  None
| [mindspore.ops.operations.Log](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Log)    |  None
| [mindspore.ops.operations.Exp](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Exp)    |  None
| [mindspore.ops.operations.LogSoftmax](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LogSoftmax)    |  输入（logits）在轴（axis）对应的维度不可切分，切分后，在数学逻辑上和单机不等价
| [mindspore.ops.operations.Softmax](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Softmax)    |  输入（logits）在轴（axis）对应的维度不可切分，切分后，在数学逻辑上和单机不等价
| [mindspore.ops.operations.Tanh](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Tanh)    |  None
| [mindspore.ops.operations.Gelu](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Gelu)    |  None
| [mindspore.ops.operations.ReLU](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReLU)    |  None
| [mindspore.ops.operations.Sqrt](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Sqrt)    |  None
| [mindspore.ops.operations.Cast](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Cast)    |  None
| [mindspore.ops.operations.Neg](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Neg)    |  None
| [mindspore.ops.operations.ExpandDims](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ExpandDims)    |  None
| [mindspore.ops.operations.Squeeze](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Squeeze)    |  None
| [mindspore.ops.operations.Square](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Square)    |  None
| [mindspore.ops.operations.Sigmoid](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Sigmoid)    |  None
| [mindspore.ops.operations.Dropout](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Dropout)    |  不支持重复计算
| [mindspore.ops.operations.Div](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Div)    |  None
| [mindspore.ops.operations.TensorAdd](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.TensorAdd)    |  None
| [mindspore.ops.operations.RealDiv](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.RealDiv)    |  None
| [mindspore.ops.operations.Mul](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Mul)    |  None
| [mindspore.ops.operations.Sub](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Sub)    |  None
| [mindspore.ops.operations.Pow](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Pow)    |  None
| [mindspore.ops.operations.FloorDiv](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FloorDiv)    |  None
| [mindspore.ops.operations.Greater](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Greater)    |  None
| [mindspore.ops.operations.AssignSub](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.AssignSub)    |  None
| [mindspore.ops.operations.SigmoidCrossEntropyWithLogits](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SigmoidCrossEntropyWithLogits)    |  None
| [mindspore.ops.operations.Equal](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Equal)    | None
| [mindspore.ops.operations.NotEqual](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.NotEqual)    |  None
| [mindspore.ops.operations.Maximum](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Maximum)    |  None
| [mindspore.ops.operations.Minimum](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Minimum)    |  None
| [mindspore.ops.operations.BiasAdd](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BiasAdd)    |  None
| [mindspore.ops.operations.Concat](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Concat)    |  输入（input_x）在轴（axis）所对应的维度不能切分，切分后，在数学逻辑上和单机不等价
| [mindspore.ops.operations.DropoutGenMask](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.DropoutGenMask)    |  需和`DropoutDoMask`联合使用
| [mindspore.ops.operations.DropoutDoMask](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.DropoutDoMask)    |  需和`DropoutGenMask`联合使用，不支持配置切分策略
| [mindspore.ops.operations.GatherV2](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.GatherV2)    |  仅支持1维和2维的input_params，并且input_params的最后一维要32字节对齐（出于性能考虑）；不支持标量input_indices；参数在轴（axis）所在维度切分时，不支持重复计算；不支持input_indices和input_params同时进行切分
| [mindspore.ops.operations.SparseGatherV2](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseGatherV2)    |  同GatherV2
| [mindspore.ops.operations.EmbeddingLookup](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.EmbeddingLookup)    |  同GatherV2
| [mindspore.ops.operations.L2Normalize](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.L2Normalize)    | 输入（input_x）在轴（axis）对应的维度不能切，切分后，在数学逻辑上和单机不等价
| [mindspore.ops.operations.SoftmaxCrossEntropyWithLogits](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SoftmaxCrossEntropyWithLogits)    | 输入（logits、labels）的最后一维不能切分；有两个输出，正向的loss只支持取[0]
| [mindspore.ops.operations.MatMul](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.MatMul)    |  不支持`transpose_a=True`
| [mindspore.ops.operations.BatchMatMul](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BatchMatMul)    |  不支持`transpore_a=True`
| [mindspore.ops.operations.PReLU](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.PReLU)    |  输入（input_x）的Channel维要和weight的切分方式一致
| [mindspore.ops.operations.OneHot](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.OneHot)    |  仅支持输入（indices）是1维的Tensor
| [mindspore.ops.operations.ReduceSum](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReduceSum)    |  None
| [mindspore.ops.operations.ReduceMax](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReduceMax)    |  None
| [mindspore.ops.operations.ReduceMin](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReduceMin)    |  None
| [mindspore.ops.operations.ArgMinWithValue](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ArgMinWithValue)    |  第一个输出(index)不能作为其他算子的输入
| [mindspore.ops.operations.ArgMaxWithValue](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ArgMaxWithValue)    |  第一个输出(index)不能作为其他算子的输入
| [mindspore.ops.operations.ReduceMean](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReduceMean)    |  None
| [mindspore.ops.operations.Reshape](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Reshape)    |  不支持配置切分策略
| [mindspore.ops.operations.StridedSlice](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.StridedSlice)    |  仅支持值为全0的mask；需要切分的维度必须全部提取；输入在strides不为1对应的维度不支持切分
| [mindspore.ops.operations.Tile](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Tile)    |  仅支持对multiples配置切分策略
| [mindspore.ops.operations.Transpose](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Transpose)    |  None
| [mindspore.ops.operations.Diag](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Diag)    |  不支持配置切分策略

> 重复计算是指，机器没有用满，比如：集群有8张卡跑分布式训练，切分策略只对输入切成了4份。这种情况下会发生重复计算。
