# MindSpore分布式算子支持

`Linux` `Ascend` `GPU` `CPU` `模型开发` `初级` `中级` `高级`

<!-- TOC -->

- [MindSpore分布式算子支持](#mindspore分布式算子支持)
    - [分布式算子](#分布式算子)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/note/source_zh_cn/operator_list_parallel.md" target="_blank"><img src="./_static/logo_source.png"></a>

## 分布式算子

| 操作名                | 约束
| :-----------         | :-----------
| [mindspore.ops.Abs](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Abs)    |  无
| [mindspore.ops.ACos](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.ACos)     |  无
| [mindspore.ops.Acosh](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Acosh)    |  无
| [mindspore.ops.ApproximateEqual](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.ApproximateEqual)    |  无
| [mindspore.ops.ArgMaxWithValue](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.ArgMaxWithValue)    |  输入在轴（axis）的维度进行切分时，分布式结果可能会和单机不一致
| [mindspore.ops.ArgMinWithValue](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.ArgMinWithValue)    |  输入在轴（axis）的维度进行切分时，分布式结果可能会和单机不一致
| [mindspore.ops.Asin](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Asin)    |  无
| [mindspore.ops.Asinh](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Asinh)    |  无
| [mindspore.ops.Assign](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Assign)    |  无
| [mindspore.ops.AssignAdd](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.AssignAdd)    |  无
| [mindspore.ops.AssignSub](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.AssignSub)    |  无
| [mindspore.ops.Atan](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Atan)    |  无
| [mindspore.ops.Atan2](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Atan2)    |  无
| [mindspore.ops.Atanh](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Atanh)    |  无
| [mindspore.ops.BatchMatMul](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.BatchMatMul)    |   不支持`transpose_a=True`
| [mindspore.ops.BesselI0e](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.BesselI0e)    |  无
| [mindspore.ops.BesselI1e](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.BesselI1e)    |  无
| [mindspore.ops.BiasAdd](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.BiasAdd)    |  无
| [mindspore.ops.BroadcastTo](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.BroadcastTo)    |  无
| [mindspore.ops.Cast](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Cast)    |  Auto Parallel和Semi Auto Parallel模式下，配置策略不生效
| [mindspore.ops.Ceil](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Ceil)    |  无
| [mindspore.ops.Concat](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Concat)    |  输入（input_x）在轴（axis）所对应的维度不能切分，切分后，在数学逻辑上和单机不等价
| [mindspore.ops.Cos](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Cos)    |  无
| [mindspore.ops.Cosh](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Cosh)    |  无
| [mindspore.ops.Div](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Div)    |  无
| [mindspore.ops.DivNoNan](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.DivNoNan)    |  无
| [mindspore.ops.Dropout](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Dropout)    |  不支持重复计算
| [mindspore.ops.DropoutDoMask](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.DropoutDoMask)    |  需和`DropoutGenMask`联合使用，不支持配置切分策略
| [mindspore.ops.DropoutGenMask](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.DropoutGenMask)    |  需和`DropoutDoMask`联合使用
| [mindspore.ops.Elu](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Elu)    |  无
| [mindspore.ops.EmbeddingLookup](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.EmbeddingLookup)    |  同GatherV2
| [mindspore.ops.Equal](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Equal)    | 无
| [mindspore.ops.Erf](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Erf)    |  无
| [mindspore.ops.Erfc](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Erfc)    |  无
| [mindspore.ops.Exp](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Exp)    |  无
| [mindspore.ops.ExpandDims](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.ExpandDims)    |  无
| [mindspore.ops.Expm1](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Expm1)    |  无
| [mindspore.ops.Floor](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Floor)    |  无
| [mindspore.ops.FloorDiv](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.FloorDiv)    |  无
| [mindspore.ops.FloorMod](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.FloorMod)    |  无
| [mindspore.ops.GatherV2](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.GatherV2)    |  仅支持1维和2维的input_params，并且input_params的最后一维要32字节对齐（出于性能考虑）；不支持标量input_indices；参数在轴（axis）所在维度切分时，不支持重复计算；不支持input_indices和input_params同时进行切分
| [mindspore.ops.Gelu](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Gelu)    |  无
| [mindspore.ops.Greater](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Greater)    |  无
| [mindspore.ops.GreaterEqual](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.GreaterEqual)    |  无
| [mindspore.ops.Inv](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Inv)    |  无
| [mindspore.ops.L2Normalize](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.L2Normalize)    |  输入（input_x）在轴（axis）对应的维度不能切，切分后，在数学逻辑上和单机不等价
| [mindspore.ops.Less](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Less)    |  无
| [mindspore.ops.LessEqual](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.LessEqual)    |  无
| [mindspore.ops.LogicalAnd](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.LogicalAnd)    |  无
| [mindspore.ops.LogicalNot](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.LogicalNot)    |  无
| [mindspore.ops.LogicalOr](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.LogicalOr)    |  无
| [mindspore.ops.Log](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Log)    |  无
| [mindspore.ops.Log1p](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Log1p)    |  无
| [mindspore.ops.LogSoftmax](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.LogSoftmax)    |  输入（logits）在轴（axis）对应的维度不可切分，切分后，在数学逻辑上和单机不等价
| [mindspore.ops.MatMul](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.MatMul)    |   不支持`transpose_a=True`
| [mindspore.ops.Maximum](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Maximum)    |  无
| [mindspore.ops.Minimum](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Minimum)    |  无
| [mindspore.ops.Mod](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Mod)    |  无
| [mindspore.ops.Mul](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Mul)    |  无
| [mindspore.ops.Neg](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Neg)    |  无
| [mindspore.ops.NotEqual](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.NotEqual)    |  无
| [mindspore.ops.OneHot](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.OneHot)    |  仅支持输入（indices）是1维的Tensor，切分策略要配置输出的切分策略，以及第1和第2个输入的切分策略
| [mindspore.ops.OnesLike](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.OnesLike)    |  无
| [mindspore.ops.Pack](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Pack)    |  无
| [mindspore.ops.Pow](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Pow)    |  无
| [mindspore.ops.PReLU](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.PReLU)    | weight的shape在非[1]的情况下，输入（input_x）的Channel维要和weight的切分方式一致
| [mindspore.ops.RealDiv](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.RealDiv)    |  无
| [mindspore.ops.Reciprocal](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Reciprocal)    |  无
| [mindspore.ops.ReduceMax](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.ReduceMax)    |  输入在轴（axis）的维度进行切分时，分布式结果可能会和单机不一致
| [mindspore.ops.ReduceMin](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.ReduceMin)    |  输入在轴（axis）的维度进行切分时，分布式结果可能会和单机不一致
| [mindspore.ops.ReduceSum](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.ReduceSum)    |  无
| [mindspore.ops.ReduceMean](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.ReduceMean)    |  无
| [mindspore.ops.ReLU](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.ReLU)    |  无
| [mindspore.ops.ReLU6](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.ReLU6)    |  无
| [mindspore.ops.ReLUV2](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.ReLUV2)    |  无
| [mindspore.ops.Reshape](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Reshape)    |  不支持配置切分策略
| [mindspore.ops.Round](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Round)    |  无
| [mindspore.ops.Rsqrt](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Rsqrt)    |  无
| [mindspore.ops.Sigmoid](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Sigmoid)    |  无
| [mindspore.ops.SigmoidCrossEntropyWithLogits](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.SigmoidCrossEntropyWithLogits)    |  无
| [mindspore.ops.Sign](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Sign)    |  无
| [mindspore.ops.Sin](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Sin)    |  无
| [mindspore.ops.Sinh](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Sinh)    |  无
| [mindspore.ops.Softmax](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Softmax)    |  输入（logits）在轴（axis）对应的维度不可切分，切分后，在数学逻辑上和单机不等价
| [mindspore.ops.SoftmaxCrossEntropyWithLogits](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.SoftmaxCrossEntropyWithLogits)    |  输入（logits、labels）的最后一维不能切分；有两个输出，正向的loss只支持取[0]
| [mindspore.ops.Softplus](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Softplus)    |  无
| [mindspore.ops.Softsign](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Softsign)    |  无
| [mindspore.ops.SparseGatherV2](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.SparseGatherV2)    |  同GatherV2
| [mindspore.ops.Split](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Split)    |  轴（axis）所对应的维度不能切分，切分后，在数学逻辑上和单机不等价
| [mindspore.ops.Sqrt](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Sqrt)    |  无
| [mindspore.ops.Square](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Square)    |  无
| [mindspore.ops.Squeeze](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Squeeze)    |  无
| [mindspore.ops.StridedSlice](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.StridedSlice)    |  仅支持值为全0的mask；需要切分的维度必须全部提取；输入在strides不为1对应的维度不支持切分
| [mindspore.ops.Sub](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Sub)    |  无
| [mindspore.ops.Tan](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Tan)    |  无
| [mindspore.ops.Tanh](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Tanh)    |  无
| [mindspore.ops.TensorAdd](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.TensorAdd)    |  无
| [mindspore.ops.Tile](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Tile)    |  仅支持对multiples配置切分策略
| [mindspore.ops.TopK](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.TopK)    |  最后一维不支持切分，切分后，在数学逻辑上和单机不等价
| [mindspore.ops.Transpose](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.Transpose)    |  无
| [mindspore.ops.ZerosLike](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.ops.html#mindspore.ops.ZerosLike)    |  无

> 重复计算是指，机器没有用满，比如：集群有8张卡跑分布式训练，切分策略只对输入切成了4份。这种情况下会发生重复计算。
