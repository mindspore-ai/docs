# MindSpore Distributed Operator List

`Linux` `Ascend` `GPU` `CPU` `Model Development` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [MindSpore Distributed Operator List](#mindspore-distributed-operator-list)
    - [Distributed Operator](#distributed-operator)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/note/source_en/operator_list_parallel.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Distributed Operator

| op name                | constraints
| :-----------         | :-----------
| [mindspore.ops.Abs](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Abs)    |  None
| [mindspore.ops.ACos](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.ACos)     |  None
| [mindspore.ops.Acosh](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Acosh)    |  None
| [mindspore.ops.ApproximateEqual](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.ApproximateEqual)    |  None
| [mindspore.ops.ArgMaxWithValue](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.ArgMaxWithValue)    |  When the input_x is splited on the axis dimension, the distributed result may be inconsistent with that on the single machine.
| [mindspore.ops.ArgMinWithValue](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.ArgMinWithValue)    |  When the input_x is splited on the axis dimension, the distributed result may be inconsistent with that on the single machine.
| [mindspore.ops.Asin](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Asin)    |  None
| [mindspore.ops.Asinh](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Asinh)    |  None
| [mindspore.ops.Assign](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Assign)    |  None
| [mindspore.ops.AssignAdd](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.AssignAdd)    |  None
| [mindspore.ops.AssignSub](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.AssignSub)    |  None
| [mindspore.ops.Atan](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Atan)    |  None
| [mindspore.ops.Atan2](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Atan2)    |  None
| [mindspore.ops.Atanh](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Atanh)    |  None
| [mindspore.ops.BatchMatMul](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.BatchMatMul)    |   `transpore_a=True` is not supported.
| [mindspore.ops.BesselI0e](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.BesselI0e)    |  None
| [mindspore.ops.BesselI1e](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.BesselI1e)    |  None
| [mindspore.ops.BiasAdd](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.BiasAdd)    |  None
| [mindspore.ops.BroadcastTo](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.BroadcastTo)    |  None
| [mindspore.ops.Cast](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Cast)    |  The shard strategy is ignored in the Auto Parallel and Semi Auto Parallel mode. 
| [mindspore.ops.Ceil](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Ceil)    |  None
| [mindspore.ops.Concat](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Concat)    |  The input_x can't be split into the dimension of axis, otherwise it's inconsistent with the single machine in the mathematical logic.
| [mindspore.ops.Cos](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Cos)    |  None
| [mindspore.ops.Cosh](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Cosh)    |  None
| [mindspore.ops.Div](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Div)    |  None
| [mindspore.ops.DivNoNan](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.DivNoNan)    |  None
| [mindspore.ops.Dropout](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Dropout)    |  Repeated calculation is not supported.
| [mindspore.ops.DropoutDoMask](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.DropoutDoMask)    |  Need to be used in conjunction with `DropoutGenMask`，configuring shard strategy is not supported.
| [mindspore.ops.DropoutGenMask](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.DropoutGenMask)    |  Need to be used in conjunction with `DropoutDoMask`.
| [mindspore.ops.Elu](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Elu)    |  None
| [mindspore.ops.EmbeddingLookup](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.EmbeddingLookup)    |  The same as GatherV2.
| [mindspore.ops.Equal](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Equal)    | None
| [mindspore.ops.Erf](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Erf)    |  None
| [mindspore.ops.Erfc](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Erfc)    |  None
| [mindspore.ops.Exp](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Exp)    |  None
| [mindspore.ops.ExpandDims](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.ExpandDims)    |  None
| [mindspore.ops.Expm1](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Expm1)    |  None
| [mindspore.ops.Floor](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Floor)    |  None
| [mindspore.ops.FloorDiv](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.FloorDiv)    |  None
| [mindspore.ops.FloorMod](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.FloorMod)    |  None
| [mindspore.ops.GatherV2](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.GatherV2)    |  Only support 1-dim and 2-dim parameters and the last dimension of the input_params should be 32-byte aligned; Scalar input_indices is not supported; Repeated calculation is not supported when the parameters are split in the dimension of the axis; Split input_indices and input_params at the same time is not supported.
| [mindspore.ops.Gelu](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Gelu)    |  None
| [mindspore.ops.Greater](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Greater)    |  None
| [mindspore.ops.GreaterEqual](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.GreaterEqual)    |  None
| [mindspore.ops.Inv](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Inv)    |  None
| [mindspore.ops.L2Normalize](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.L2Normalize)    |  The input_x can't be split into the dimension of axis, otherwise it's inconsistent with the single machine in the mathematical logic.
| [mindspore.ops.Less](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Less)    |  None
| [mindspore.ops.LessEqual](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.LessEqual)    |  None
| [mindspore.ops.LogicalAnd](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.LogicalAnd)    |  None
| [mindspore.ops.LogicalNot](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.LogicalNot)    |  None
| [mindspore.ops.LogicalOr](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.LogicalOr)    |  None
| [mindspore.ops.Log](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Log)    |  None
| [mindspore.ops.Log1p](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Log1p)    |  None
| [mindspore.ops.LogSoftmax](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.LogSoftmax)    |  The logits can't be split into the dimension of axis, otherwise it's inconsistent with the single machine in the mathematical logic.
| [mindspore.ops.MatMul](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.MatMul)    |   `transpose_a=True` is not supported.
| [mindspore.ops.Maximum](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Maximum)    |  None
| [mindspore.ops.Minimum](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Minimum)    |  None
| [mindspore.ops.Mod](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Mod)    |  None
| [mindspore.ops.Mul](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Mul)    |  None
| [mindspore.ops.Neg](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Neg)    |  None
| [mindspore.ops.NotEqual](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.NotEqual)    |  None
| [mindspore.ops.OneHot](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.OneHot)    |  Only support 1-dim indices. Must configure strategy for the output and the first and second inputs.
| [mindspore.ops.OnesLike](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.OnesLike)    |  None
| [mindspore.ops.Pack](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Pack)    |  None
| [mindspore.ops.Pow](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Pow)    |  None
| [mindspore.ops.PReLU](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.PReLU)    |  When the shape of weight is not [1], the shard strategy in channel dimension of input_x should be consistent with weight.
| [mindspore.ops.RealDiv](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.RealDiv)    |  None
| [mindspore.ops.Reciprocal](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Reciprocal)    |  None
| [mindspore.ops.ReduceMax](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.ReduceMax)    |  When the input_x is splited on the axis dimension, the distributed result may be inconsistent with that on the single machine.
| [mindspore.ops.ReduceMin](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.ReduceMin)    |  When the input_x is splited on the axis dimension, the distributed result may be inconsistent with that on the single machine.
| [mindspore.ops.ReduceSum](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.ReduceSum)    |  None
| [mindspore.ops.ReduceMean](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.ReduceMean)    |  None
| [mindspore.ops.ReLU](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.ReLU)    |  None
| [mindspore.ops.ReLU6](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.ReLU6)    |  None
| [mindspore.ops.ReLUV2](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.ReLUV2)    |  None
| [mindspore.ops.Reshape](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Reshape)    |  Configuring shard strategy is not supported. In auto parallel mode, if multiple operators are followed by the reshape operator, different shard strategys are not allowed to be configured for these operators.
| [mindspore.ops.Round](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Round)    |  None
| [mindspore.ops.Rsqrt](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Rsqrt)    |  None
| [mindspore.ops.Sigmoid](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Sigmoid)    |  None
| [mindspore.ops.SigmoidCrossEntropyWithLogits](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.SigmoidCrossEntropyWithLogits)    |  None
| [mindspore.ops.Sign](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Sign)    |  None
| [mindspore.ops.Sin](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Sin)    |  None
| [mindspore.ops.Sinh](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Sinh)    |  None
| [mindspore.ops.Softmax](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Softmax)    |  The logits can't be split into the dimension of axis, otherwise it's inconsistent with the single machine in the mathematical logic.
| [mindspore.ops.SoftmaxCrossEntropyWithLogits](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.SoftmaxCrossEntropyWithLogits)    | The last dimension of logits and labels can't be splited; Only supports using output[0].
| [mindspore.ops.Softplus](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Softplus)    |  None
| [mindspore.ops.Softsign](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Softsign)    |  None
| [mindspore.ops.SparseGatherV2](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.SparseGatherV2)    |  The same as GatherV2.
| [mindspore.ops.Split](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Split)    |  The input_x can't be split into the dimension of axis, otherwise it's inconsistent with the single machine in the mathematical logic.
| [mindspore.ops.Sqrt](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Sqrt)    |  None
| [mindspore.ops.Square](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Square)    |  None
| [mindspore.ops.Squeeze](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Squeeze)    |  None
| [mindspore.ops.StridedSlice](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.StridedSlice)    |  Only support mask with all 0 values; The dimension needs to be split should be all extracted; Split is not supported when the strides of dimension is 1.
| [mindspore.ops.Sub](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Sub)    |  None
| [mindspore.ops.Tan](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Tan)    |  None
| [mindspore.ops.Tanh](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Tanh)    |  None
| [mindspore.ops.TensorAdd](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.TensorAdd)    |  None
| [mindspore.ops.Tile](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Tile)    |  Only support configuring shard strategy for multiples.
| [mindspore.ops.TopK](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.TopK)    |  The input_x can't be split into the last dimension, otherwise it's inconsistent with the single machine in the mathematical logic.
| [mindspore.ops.Transpose](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Transpose)    |  None
| [mindspore.ops.Unique](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.Unique)    |  Only support the repeat calculate shard strategy (1,).
| [mindspore.ops.UnsortedSegmentSum](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.UnsortedSegmentSum)    |  The shard of input_x and segment_ids must be the same as the dimension of segment_ids.
| [mindspore.ops.UnsortedSegmentMin](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.UnsortedSegmentMin)    |  The shard of input_x and segment_ids must be the same as the dimension of segment_ids.
| [mindspore.ops.ZerosLike](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.ops.html#mindspore.ops.ZerosLike)    |  None

> Repeated calculation means that the device is not fully used. For example, the cluster has 8 devices to run distributed training, the splitting strategy only cuts the input into 4 copies. In this case, double counting will occur.
>
