# MindSpore Distributed Operator List

`Linux` `Ascend` `GPU` `CPU` `Model Development` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [MindSpore Distributed Operator List](#mindspore-distributed-operator-list)
    - [Distributed Operator](#distributed-operator)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.0/docs/source_en/operator_list_parallel.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Distributed Operator

| op name                | constraints
| :-----------         | :-----------
| [mindspore.ops.operations.ACos](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ACos)     |  None
| [mindspore.ops.operations.Cos](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Cos)    |  None
| [mindspore.ops.operations.LogicalNot](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LogicalNot)    |  None
| [mindspore.ops.operations.Log](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Log)    |  None
| [mindspore.ops.operations.Exp](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Exp)    |  None
| [mindspore.ops.operations.LogSoftmax](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LogSoftmax)    |  The logits can't be split into the dimension of axis, otherwise it's inconsistent with the single machine in the mathematical logic.
| [mindspore.ops.operations.Softmax](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Softmax)    |  The logits can't be split into the dimension of axis, otherwise it's inconsistent with the single machine in the mathematical logic.
| [mindspore.ops.operations.Tanh](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Tanh)    |  None
| [mindspore.ops.operations.Gelu](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Gelu)    |  None
| [mindspore.ops.operations.ReLU](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReLU)    |  None
| [mindspore.ops.operations.Sqrt](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Sqrt)    |  None
| [mindspore.ops.operations.Cast](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Cast)    |  None
| [mindspore.ops.operations.Neg](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Neg)    |  None
| [mindspore.ops.operations.ExpandDims](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ExpandDims)    |  None
| [mindspore.ops.operations.Squeeze](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Squeeze)    |  None
| [mindspore.ops.operations.Square](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Square)    |  None
| [mindspore.ops.operations.Sigmoid](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Sigmoid)    |  None
| [mindspore.ops.operations.Dropout](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Dropout)    |  Repeated calculation is not supported.
| [mindspore.ops.operations.Div](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Div)    |  None
| [mindspore.ops.operations.TensorAdd](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.TensorAdd)    |  None
| [mindspore.ops.operations.RealDiv](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.RealDiv)    |  None
| [mindspore.ops.operations.Mul](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Mul)    |  None
| [mindspore.ops.operations.Sub](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Sub)    |  None
| [mindspore.ops.operations.Pow](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Pow)    |  None
| [mindspore.ops.operations.FloorDiv](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FloorDiv)    |  None
| [mindspore.ops.operations.Greater](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Greater)    |  None
| [mindspore.ops.operations.AssignSub](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.AssignSub)    |  None
| [mindspore.ops.operations.SigmoidCrossEntropyWithLogits](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SigmoidCrossEntropyWithLogits)    |  None
| [mindspore.ops.operations.Equal](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Equal)    | None
| [mindspore.ops.operations.NotEqual](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.NotEqual)    |  None
| [mindspore.ops.operations.Maximum](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Maximum)    |  None
| [mindspore.ops.operations.Minimum](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Minimum)    |  None
| [mindspore.ops.operations.BiasAdd](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BiasAdd)    |  None
| [mindspore.ops.operations.Concat](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Concat)    |  The input_x can't be split into the dimension of axis, otherwise it's inconsistent with the single machine in the mathematical logic.
| [mindspore.ops.operations.DropoutGenMask](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.DropoutGenMask)    |  Need to be used in conjunction with `DropoutDoMask`.
| [mindspore.ops.operations.DropoutDoMask](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.DropoutDoMask)    |  Need to be used in conjunction with `DropoutGenMask`，configuring shard strategy is not supported.
| [mindspore.ops.operations.GatherV2](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.GatherV2)    |  Only support 1-dim and 2-dim parameters and the last dimension of the input_params should be 32-byte aligned; Scalar input_indices is not supported; Repeated calculation is not supported when the parameters are split in the dimension of the axis; Split input_indices and input_params at the same time is not supported.
| [mindspore.ops.operations.SparseGatherV2](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseGatherV2)    |  The same as GatherV2.
| [mindspore.ops.operations.EmbeddingLookup](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.EmbeddingLookup)    |  The same as GatherV2.
| [mindspore.ops.operations.L2Normalize](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.L2Normalize)    |  The input_x can't be split into the dimension of axis, otherwise it's inconsistent with the single machine in the mathematical logic.
| [mindspore.ops.operations.SoftmaxCrossEntropyWithLogits](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SoftmaxCrossEntropyWithLogits)    | The last dimension of logits and labels can't be splited; Only supports using output[0].
| [mindspore.ops.operations.MatMul](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.MatMul)    |   `transpose_a=True` is not supported.
| [mindspore.ops.operations.BatchMatMul](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BatchMatMul)    |   `transpore_a=True` is not supported.
| [mindspore.ops.operations.PReLU](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.PReLU)    |  The shard strategy in channel dimension of input_x should be consistent with weight.
| [mindspore.ops.operations.OneHot](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.OneHot)    |  Only support 1-dim indices.
| [mindspore.ops.operations.ReduceSum](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReduceSum)    |  None
| [mindspore.ops.operations.ReduceMax](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReduceMax)    |  None
| [mindspore.ops.operations.ReduceMin](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReduceMin)    |  None
| [mindspore.ops.operations.ArgMinWithValue](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ArgMinWithValue)    |  The output index can't be used as the input of other operators.
| [mindspore.ops.operations.ArgMaxWithValue](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ArgMaxWithValue)    |  The output index can't be used as the input of other operators.
| [mindspore.ops.operations.ReduceMean](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ReduceMean)    |  None
| [mindspore.ops.operations.Reshape](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Reshape)    |  Configuring shard strategy is not supported.
| [mindspore.ops.operations.StridedSlice](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.StridedSlice)    |  Only support mask with all 0 values; The dimension needs to be split should be all extracted; Split is not supported when the strides of dimension is 1.
| [mindspore.ops.operations.Tile](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Tile)    |  Only support configuring shard strategy for multiples.
| [mindspore.ops.operations.Transpose](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Transpose)    |  None
| [mindspore.ops.operations.Diag](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Diag)    |  Configuring shard strategy is not supported.

> Repeated calculation means that the device is not fully used. For example, the cluster has 8 devices to run distributed training, the splitting strategy only cuts the input into 4 copies. In this case, double counting will occur.
>
