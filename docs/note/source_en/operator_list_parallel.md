# MindSpore Distributed Operator List

`Linux` `Ascend` `GPU` `CPU` `Model Development` `Beginner` `Intermediate` `Expert`

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/docs/note/source_en/operator_list_parallel.md)

## Distributed Operator

| op name                | constraints
| :-----------         | :-----------
| [mindspore.ops.ACos](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ACos)     |  None
| [mindspore.ops.Cos](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Cos)    |  None
| [mindspore.ops.LogicalNot](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LogicalNot)    |  None
| [mindspore.ops.Log](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Log)    |  None
| [mindspore.ops.Exp](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Exp)    |  None
| [mindspore.ops.LogSoftmax](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LogSoftmax)    |  The logits can't be split into the dimension of axis, otherwise it's inconsistent with the single machine in the mathematical logic.
| [mindspore.ops.Softmax](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Softmax)    |  The logits can't be split into the dimension of axis, otherwise it's inconsistent with the single machine in the mathematical logic.
| [mindspore.ops.Tanh](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Tanh)    |  None
| [mindspore.ops.Gelu](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Gelu)    |  None
| [mindspore.ops.ReLU](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReLU)    |  None
| [mindspore.ops.Sqrt](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Sqrt)    |  None
| [mindspore.ops.Cast](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Cast)    |  None
| [mindspore.ops.Neg](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Neg)    |  None
| [mindspore.ops.ExpandDims](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ExpandDims)    |  None
| [mindspore.ops.Squeeze](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Squeeze)    |  None
| [mindspore.ops.Square](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Square)    |  None
| [mindspore.ops.Sigmoid](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Sigmoid)    |  None
| [mindspore.ops.Dropout](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Dropout)    |  Repeated calculation is not supported.
| [mindspore.ops.Div](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Div)    |  None
| [mindspore.ops.TensorAdd](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.TensorAdd)    |  None
| [mindspore.ops.RealDiv](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.RealDiv)    |  None
| [mindspore.ops.Mul](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Mul)    |  None
| [mindspore.ops.Sub](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Sub)    |  None
| [mindspore.ops.Pow](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Pow)    |  None
| [mindspore.ops.FloorDiv](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FloorDiv)    |  None
| [mindspore.ops.Greater](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Greater)    |  None
| [mindspore.ops.AssignSub](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.AssignSub)    |  None
| [mindspore.ops.SigmoidCrossEntropyWithLogits](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SigmoidCrossEntropyWithLogits)    |  None
| [mindspore.ops.Equal](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Equal)    | None
| [mindspore.ops.NotEqual](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.NotEqual)    |  None
| [mindspore.ops.Maximum](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Maximum)    |  None
| [mindspore.ops.Minimum](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Minimum)    |  None
| [mindspore.ops.BiasAdd](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BiasAdd)    |  None
| [mindspore.ops.Concat](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Concat)    |  The input_x can't be split into the dimension of axis, otherwise it's inconsistent with the single machine in the mathematical logic.
| [mindspore.ops.DropoutGenMask](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.DropoutGenMask)    |  Need to be used in conjunction with `DropoutDoMask`.
| [mindspore.ops.DropoutDoMask](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.DropoutDoMask)    |  Need to be used in conjunction with `DropoutGenMask`，configuring shard strategy is not supported.
| [mindspore.ops.GatherV2](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.GatherV2)    |  Only support 1-dim and 2-dim parameters and the last dimension of the input_params should be 32-byte aligned; Scalar input_indices is not supported; Repeated calculation is not supported when the parameters are split in the dimension of the axis; Split input_indices and input_params at the same time is not supported.
| [mindspore.ops.SparseGatherV2](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SparseGatherV2)    |  The same as GatherV2.
| [mindspore.ops.EmbeddingLookup](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.EmbeddingLookup)    |  The same as GatherV2.
| [mindspore.ops.L2Normalize](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.L2Normalize)    |  The input_x can't be split into the dimension of axis, otherwise it's inconsistent with the single machine in the mathematical logic.
| [mindspore.ops.SoftmaxCrossEntropyWithLogits](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SoftmaxCrossEntropyWithLogits)    | The last dimension of logits and labels can't be splited; Only supports using output[0].
| [mindspore.ops.MatMul](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.MatMul)    |   `transpose_a=True` is not supported.
| [mindspore.ops.BatchMatMul](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BatchMatMul)    |   `transpore_a=True` is not supported.
| [mindspore.ops.PReLU](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.PReLU)    |  When the shape of weight is not [1], the shard strategy in channel dimension of input_x should be consistent with weight.
| [mindspore.ops.OneHot](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.OneHot)    |  Only support 1-dim indices. Must configure strategy for the output and the first and second inputs.
| [mindspore.ops.ReduceSum](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReduceSum)    |  None
| [mindspore.ops.ReduceMax](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReduceMax)    |  When the input_x is splited on the axis dimension, the distributed result may be inconsistent with that on the single machine.
| [mindspore.ops.ReduceMin](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReduceMin)    |  When the input_x is splited on the axis dimension, the distributed result may be inconsistent with that on the single machine.
| [mindspore.ops.ArgMinWithValue](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ArgMinWithValue)    |  When the input_x is splited on the axis dimension, the distributed result may be inconsistent with that on the single machine.
| [mindspore.ops.ArgMaxWithValue](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ArgMaxWithValue)    |  When the input_x is splited on the axis dimension, the distributed result may be inconsistent with that on the single machine.
| [mindspore.ops.ReduceMean](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ReduceMean)    |  None
| [mindspore.ops.Reshape](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Reshape)    |  Configuring shard strategy is not supported.
| [mindspore.ops.StridedSlice](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.StridedSlice)    |  Only support mask with all 0 values; The dimension needs to be split should be all extracted; Split is not supported when the strides of dimension is 1.
| [mindspore.ops.Tile](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Tile)    |  Only support configuring shard strategy for multiples.
| [mindspore.ops.Transpose](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Transpose)    |  None

> Repeated calculation means that the device is not fully used. For example, the cluster has 8 devices to run distributed training, the splitting strategy only cuts the input into 4 copies. In this case, double counting will occur.
>
