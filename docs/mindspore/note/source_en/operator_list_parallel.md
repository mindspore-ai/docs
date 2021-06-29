# MindSpore Distributed Operator List

`Linux` `Ascend` `GPU` `CPU` `Model Development` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [MindSpore Distributed Operator List](#mindspore-distributed-operator-list)
    - [Distributed Operator](#distributed-operator)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/note/source_en/operator_list_parallel.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

## Distributed Operator

| op name                                                      | constraints                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [mindspore.ops.Abs](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Abs.html) | None                                                         |
| [mindspore.ops.ACos](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.ACos.html) | None                                                         |
| [mindspore.ops.Acosh](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Acosh.html) | None                                                         |
| [mindspore.ops.Add](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Add.html) | None                                                         |
| [mindspore.ops.ApproximateEqual](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.ApproximateEqual.html) | None                                                         |
| [mindspore.ops.ArgMaxWithValue](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.ArgMaxWithValue.html) | When the input_x is splited on the axis dimension, the distributed result may be inconsistent with that on the single machine. |
| [mindspore.ops.ArgMinWithValue](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.ArgMinWithValue.html) | When the input_x is splited on the axis dimension, the distributed result may be inconsistent with that on the single machine. |
| [mindspore.ops.Asin](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Asin.html) | None                                                         |
| [mindspore.ops.Asinh](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Asinh.html) | None                                                         |
| [mindspore.ops.Assign](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Assign.html) | None                                                         |
| [mindspore.ops.AssignAdd](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.AssignAdd.html) | None                                                         |
| [mindspore.ops.AssignSub](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.AssignSub.html) | None                                                         |
| [mindspore.ops.Atan](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Atan.html) | None                                                         |
| [mindspore.ops.Atan2](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Atan2.html) | None                                                         |
| [mindspore.ops.Atanh](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Atanh.html) | None                                                         |
| [mindspore.ops.BatchMatMul](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.BatchMatMul.html) | `transpore_a=True` is not supported.                         |
| [mindspore.ops.BesselI0e](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.BesselI0e.html) | None                                                         |
| [mindspore.ops.BesselI1e](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.BesselI1e.html) | None                                                         |
| [mindspore.ops.BiasAdd](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.BiasAdd.html) | None                                                         |
| [mindspore.ops.BroadcastTo](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.BroadcastTo.html) | None                                                         |
| [mindspore.ops.Cast](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Cast.html) | The shard strategy is ignored in the Auto Parallel and Semi Auto Parallel mode. |
| [mindspore.ops.Ceil](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Ceil.html) | None                                                         |
| [mindspore.ops.Concat](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Concat.html) | The input_x can't be split into the dimension of axis, otherwise it's inconsistent with the single machine in the mathematical logic. |
| [mindspore.ops.Cos](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Cos.html) | None                                                         |
| [mindspore.ops.Cosh](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Cosh.html) | None                                                         |
| [mindspore.ops.Div](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Div.html) | None                                                         |
| [mindspore.ops.DivNoNan](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.DivNoNan.html) | None                                                         |
| [mindspore.ops.Dropout](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Dropout.html) | None                                                         |
| [mindspore.ops.DropoutDoMask](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.DropoutDoMask.html) | Need to be used in conjunction with `DropoutGenMask`         |
| [mindspore.ops.DropoutGenMask](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.DropoutGenMask.html) | Need to be used in conjunction with `DropoutDoMask`, configuring shard strategy is not supported. |
| [mindspore.ops.Elu](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Elu.html) | None                                                         |
| [mindspore.ops.EmbeddingLookup](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.EmbeddingLookup.html) | The same as Gather.                                          |
| [mindspore.ops.Equal](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Equal.html) | None                                                         |
| [mindspore.ops.Erf](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Erf.html) | None                                                         |
| [mindspore.ops.Erfc](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Erfc.html) | None                                                         |
| [mindspore.ops.Exp](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Exp.html) | None                                                         |
| [mindspore.ops.ExpandDims](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.ExpandDims.html) | None                                                         |
| [mindspore.ops.Expm1](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Expm1.html) | None                                                         |
| [mindspore.ops.Floor](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Floor.html) | None                                                         |
| [mindspore.ops.FloorDiv](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.FloorDiv.html) | None                                                         |
| [mindspore.ops.FloorMod](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.FloorMod.html) | None                                                         |
| [mindspore.ops.Gather](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Gather.html) | Only support 1-dim and 2-dim parameters and the last dimension of the input_params should be 32-byte aligned; Scalar input_indices is not supported; Repeated calculation is not supported when the parameters are split in the dimension of the axis; Split input_indices and input_params at the same time is not supported. |
| [mindspore.ops.GatherNd](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.GatherNd.html) | The first input can't be split, and the last dimension of the second input can't be split; In auto_parallel mode, the strategy's searching algorithm can not use "recursive_programming". |
| [mindspore.ops.GeLU](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.GeLU.html) | None                                                         |
| [mindspore.ops.Greater](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Greater.html) | None                                                         |
| [mindspore.ops.GreaterEqual](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.GreaterEqual.html) | None                                                         |
| [mindspore.ops.Inv](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Inv.html) | None                                                         |
| [mindspore.ops.L2Normalize](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.L2Normalize.html) | The input_x can't be split into the dimension of axis, otherwise it's inconsistent with the single machine in the mathematical logic. |
| [mindspore.ops.Less](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Less.html) | None                                                         |
| [mindspore.ops.LessEqual](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.LessEqual.html) | None                                                         |
| [mindspore.ops.LogicalAnd](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.LogicalAnd.html) | None                                                         |
| [mindspore.ops.LogicalNot](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.LogicalNot.html) | None                                                         |
| [mindspore.ops.LogicalOr](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.LogicalOr.html) | None                                                         |
| [mindspore.ops.Log](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Log.html) | None                                                         |
| [mindspore.ops.Log1p](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Log1p.html) | None                                                         |
| [mindspore.ops.LogSoftmax](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.LogSoftmax.html) | The logits can't be split into the dimension of axis, otherwise it's inconsistent with the single machine in the mathematical logic. |
| [mindspore.ops.MatMul](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.MatMul.html) | `transpose_a=True` is not supported.                         |
| [mindspore.ops.Maximum](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Maximum.html) | None                                                         |
| [mindspore.ops.Minimum](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Minimum.html) | None                                                         |
| [mindspore.ops.Mod](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Mod.html) | None                                                         |
| [mindspore.ops.Mul](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Mul.html) | None                                                         |
| [mindspore.ops.Neg](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Neg.html) | None                                                         |
| [mindspore.ops.NotEqual](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.NotEqual.html) | None                                                         |
| [mindspore.ops.OneHot](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.OneHot.html) | Only support 1-dim indices. Must configure strategy for the output and the first and second inputs. |
| [mindspore.ops.OnesLike](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.OnesLike.html) | None                                                         |
| [mindspore.ops.Pow](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Pow.html) | None                                                         |
| [mindspore.ops.PReLU](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.PReLU.html) | When the shape of weight is not [1], the shard strategy in channel dimension of input_x should be consistent with weight. |
| [mindspore.ops.RealDiv](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.RealDiv.html) | None                                                         |
| [mindspore.ops.Reciprocal](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Reciprocal.html) | None                                                         |
| [mindspore.ops.ReduceMax](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.ReduceMax.html) | When the input_x is splited on the axis dimension, the distributed result may be inconsistent with that on the single machine. |
| [mindspore.ops.ReduceMin](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.ReduceMin.html) | When the input_x is splited on the axis dimension, the distributed result may be inconsistent with that on the single machine. |
| [mindspore.ops.ReduceSum](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.ReduceSum.html) | None                                                         |
| [mindspore.ops.ReduceMean](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.ReduceMean.html) | None                                                         |
| [mindspore.ops.ReLU](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.ReLU.html) | None                                                         |
| [mindspore.ops.ReLU6](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.ReLU6.html) | None                                                         |
| [mindspore.ops.ReLUV2](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.ReLUV2.html) | None                                                         |
| [mindspore.ops.Reshape](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Reshape.html) | Configuring shard strategy is not supported. In auto parallel mode, if multiple operators are followed by the reshape operator, different shard strategys are not allowed to be configured for these operators. |
| [mindspore.ops.Round](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Round.html) | None                                                         |
| [mindspore.ops.Rsqrt](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Rsqrt.html) | None                                                         |
| [mindspore.ops.ScatterUpdate](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.ScatterUpdate.html) | The first dimension of first input can not be split, the second input can not  be split, and the first n dimensions (n is the dimension size of the second input) of the third input can not be split; In auto_parallel mode, the strategy's searching algorithm can not use "recursive_programming". |
| [mindspore.ops.Select](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Select.html) | In auto_parallel mode, the strategy's searching algorithm can not use "recursive_programming". |
| [mindspore.ops.Sigmoid](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Sigmoid.html) | None                                                         |
| [mindspore.ops.SigmoidCrossEntropyWithLogits](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.SigmoidCrossEntropyWithLogits.html) | None                                                         |
| [mindspore.ops.Sign](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Sign.html) | None                                                         |
| [mindspore.ops.Sin](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Sin.html) | None                                                         |
| [mindspore.ops.Sinh](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Sinh.html) | None                                                         |
| [mindspore.ops.Softmax](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Softmax.html) | The logits can't be split into the dimension of axis, otherwise it's inconsistent with the single machine in the mathematical logic. |
| [mindspore.ops.SoftmaxCrossEntropyWithLogits](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.SoftmaxCrossEntropyWithLogits.html) | The last dimension of logits and labels can't be splited; Only supports using output[0]. |
| [mindspore.ops.Softplus](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Softplus.html) | None                                                         |
| [mindspore.ops.Softsign](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Softsign.html) | None                                                         |
| [mindspore.ops.SparseGatherV2](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.SparseGatherV2.html) | The same as GatherV2.                                        |
| [mindspore.ops.Split](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Split.html) | The input_x can't be split into the dimension of axis, otherwise it's inconsistent with the single machine in the mathematical logic. |
| [mindspore.ops.Sqrt](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Sqrt.html) | None                                                         |
| [mindspore.ops.Square](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Square.html) | None                                                         |
| [mindspore.ops.Squeeze](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Squeeze.html) | None                                                         |
| [mindspore.ops.Stack](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Stack.html) | None                                                         |
| [mindspore.ops.StridedSlice](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.StridedSlice.html) | Only support mask with all 0 values; The dimension needs to be split should be all extracted; Split is supported when the strides of dimension is 1. |
| [mindspore.ops.Slice](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Slice.html) | The dimension needs to be split should be all extracted.     |
| [mindspore.ops.Sub](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Sub.html) | None                                                         |
| [mindspore.ops.Tan](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Tan.html) | None                                                         |
| [mindspore.ops.Tanh](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Tanh.html) | None                                                         |
| [mindspore.ops.Tile](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Tile.html) | Only support configuring shard strategy for multiples.       |
| [mindspore.ops.TopK](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.TopK.html) | The input_x can't be split into the last dimension, otherwise it's inconsistent with the single machine in the mathematical logic. |
| [mindspore.ops.Transpose](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Transpose.html) | None                                                         |
| [mindspore.ops.Unique](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.Unique.html) | Only support the repeat calculate shard strategy (1,).       |
| [mindspore.ops.UnsortedSegmentSum](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.UnsortedSegmentSum.html) | The shard of input_x and segment_ids must be the same as the dimension of segment_ids. |
| [mindspore.ops.UnsortedSegmentMin](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.UnsortedSegmentMin.html) | The shard of input_x and segment_ids must be the same as the dimension of segment_ids. Note that if the segment id i is missing, then the output[i] will be filled with the maximum of the input type. The user needs to mask the maximum value to avoid value overflow. The communication operation such as AllReudce will raise an Run Task Error due to overflow. |
| [mindspore.ops.UnsortedSegmentMax](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.UnsortedSegmentMax.html) | The shard of input_x and segment_ids must be the same as the dimension of segment_ids. Note that if the segment id i is missing, then the output[i] will be filled with the minimum of the input type. The user needs to mask the minimum value to avoid value overflow. The communication operation such as AllReudce will raise an Run Task Error due to overflow. |
| [mindspore.ops.ZerosLike](https://www.mindspore.cn/docs/api/en/r1.3/api_python/ops/mindspore.ops.ZerosLike.html) | None                                                         |

> Repeated calculation means that the device is not fully used. For example, the cluster has 8 devices to run distributed training, the splitting strategy only cuts the input into 4 copies. In this case, double counting will occur.
