# MindSpore Implicit Type Conversion Operator List

`Linux` `Ascend` `GPU` `CPU` `Model Development` `Beginner` `Intermediate` `Expert`

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/docs/note/source_en/operator_list_implicit.md)

## Implicit Type Conversion

### conversion rules

- Scalar and Tensor operations: during operation, the scalar is automatically converted to Tensor, and the data type is consistent with the Tensor data type involved in the operation; when Tensor is a bool data type and the scalar is int or float, both the scalar and Tensor are converted to the Tensor with the data type of int32 or float32.
- Tensor operation of different data types: the priority of data type is bool < uint8 < int8 < int16 < int32 < int64 < float16 < float32 < float64, during the operation, first determine the data type with the relatively highest priority among the Tensors involved in the operation, and then convert the low priority data type Tensor to the relatively highest priority data type; when the Tensor of int8 and uint8 data types are operated, they are converted to int16 Tensor.
- Data type conversion of Parameter is not supported: If inferred according to the conversion rules, RuntimeError exception will be thrown when the data type conversion of Parameter defined in the network is required.

### data types involved in conversion

- bool
- int8
- uint8
- int16
- int32
- int64
- float16
- float32
- float64

### support ops

| op name
| :-----------
| [mindspore.ops.Assign](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Assign)
| [mindspore.ops.AssignSub](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.AssignSub)
| [mindspore.ops.ApplyMomentum](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyMomentum)
| [mindspore.ops.FusedSparseAdam](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FusedSparseAdam)
| [mindspore.ops.FusedSparseLazyAdam](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FusedSparseLazyAdam)
| [mindspore.ops.FusedSparseFtrl](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FusedSparseFtrl)
| [mindspore.ops.FusedSparseProximalAdagrad](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FusedSparseProximalAdagrad)
| [mindspore.ops.ApplyAdaMax](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyAdaMax)
| [mindspore.ops.ApplyAdadelta](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyAdadelta)
| [mindspore.ops.ApplyAdagrad](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyAdagrad)
| [mindspore.ops.ApplyAdagradV2](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyAdagradV2)
| [mindspore.ops.SparseApplyAdagrad](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SparseApplyAdagrad)
| [mindspore.ops.SparseApplyAdagradV2](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SparseApplyAdagradV2)
| [mindspore.ops.ApplyProximalAdagrad](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyProximalAdagrad)
| [mindspore.ops.SparseApplyProximalAdagrad](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SparseApplyProximalAdagrad)
| [mindspore.ops.ApplyAddSign](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyAddSign)
| [mindspore.ops.ApplyPowerSign](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyPowerSign)
| [mindspore.ops.ApplyGradientDescent](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyGradientDescent)
| [mindspore.ops.ApplyProximalGradientDescent](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyProximalGradientDescent)
| [mindspore.ops.SparseApplyFtrl](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SparseApplyFtrl)
| [mindspore.ops.SparseApplyFtrlV2](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SparseApplyFtrlV2)
| [mindspore.ops.BitwiseAnd](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BitwiseAnd)
| [mindspore.ops.BitwiseOr](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BitwiseOr)
| [mindspore.ops.BitwiseXor](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BitwiseXor)
| [mindspore.ops.TensorAdd](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.TensorAdd)
| [mindspore.ops.Sub](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Sub)
| [mindspore.ops.Mul](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Mul)
| [mindspore.ops.Pow](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Pow)
| [mindspore.ops.Minimum](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Minimum)
| [mindspore.ops.Maximum](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Maximum)
| [mindspore.ops.RealDiv](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.RealDiv)
| [mindspore.ops.Div](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Div)
| [mindspore.ops.DivNoNan](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.DivNoNan)
| [mindspore.ops.FloorDiv](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FloorDiv)
| [mindspore.ops.TruncateDiv](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.TruncateDiv)
| [mindspore.ops.TruncateMod](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.TruncateMod)
| [mindspore.ops.Mod](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Mod)
| [mindspore.ops.FloorMod](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FloorMod)
| [mindspore.ops.Atan2](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Atan2)
| [mindspore.ops.SquaredDifference](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SquaredDifference)
| [mindspore.ops.Xdivy](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Xdivy)
| [mindspore.ops.Xlogy](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Xlogy)
| [mindspore.ops.Equal](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Equal)
| [mindspore.ops.ApproximateEqual](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApproximateEqual)
| [mindspore.ops.NotEqual](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.NotEqual)
| [mindspore.ops.Greater](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Greater)
| [mindspore.ops.GreaterEqual](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.GreaterEqual)
| [mindspore.ops.Less](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Less)
| [mindspore.ops.LessEqual](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LessEqual)
| [mindspore.ops.LogicalAnd](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LogicalAnd)
| [mindspore.ops.LogicalOr](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LogicalOr)
| [mindspore.ops.ScatterNdUpdate](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterNdUpdate)
| [mindspore.ops.ScatterNdAdd](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterNdAdd)
| [mindspore.ops.ScatterNdSub](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterNdSub)
| [mindspore.ops.ScatterNonAliasingAdd](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterNonAliasingAdd)
| [mindspore.ops.ScatterUpdate](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterUpdate)
| [mindspore.ops.ScatterMax](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterMax)
| [mindspore.ops.ScatterMin](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterMin)
| [mindspore.ops.ScatterAdd](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterAdd)
| [mindspore.ops.ScatterSub](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterSub)
| [mindspore.ops.ScatterMul](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterMul)
| [mindspore.ops.ScatterDiv](https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterDiv)

>