# MindSpore Implicit Type Conversion Operator List

`Linux` `Ascend` `GPU` `CPU` `Model Development` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [MindSpore Implicit Type Conversion Operator List](#mindspore-implicit-type-conversion-operator-list)
    - [Implicit Type Conversion](#implicit-type-conversion)
        - [conversion rules](#conversion-rules)
        - [data types involved in conversion](#data-types-involved-in-conversion)
        - [support ops](#support-ops)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.0/docs/note/source_en/operator_list_implicit.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Implicit Type Conversion

### conversion rules
* Scalar and Tensor operations: during operation, the scalar is automatically converted to Tensor, and the data type is consistent with the Tensor data type involved in the operation;
when Tensor is a bool data type and the scalar is int or float, both the scalar and Tensor are converted to the Tensor with the data type of int32 or float32.
* Tensor operation of different data types: the priority of data type is bool < uint8 < int8 < int16 < int32 < int64 < float16 < float32 <float64,
during the operation, first determine the data type with the relatively highest priority among the Tensors involved in the operation, and then convert the low priority data type Tensor to the relatively highest priority data type;
when the Tensor of int8 and uint8 data types are operated, they are converted to int16 Tensor.
* Data type conversion of Parameter is not supported: If inferred according to the conversion rules, RuntimeError exception will be thrown when the data type conversion of Parameter defined in the network is required.

### data types involved in conversion
* bool
* int8
* uint8
* int16
* int32
* int64
* float16
* float32
* float64
  
### support ops

| op name
| :-----------
| [mindspore.ops.operations.Assign](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Assign)
| [mindspore.ops.operations.AssignSub](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.AssignSub)
| [mindspore.ops.operations.ApplyMomentum](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyMomentum)
| [mindspore.ops.operations.FusedSparseAdam](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FusedSparseAdam)
| [mindspore.ops.operations.FusedSparseLazyAdam](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FusedSparseLazyAdam)
| [mindspore.ops.operations.FusedSparseFtrl](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FusedSparseFtrl)
| [mindspore.ops.operations.FusedSparseProximalAdagrad](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FusedSparseProximalAdagrad)
| [mindspore.ops.operations.ApplyAdaMax](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyAdaMax)
| [mindspore.ops.operations.ApplyAdadelta](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyAdadelta)
| [mindspore.ops.operations.ApplyAdagrad](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyAdagrad)
| [mindspore.ops.operations.ApplyAdagradV2](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyAdagradV2)
| [mindspore.ops.operations.SparseApplyAdagrad](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyAdagrad)
| [mindspore.ops.operations.SparseApplyAdagradV2](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyAdagradV2)
| [mindspore.ops.operations.ApplyProximalAdagrad](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyProximalAdagrad)
| [mindspore.ops.operations.SparseApplyProximalAdagrad](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyProximalAdagrad)
| [mindspore.ops.operations.ApplyAddSign](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyAddSign)
| [mindspore.ops.operations.ApplyPowerSign](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyPowerSign)
| [mindspore.ops.operations.ApplyGradientDescent](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyGradientDescent)
| [mindspore.ops.operations.ApplyProximalGradientDescent](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyProximalGradientDescent)
| [mindspore.ops.operations.SparseApplyFtrl](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyFtrl)
| [mindspore.ops.operations.SparseApplyFtrlV2](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyFtrlV2)
| [mindspore.ops.operations.BitwiseAnd](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BitwiseAnd)
| [mindspore.ops.operations.BitwiseOr](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BitwiseOr)
| [mindspore.ops.operations.BitwiseXor](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BitwiseXor)
| [mindspore.ops.operations.TensorAdd](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.TensorAdd)
| [mindspore.ops.operations.Sub](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Sub)
| [mindspore.ops.operations.Mul](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Mul)
| [mindspore.ops.operations.Pow](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Pow)
| [mindspore.ops.operations.Minimum](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Minimum)
| [mindspore.ops.operations.Maximum](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Maximum)
| [mindspore.ops.operations.RealDiv](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.RealDiv)
| [mindspore.ops.operations.Div](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Div)
| [mindspore.ops.operations.DivNoNan](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.DivNoNan)
| [mindspore.ops.operations.FloorDiv](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FloorDiv)
| [mindspore.ops.operations.TruncateDiv](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.TruncateDiv)
| [mindspore.ops.operations.TruncateMod](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.TruncateMod)
| [mindspore.ops.operations.Mod](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Mod)
| [mindspore.ops.operations.FloorMod](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FloorMod)
| [mindspore.ops.operations.Atan2](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Atan2)
| [mindspore.ops.operations.SquaredDifference](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SquaredDifference)
| [mindspore.ops.operations.Xdivy](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Xdivy)
| [mindspore.ops.operations.Xlogy](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Xlogy)
| [mindspore.ops.operations.Equal](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Equal)
| [mindspore.ops.operations.ApproximateEqual](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApproximateEqual)
| [mindspore.ops.operations.NotEqual](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.NotEqual)
| [mindspore.ops.operations.Greater](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Greater)
| [mindspore.ops.operations.GreaterEqual](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.GreaterEqual)
| [mindspore.ops.operations.Less](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Less)
| [mindspore.ops.operations.LessEqual](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LessEqual)
| [mindspore.ops.operations.LogicalAnd](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LogicalAnd)
| [mindspore.ops.operations.LogicalOr](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LogicalOr)
| [mindspore.ops.operations.ScatterNdUpdate](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterNdUpdate)
| [mindspore.ops.operations.ScatterNdAdd](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterNdAdd)
| [mindspore.ops.operations.ScatterNdSub](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterNdSub)
| [mindspore.ops.operations.ScatterNonAliasingAdd](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterNonAliasingAdd)
| [mindspore.ops.operations.ScatterUpdate](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterUpdate)
| [mindspore.ops.operations.ScatterMax](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterMax)
| [mindspore.ops.operations.ScatterMin](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterMin)
| [mindspore.ops.operations.ScatterAdd](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterAdd)
| [mindspore.ops.operations.ScatterSub](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterSub)
| [mindspore.ops.operations.ScatterMul](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterMul)
| [mindspore.ops.operations.ScatterDiv](https://www.mindspore.cn/api/en/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterDiv)

