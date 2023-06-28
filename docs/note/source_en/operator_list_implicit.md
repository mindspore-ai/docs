# MindSpore Implicit Type Conversion Operator List

`Linux` `Ascend` `GPU` `CPU` `Model Development` `Beginner` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r1.1/docs/note/source_en/operator_list_implicit.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Implicit Type Conversion

### conversion rules

- Scalar and Tensor operations: during operation, the scalar is automatically converted to Tensor, and the data type is consistent with the Tensor data type involved in the operation; when Tensor is bool data type and the scalar is int or float, both the scalar and Tensor are converted to the Tensor with the data type of int32 or float32; when Tensor is int or uint data type and the scalar is float, both the scalar and Tensor are converted to the Tensor with the data type of float32.
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

| op name                                                                                                                                                       |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [mindspore.ops.Assign](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.Assign.html)                                             |
| [mindspore.ops.AssignSub](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.AssignSub.html)                                       |
| [mindspore.ops.ApplyMomentum](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ApplyMomentum.html)                               |
| [mindspore.ops.FusedSparseAdam](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.FusedSparseAdam.html)                           |
| [mindspore.ops.FusedSparseLazyAdam](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.FusedSparseLazyAdam.html)                   |
| [mindspore.ops.FusedSparseFtrl](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.FusedSparseFtrl.html)                           |
| [mindspore.ops.FusedSparseProximalAdagrad](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.FusedSparseProximalAdagrad.html)     |
| [mindspore.ops.ApplyAdaMax](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ApplyAdaMax.html)                                   |
| [mindspore.ops.ApplyAdadelta](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ApplyAdadelta.html)                               |
| [mindspore.ops.ApplyAdagrad](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ApplyAdagrad.html)                                 |
| [mindspore.ops.ApplyAdagradV2](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ApplyAdagradV2.html)                             |
| [mindspore.ops.SparseApplyAdagrad](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.SparseApplyAdagrad.html)                     |
| [mindspore.ops.SparseApplyAdagradV2](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.SparseApplyAdagradV2.html)                 |
| [mindspore.ops.ApplyProximalAdagrad](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ApplyProximalAdagrad.html)                 |
| [mindspore.ops.SparseApplyProximalAdagrad](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.SparseApplyProximalAdagrad.html)     |
| [mindspore.ops.ApplyAddSign](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ApplyAddSign.html)                                 |
| [mindspore.ops.ApplyPowerSign](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ApplyPowerSign.html)                             |
| [mindspore.ops.ApplyGradientDescent](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ApplyGradientDescent.html)                 |
| [mindspore.ops.ApplyProximalGradientDescent](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ApplyProximalGradientDescent.html) |
| [mindspore.ops.SparseApplyFtrl](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.SparseApplyFtrl.html)                           |
| [mindspore.ops.SparseApplyFtrlV2](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.SparseApplyFtrlV2.html)                       |
| [mindspore.ops.BitwiseAnd](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.BitwiseAnd.html)                                     |
| [mindspore.ops.BitwiseOr](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.BitwiseOr.html)                                       |
| [mindspore.ops.BitwiseXor](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.BitwiseXor.html)                                     |
| [mindspore.ops.TensorAdd](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.TensorAdd.html)                                       |
| [mindspore.ops.Add](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.Add.html)                                                   |
| [mindspore.ops.Sub](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.Sub.html)                                                   |
| [mindspore.ops.Mul](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.Mul.html)                                                   |
| [mindspore.ops.Pow](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.Pow.html)                                                   |
| [mindspore.ops.Minimum](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.Minimum.html)                                           |
| [mindspore.ops.Maximum](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.Maximum.html)                                           |
| [mindspore.ops.RealDiv](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.RealDiv.html)                                           |
| [mindspore.ops.Div](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.Div.html)                                                   |
| [mindspore.ops.DivNoNan](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.DivNoNan.html)                                         |
| [mindspore.ops.FloorDiv](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.FloorDiv.html)                                         |
| [mindspore.ops.TruncateDiv](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.TruncateDiv.html)                                   |
| [mindspore.ops.TruncateMod](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.TruncateMod.html)                                   |
| [mindspore.ops.Mod](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.Mod.html)                                                   |
| [mindspore.ops.FloorMod](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.FloorMod.html)                                         |
| [mindspore.ops.Atan2](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.Atan2.html)                                               |
| [mindspore.ops.SquaredDifference](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.SquaredDifference.html)                       |
| [mindspore.ops.Xdivy](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.Xdivy.html)                                               |
| [mindspore.ops.Xlogy](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.Xlogy.html)                                               |
| [mindspore.ops.Equal](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.Equal.html)                                               |
| [mindspore.ops.ApproximateEqual](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ApproximateEqual.html)                         |
| [mindspore.ops.NotEqual](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.NotEqual.html)                                         |
| [mindspore.ops.Greater](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.Greater.html)                                           |
| [mindspore.ops.GreaterEqual](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.GreaterEqual.html)                                 |
| [mindspore.ops.Less](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.Less.html)                                                 |
| [mindspore.ops.LessEqual](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.LessEqual.html)                                       |
| [mindspore.ops.LogicalAnd](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.LogicalAnd.html)                                     |
| [mindspore.ops.LogicalOr](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.LogicalOr.html)                                       |
| [mindspore.ops.ScatterNdUpdate](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ScatterNdUpdate.html)                           |
| [mindspore.ops.ScatterNdAdd](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ScatterNdAdd.html)                                 |
| [mindspore.ops.ScatterNdSub](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ScatterNdSub.html)                                 |
| [mindspore.ops.ScatterNonAliasingAdd](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ScatterNonAliasingAdd.html)               |
| [mindspore.ops.ScatterUpdate](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ScatterUpdate.html)                               |
| [mindspore.ops.ScatterMax](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ScatterMax.html)                                     |
| [mindspore.ops.ScatterMin](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ScatterMin.html)                                     |
| [mindspore.ops.ScatterAdd](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ScatterAdd.html)                                     |
| [mindspore.ops.ScatterSub](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ScatterSub.html)                                     |
| [mindspore.ops.ScatterMul](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ScatterMul.html)                                     |
| [mindspore.ops.ScatterDiv](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.ScatterDiv.html)                                     |
| [mindspore.ops.AssignAdd](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/ops/mindspore.ops.AssignAdd.html)                                       |

>
