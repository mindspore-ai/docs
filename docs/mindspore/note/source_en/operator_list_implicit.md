# MindSpore Implicit Type Conversion Operator List

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/note/source_en/operator_list_implicit.md)

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

<table class="docutils">
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Add.html">mindspore.ops.Add</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ApplyAdadelta.html">mindspore.ops.ApplyAdadelta</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ApplyAdagrad.html">mindspore.ops.ApplyAdagrad</a></td>
</tr>
<tr>
  <td><a href="ttps://www.mindspore.cn/doc/api_python/en/r1.5/mindspore/ops/mindspore.ops.ApplyAdagradV2.html">mindspore.ops.ApplyAdagradV2</a></td>
  <td><a href="ttps://www.mindspore.cn/doc/api_python/en/r1.5/mindspore/ops/mindspore.ops.ApplyAdaMax.html">mindspore.ops.ApplyAdaMax</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ApplyAddSign.html">mindspore.ops.ApplyAddSign</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ApplyGradientDescent.html">mindspore.ops.ApplyGradientDescent</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ApplyMomentum.html">mindspore.ops.ApplyMomentum</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ApplyPowerSign.html">mindspore.ops.ApplyPowerSign</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ApplyProximalAdagrad.html">mindspore.ops.ApplyProximalAdagrad</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ApplyProximalGradientDescent.html">mindspore.ops.ApplyProximalGradientDescent</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ApproximateEqual.html">mindspore.ops.ApproximateEqual</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Assign.html">mindspore.ops.Assign</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.AssignAdd.html">mindspore.ops.AssignAdd</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.AssignSub.html">mindspore.ops.AssignSub</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Atan2.html">mindspore.ops.Atan2</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.BitwiseAnd.html">mindspore.ops.BitwiseAnd</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.BitwiseOr.html">mindspore.ops.BitwiseOr</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.BitwiseXor.html">mindspore.ops.BitwiseXor</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Div.html">mindspore.ops.Div</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.DivNoNan.html">mindspore.ops.DivNoNan</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Equal.html">mindspore.ops.Equal</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.FloorDiv.html">mindspore.ops.FloorDiv</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.FloorMod.html">mindspore.ops.FloorMod</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.FusedSparseAdam.html">mindspore.ops.FusedSparseAdam</td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.FusedSparseFtrl.html">mindspore.ops.FusedSparseFtrl</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.FusedSparseLazyAdam.html">mindspore.ops.FusedSparseLazyAdam</td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.FusedSparseProximalAdagrad.html">mindspore.ops.FusedSparseProximalAdagrad</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Greater.html">mindspore.ops.Greater</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.GreaterEqual.html">mindspore.ops.GreaterEqual</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Less.html">mindspore.ops.Less</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.LessEqual.html">mindspore.ops.LessEqual</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.LogicalAnd.html">mindspore.ops.LogicalAnd</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.LogicalOr.html">mindspore.ops.LogicalOr</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Maximum.html">mindspore.ops.Maximum</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Minimum.html">mindspore.ops.Minimum</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Mod.html">mindspore.ops.Mod</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Mul.html">mindspore.ops.Mul</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.NotEqual.html">mindspore.ops.NotEqual</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Pow.html">mindspore.ops.Pow</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.RealDiv.html">mindspore.ops.RealDiv</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ScatterAdd.html">mindspore.ops.ScatterAdd</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ScatterDiv.html">mindspore.ops.ScatterDiv</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ScatterMax.html">mindspore.ops.ScatterMax</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ScatterMin.html">mindspore.ops.ScatterMin</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ScatterMul.html">mindspore.ops.ScatterMul</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ScatterNdAdd.html">mindspore.ops.ScatterNdAdd</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ScatterNdSub.html">mindspore.ops.ScatterNdSub</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ScatterNdUpdate.html">mindspore.ops.ScatterNdUpdate</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ScatterNonAliasingAdd.html">mindspore.ops.ScatterNonAliasingAdd</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ScatterSub.html">mindspore.ops.ScatterSub</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.ScatterUpdate.html">mindspore.ops.ScatterUpdate</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.SparseApplyAdagrad.html">mindspore.ops.SparseApplyAdagrad</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.SparseApplyAdagradV2.html">mindspore.ops.SparseApplyAdagradV2</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.SparseApplyFtrl.html">mindspore.ops.SparseApplyFtrl</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.SparseApplyFtrlV2.html">mindspore.ops.SparseApplyFtrlV2</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.SparseApplyProximalAdagrad.html">mindspore.ops.SparseApplyProximalAdagrad</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.SquaredDifference.html">mindspore.ops.SquaredDifference</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Sub.html">mindspore.ops.Sub</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.TruncateDiv.html">mindspore.ops.TruncateDiv</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.TruncateMod.html">mindspore.ops.TruncateMod</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Xdivy.html">mindspore.ops.Xdivy</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Xlogy.html">mindspore.ops.Xlogy</a></td>
</tr>
</table>
