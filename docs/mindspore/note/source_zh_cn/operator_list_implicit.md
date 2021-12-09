# MindSpore隐式类型转换的算子支持

<!-- TOC -->

- [MindSpore隐式类型转换的算子支持](#mindspore隐式类型转换的算子支持)
    - [隐式类型转换](#隐式类型转换)
        - [转换规则](#转换规则)
        - [参与转换的数据类型](#参与转换的数据类型)
        - [支持算子](#支持算子)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/note/source_zh_cn/operator_list_implicit.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 隐式类型转换

### 转换规则

- 标量与Tensor运算：运算时，将标量自动转为Tensor，数据类型和参与运算的Tensor数据类型保持一致；当Tensor是bool数据类型，标量是int或float时，将标量和Tensor都转为数据类型为int32或float32的Tensor；当Tensor是int或者uint数据类型，标量是float时，将标量和Tensor都转为数据类型为float32的Tensor。
- 不同数据类型Tensor运算：数据类型优先级排序为bool < uint8 < int8 < int16 < int32 < int64 < float16 < float32 < float64，运算时，先确定参与运算的Tensor中优先级相对最高的数据类型，然后将低优先级数据类型Tensor转换为相对最高优先级数据类型；而当int8和uint8数据类型的Tensor进行运算时，将其都转为int16的Tensor。
- 不支持对Parameter进行数据类型转换：如果按照转换规则推导，需要对网络中定义的Parameter进行数据类型转换时，会抛出RuntimeError异常。

### 参与转换的数据类型

- bool
- int8
- uint8
- int16
- int32
- int64
- float16
- float32
- float64

### 支持算子

<table class="docutils">
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Add.html">mindspore.ops.Add</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ApplyAdadelta.html">mindspore.ops.ApplyAdadelta</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ApplyAdagrad.html">mindspore.ops.ApplyAdagrad</a></td>
</tr>
<tr>
  <td><a href="ttps://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/ops/mindspore.ops.ApplyAdagradV2.html">mindspore.ops.ApplyAdagradV2</a></td>
  <td><a href="ttps://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/ops/mindspore.ops.ApplyAdaMax.html">mindspore.ops.ApplyAdaMax</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ApplyAddSign.html">mindspore.ops.ApplyAddSign</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ApplyGradientDescent.html">mindspore.ops.ApplyGradientDescent</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ApplyMomentum.html">mindspore.ops.ApplyMomentum</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ApplyPowerSign.html">mindspore.ops.ApplyPowerSign</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ApplyProximalAdagrad.html">mindspore.ops.ApplyProximalAdagrad</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ApplyProximalGradientDescent.html">mindspore.ops.ApplyProximalGradientDescent</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ApproximateEqual.html">mindspore.ops.ApproximateEqual</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Assign.html">mindspore.ops.Assign</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.AssignAdd.html">mindspore.ops.AssignAdd</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.AssignSub.html">mindspore.ops.AssignSub</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Atan2.html">mindspore.ops.Atan2</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.BitwiseAnd.html">mindspore.ops.BitwiseAnd</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.BitwiseOr.html">mindspore.ops.BitwiseOr</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.BitwiseXor.html">mindspore.ops.BitwiseXor</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Div.html">mindspore.ops.Div</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.DivNoNan.html">mindspore.ops.DivNoNan</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Equal.html">mindspore.ops.Equal</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.FloorDiv.html">mindspore.ops.FloorDiv</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.FloorMod.html">mindspore.ops.FloorMod</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.FusedSparseAdam.html">mindspore.ops.FusedSparseAdam</td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.FusedSparseFtrl.html">mindspore.ops.FusedSparseFtrl</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.FusedSparseLazyAdam.html">mindspore.ops.FusedSparseLazyAdam</td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.FusedSparseProximalAdagrad.html">mindspore.ops.FusedSparseProximalAdagrad</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Greater.html">mindspore.ops.Greater</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.GreaterEqual.html">mindspore.ops.GreaterEqual</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Less.html">mindspore.ops.Less</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.LessEqual.html">mindspore.ops.LessEqual</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.LogicalAnd.html">mindspore.ops.LogicalAnd</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.LogicalOr.html">mindspore.ops.LogicalOr</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Maximum.html">mindspore.ops.Maximum</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Minimum.html">mindspore.ops.Minimum</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Mod.html">mindspore.ops.Mod</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Mul.html">mindspore.ops.Mul</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.NotEqual.html">mindspore.ops.NotEqual</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Pow.html">mindspore.ops.Pow</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.RealDiv.html">mindspore.ops.RealDiv</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ScatterAdd.html">mindspore.ops.ScatterAdd</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ScatterDiv.html">mindspore.ops.ScatterDiv</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ScatterMax.html">mindspore.ops.ScatterMax</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ScatterMin.html">mindspore.ops.ScatterMin</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ScatterMul.html">mindspore.ops.ScatterMul</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ScatterNdAdd.html">mindspore.ops.ScatterNdAdd</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ScatterNdSub.html">mindspore.ops.ScatterNdSub</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ScatterNdUpdate.html">mindspore.ops.ScatterNdUpdate</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ScatterNonAliasingAdd.html">mindspore.ops.ScatterNonAliasingAdd</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ScatterSub.html">mindspore.ops.ScatterSub</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ScatterUpdate.html">mindspore.ops.ScatterUpdate</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.SparseApplyAdagrad.html">mindspore.ops.SparseApplyAdagrad</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.SparseApplyAdagradV2.html">mindspore.ops.SparseApplyAdagradV2</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.SparseApplyFtrl.html">mindspore.ops.SparseApplyFtrl</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.SparseApplyFtrlV2.html">mindspore.ops.SparseApplyFtrlV2</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.SparseApplyProximalAdagrad.html">mindspore.ops.SparseApplyProximalAdagrad</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.SquaredDifference.html">mindspore.ops.SquaredDifference</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Sub.html">mindspore.ops.Sub</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.TruncateDiv.html">mindspore.ops.TruncateDiv</a></td>
</tr>
<tr>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.TruncateMod.html">mindspore.ops.TruncateMod</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Xdivy.html">mindspore.ops.Xdivy</a></td>
  <td><a href="https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Xlogy.html">mindspore.ops.Xlogy</a></td>
</tr>
</table>
