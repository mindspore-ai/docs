# MindSpore隐式类型转换的算子支持

`Linux` `Ascend` `GPU` `CPU` `模型开发` `初级` `中级` `高级`

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/docs/note/source_zh_cn/operator_list_implicit.md)

## 隐式类型转换

### 转换规则

- 标量与Tensor运算：运算时，将标量自动转为Tensor，数据类型和参与运算的Tensor数据类型保持一致；而当Tensor是bool数据类型，标量是int或float时，将标量和Tensor都转为数据类型为int32或float32的Tensor。
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

| 算子名
| :-----------
| [mindspore.ops.Assign](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Assign)
| [mindspore.ops.AssignSub](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.AssignSub)
| [mindspore.ops.ApplyMomentum](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyMomentum)
| [mindspore.ops.FusedSparseAdam](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FusedSparseAdam)
| [mindspore.ops.FusedSparseLazyAdam](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FusedSparseLazyAdam)
| [mindspore.ops.FusedSparseFtrl](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FusedSparseFtrl)
| [mindspore.ops.FusedSparseProximalAdagrad](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FusedSparseProximalAdagrad)
| [mindspore.ops.ApplyAdaMax](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyAdaMax)
| [mindspore.ops.ApplyAdadelta](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyAdadelta)
| [mindspore.ops.ApplyAdagrad](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyAdagrad)
| [mindspore.ops.ApplyAdagradV2](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyAdagradV2)
| [mindspore.ops.SparseApplyAdagrad](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SparseApplyAdagrad)
| [mindspore.ops.SparseApplyAdagradV2](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SparseApplyAdagradV2)
| [mindspore.ops.ApplyProximalAdagrad](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyProximalAdagrad)
| [mindspore.ops.SparseApplyProximalAdagrad](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SparseApplyProximalAdagrad)
| [mindspore.ops.ApplyAddSign](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyAddSign)
| [mindspore.ops.ApplyPowerSign](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyPowerSign)
| [mindspore.ops.ApplyGradientDescent](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyGradientDescent)
| [mindspore.ops.ApplyProximalGradientDescent](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApplyProximalGradientDescent)
| [mindspore.ops.SparseApplyFtrl](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SparseApplyFtrl)
| [mindspore.ops.SparseApplyFtrlV2](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SparseApplyFtrlV2)
| [mindspore.ops.BitwiseAnd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BitwiseAnd)
| [mindspore.ops.BitwiseOr](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BitwiseOr)
| [mindspore.ops.BitwiseXor](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.BitwiseXor)
| [mindspore.ops.TensorAdd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.TensorAdd)
| [mindspore.ops.Sub](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Sub)
| [mindspore.ops.Mul](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Mul)
| [mindspore.ops.Pow](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Pow)
| [mindspore.ops.Minimum](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Minimum)
| [mindspore.ops.Maximum](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Maximum)
| [mindspore.ops.RealDiv](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.RealDiv)
| [mindspore.ops.Div](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Div)
| [mindspore.ops.DivNoNan](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.DivNoNan)
| [mindspore.ops.FloorDiv](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FloorDiv)
| [mindspore.ops.TruncateDiv](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.TruncateDiv)
| [mindspore.ops.TruncateMod](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.TruncateMod)
| [mindspore.ops.Mod](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Mod)
| [mindspore.ops.FloorMod](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.FloorMod)
| [mindspore.ops.Atan2](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Atan2)
| [mindspore.ops.SquaredDifference](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.SquaredDifference)
| [mindspore.ops.Xdivy](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Xdivy)
| [mindspore.ops.Xlogy](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Xlogy)
| [mindspore.ops.Equal](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Equal)
| [mindspore.ops.ApproximateEqual](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ApproximateEqual)
| [mindspore.ops.NotEqual](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.NotEqual)
| [mindspore.ops.Greater](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Greater)
| [mindspore.ops.GreaterEqual](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.GreaterEqual)
| [mindspore.ops.Less](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.Less)
| [mindspore.ops.LessEqual](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LessEqual)
| [mindspore.ops.LogicalAnd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LogicalAnd)
| [mindspore.ops.LogicalOr](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.LogicalOr)
| [mindspore.ops.ScatterNdUpdate](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterNdUpdate)
| [mindspore.ops.ScatterNdAdd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterNdAdd)
| [mindspore.ops.ScatterNdSub](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterNdSub)
| [mindspore.ops.ScatterNonAliasingAdd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterNonAliasingAdd)
| [mindspore.ops.ScatterUpdate](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterUpdate)
| [mindspore.ops.ScatterMax](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterMax)
| [mindspore.ops.ScatterMin](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterMin)
| [mindspore.ops.ScatterAdd](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterAdd)
| [mindspore.ops.ScatterSub](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterSub)
| [mindspore.ops.ScatterMul](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterMul)
| [mindspore.ops.ScatterDiv](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.ops.html#mindspore.ops.ScatterDiv)

>