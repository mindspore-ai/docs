# MindSpore隐式类型转换的算子支持

`Linux` `Ascend` `GPU` `CPU` `模型开发` `初级` `中级` `高级`

<!-- TOC -->

- [MindSpore隐式类型转换的算子支持](#mindspore隐式类型转换的算子支持)
    - [隐式类型转换](#隐式类型转换)
        - [转换规则](#转换规则)
        - [参与转换的数据类型](#参与转换的数据类型)
        - [支持算子](#支持算子)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.0/docs/note/source_zh_cn/operator_list_implicit.md" target="_blank"><img src="./_static/logo_source.png"></a>

## 隐式类型转换
### 转换规则
* 标量与Tensor运算：运算时，将标量自动转为Tensor，数据类型和参与运算的Tensor数据类型保持一致；
而当Tensor是bool数据类型，标量是int或float时，将标量和Tensor都转为数据类型为int32或float32的Tensor。
* 不同数据类型Tensor运算：数据类型优先级排序为bool < uint8 < int8 < int16 < int32 < int64 < float16 < float32 < float64，
运算时，先确定参与运算的Tensor中优先级相对最高的数据类型，然后将低优先级数据类型Tensor转换为相对最高优先级数据类型；
而当int8和uint8数据类型的Tensor进行运算时，将其都转为int16的Tensor。
* 不支持对Parameter进行数据类型转换：如果按照转换规则推导，需要对网络中定义的Parameter进行数据类型转换时，会抛出RuntimeError异常。

### 参与转换的数据类型
* bool
* int8
* uint8
* int16
* int32
* int64
* float16
* float32
* float64
  
### 支持算子

| 算子名
| :-----------
| [mindspore.ops.operations.Assign](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Assign)
| [mindspore.ops.operations.AssignSub](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.AssignSub)
| [mindspore.ops.operations.ApplyMomentum](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyMomentum)
| [mindspore.ops.operations.FusedSparseAdam](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FusedSparseAdam)
| [mindspore.ops.operations.FusedSparseLazyAdam](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FusedSparseLazyAdam)
| [mindspore.ops.operations.FusedSparseFtrl](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FusedSparseFtrl)
| [mindspore.ops.operations.FusedSparseProximalAdagrad](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FusedSparseProximalAdagrad)
| [mindspore.ops.operations.ApplyAdaMax](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyAdaMax)
| [mindspore.ops.operations.ApplyAdadelta](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyAdadelta)
| [mindspore.ops.operations.ApplyAdagrad](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyAdagrad)
| [mindspore.ops.operations.ApplyAdagradV2](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyAdagradV2)
| [mindspore.ops.operations.SparseApplyAdagrad](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyAdagrad)
| [mindspore.ops.operations.SparseApplyAdagradV2](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyAdagradV2)
| [mindspore.ops.operations.ApplyProximalAdagrad](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyProximalAdagrad)
| [mindspore.ops.operations.SparseApplyProximalAdagrad](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyProximalAdagrad)
| [mindspore.ops.operations.ApplyAddSign](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyAddSign)
| [mindspore.ops.operations.ApplyPowerSign](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyPowerSign)
| [mindspore.ops.operations.ApplyGradientDescent](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyGradientDescent)
| [mindspore.ops.operations.ApplyProximalGradientDescent](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApplyProximalGradientDescent)
| [mindspore.ops.operations.SparseApplyFtrl](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyFtrl)
| [mindspore.ops.operations.SparseApplyFtrlV2](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SparseApplyFtrlV2)
| [mindspore.ops.operations.BitwiseAnd](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BitwiseAnd)
| [mindspore.ops.operations.BitwiseOr](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BitwiseOr)
| [mindspore.ops.operations.BitwiseXor](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.BitwiseXor)
| [mindspore.ops.operations.TensorAdd](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.TensorAdd)
| [mindspore.ops.operations.Sub](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Sub)
| [mindspore.ops.operations.Mul](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Mul)
| [mindspore.ops.operations.Pow](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Pow)
| [mindspore.ops.operations.Minimum](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Minimum)
| [mindspore.ops.operations.Maximum](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Maximum)
| [mindspore.ops.operations.RealDiv](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.RealDiv)
| [mindspore.ops.operations.Div](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Div)
| [mindspore.ops.operations.DivNoNan](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.DivNoNan)
| [mindspore.ops.operations.FloorDiv](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FloorDiv)
| [mindspore.ops.operations.TruncateDiv](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.TruncateDiv)
| [mindspore.ops.operations.TruncateMod](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.TruncateMod)
| [mindspore.ops.operations.Mod](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Mod)
| [mindspore.ops.operations.FloorMod](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.FloorMod)
| [mindspore.ops.operations.Atan2](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Atan2)
| [mindspore.ops.operations.SquaredDifference](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.SquaredDifference)
| [mindspore.ops.operations.Xdivy](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Xdivy)
| [mindspore.ops.operations.Xlogy](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Xlogy)
| [mindspore.ops.operations.Equal](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Equal)
| [mindspore.ops.operations.ApproximateEqual](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ApproximateEqual)
| [mindspore.ops.operations.NotEqual](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.NotEqual)
| [mindspore.ops.operations.Greater](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Greater)
| [mindspore.ops.operations.GreaterEqual](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.GreaterEqual)
| [mindspore.ops.operations.Less](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.Less)
| [mindspore.ops.operations.LessEqual](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LessEqual)
| [mindspore.ops.operations.LogicalAnd](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LogicalAnd)
| [mindspore.ops.operations.LogicalOr](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.LogicalOr)
| [mindspore.ops.operations.ScatterNdUpdate](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterNdUpdate)
| [mindspore.ops.operations.ScatterNdAdd](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterNdAdd)
| [mindspore.ops.operations.ScatterNdSub](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterNdSub)
| [mindspore.ops.operations.ScatterNonAliasingAdd](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterNonAliasingAdd)
| [mindspore.ops.operations.ScatterUpdate](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterUpdate)
| [mindspore.ops.operations.ScatterMax](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterMax)
| [mindspore.ops.operations.ScatterMin](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterMin)
| [mindspore.ops.operations.ScatterAdd](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterAdd)
| [mindspore.ops.operations.ScatterSub](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterSub)
| [mindspore.ops.operations.ScatterMul](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterMul)
| [mindspore.ops.operations.ScatterDiv](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.operations.html#mindspore.ops.operations.ScatterDiv)

