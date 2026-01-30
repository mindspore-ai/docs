# mindspore.ops.primitive API接口变更

2.8.0版本与2.7.2版本相比，MindSpore中`mindspore.ops.primitive`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.ops.DataFormatDimMap](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.DataFormatDimMap.html#mindspore.ops.DataFormatDimMap)|Changed|r2.7.2: 返回源数据格式中的目标数据格式的维度索引。 => r2.8.0: ops.DataFormatDimMap 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend/GPU/CPU => r2.8.0: Deprecated|Array操作
[mindspore.ops.ParallelConcat](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.ParallelConcat.html#mindspore.ops.ParallelConcat)|Changed|r2.7.2: 根据第一个维度连接输入Tensor。 => r2.8.0: ops.ParallelConcat 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend/GPU/CPU => r2.8.0: Deprecated|Array操作
[mindspore.ops.SparseApplyAdagradV2](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.SparseApplyAdagradV2.html#mindspore.ops.SparseApplyAdagradV2)|Changed|r2.7.2: 根据Adagrad算法更新相关参数或者Tensor。 => r2.8.0: ops.SparseApplyAdagradV2 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend/GPU/CPU => r2.8.0: Deprecated|优化器
[mindspore.ops.SparseApplyFtrl](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.SparseApplyFtrl.html#mindspore.ops.SparseApplyFtrl)|Changed|r2.7.2: 根据FTRL-proximal算法更新相关参数或者Tensor。 => r2.8.0: ops.SparseApplyFtrl 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend/GPU/CPU => r2.8.0: Deprecated|优化器
[mindspore.ops.SparseApplyProximalAdagrad](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.SparseApplyProximalAdagrad.html#mindspore.ops.SparseApplyProximalAdagrad)|Changed|r2.7.2: 根据Proximal Adagrad算法更新网络参数或者Tensor。 => r2.8.0: ops.SparseApplyProximalAdagrad 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend/GPU => r2.8.0: Deprecated|优化器
[mindspore.ops.NoRepeatNGram](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.NoRepeatNGram.html#mindspore.ops.NoRepeatNGram)|Changed|r2.7.2: 如果n-grams出现重复，则更新对应n-gram词序列出现的概率。 => r2.8.0: ops.NoRepeatNGram 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend/GPU/CPU => r2.8.0: Deprecated|文本处理
[mindspore.ops.ApproximateEqual](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.ApproximateEqual.html#mindspore.ops.ApproximateEqual)|Changed|r2.7.2: 逐元素计算abs(x-y)，如果小于tolerance则为True，否则为False。 => r2.8.0: ops.ApproximateEqual 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend/GPU/CPU => r2.8.0: Deprecated|比较算子
[mindspore.ops.Padding](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.Padding.html#mindspore.ops.Padding)|Changed|r2.7.2: 将输入Tensor的最后一个维度从1扩展到 pad_dim_size ，其填充值为0。 => r2.8.0: ops.Padding 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend/GPU/CPU => r2.8.0: Deprecated|神经网络
[mindspore.ops.SparseTensorDenseMatmul](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.SparseTensorDenseMatmul.html#mindspore.ops.SparseTensorDenseMatmul)|Changed|r2.7.2: 稀疏矩阵 A 乘以稠密矩阵 B 。 => r2.8.0: ops.SparseTensorDenseMatMul 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: GPU/CPU => r2.8.0: Deprecated|稀疏算子
[mindspore.ops.EditDistance](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.EditDistance.html#mindspore.ops.EditDistance)|Changed|r2.7.2: 计算Levenshtein编辑距离。 => r2.8.0: ops.EditDistance 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend/CPU => r2.8.0: Deprecated|距离函数
[mindspore.ops.AccumulateNV2](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.AccumulateNV2.html#mindspore.ops.AccumulateNV2)|Changed|r2.7.2: 逐元素将所有输入的Tensor相加。 => r2.8.0: ops.AccumulateNV2 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend/GPU => r2.8.0: Deprecated|逐元素运算
[mindspore.ops.ComputeAccidentalHits](https://mindspore.cn/docs/zh-CN/r2.8.0/api_python/ops/mindspore.ops.ComputeAccidentalHits.html#mindspore.ops.ComputeAccidentalHits)|Changed|r2.7.2: 计算与目标类完全匹配的抽样样本的位置id。 => r2.8.0: ops.ComputeAccidentalHits 从2.8.0版本开始已被弃用，并将在未来版本中被移除。|r2.7.2: Ascend => r2.8.0: Deprecated|采样算子

2.7.2版本与2.7.1版本相比，MindSpore中 `mindspore.ops.primitive` API接口没有变化。

2.7.0版本与2.6.0版本相比，MindSpore中`mindspore.ops.primitive`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.ops.AllGatherV](https://mindspore.cn/docs/zh-CN/r2.7.0/api_python/ops/mindspore.ops.AllGatherV.html#mindspore.ops.AllGatherV)|New|从指定的通信组中收集不均匀的张量，并返回全部收集的张量。|r2.7.0: Ascend/GPU|通信算子
[mindspore.ops.ReduceScatterV](https://mindspore.cn/docs/zh-CN/r2.7.0/api_python/ops/mindspore.ops.ReduceScatterV.html#mindspore.ops.ReduceScatterV)|New|规约并且分发指定通信组中不均匀的张量，返回分发后的张量。|r2.7.0: Ascend/GPU|通信算子

2.6.0版本与2.5.0版本相比，MindSpore中`mindspore.ops.primitive`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.ops.Morph](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.Morph.html#mindspore.ops.Morph)|New|Morph 算子用于对用户自定义函数 fn 进行封装，允许其被当做自定义算子使用。|r2.6.0: |框架算子
[mindspore.ops.Svd](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.Svd.html#mindspore.ops.Svd)|Changed|计算一个或多个矩阵的奇异值分解。|r2.5.0: GPU/CPU => r2.6.0: Ascend/GPU/CPU|线性代数算子
[mindspore.ops.CustomOpBuilder](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.CustomOpBuilder.html#mindspore.ops.CustomOpBuilder)|New|CustomOpBuilder 用于初始化和配置MindSpore的自定义算子。|r2.6.0: Ascend/CPU|自定义算子
[mindspore.ops.custom_info_register](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.custom_info_register.html#mindspore.ops.custom_info_register)|Changed|装饰器，用于将注册信息绑定到： mindspore.ops.Custom 的 func 参数。|r2.5.0:  => r2.6.0: Ascend/GPU/CPU|自定义算子
[mindspore.ops.kernel](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.kernel.html#mindspore.ops.kernel)|Changed|用于MindSpore Hybrid DSL函数书写的装饰器。|r2.5.0: Ascend/GPU/CPU => r2.6.0: GPU/CPU|自定义算子
[mindspore.ops.AlltoAllV](https://mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.AlltoAllV.html#mindspore.ops.AlltoAllV)|New|相对AlltoAll来说，AlltoAllV算子支持不等分的切分和聚合。|r2.6.0: Ascend|通信算子