# mindspore.nn API接口变更

2.10.0版本与2.9.0版本相比，MindSpore中`mindspore.nn`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.nn.FTRL](https://mindspore.cn/docs/zh-CN/r2.10.0/api_python/nn/mindspore.nn.FTRL.html#mindspore.nn.FTRL)|Changed|r2.9.0: FTRL算法实现。 => r2.10.0: mindspore.nn.FTRL 从2.9.0版本开始已被弃用，并将在未来版本中被移除。|r2.9.0: Ascend/GPU => r2.10.0: |优化器
[mindspore.nn.LARS](https://mindspore.cn/docs/zh-CN/r2.10.0/api_python/nn/mindspore.nn.LARS.html#mindspore.nn.LARS)|Changed|r2.9.0: LARS算法的实现。 => r2.10.0: mindspore.nn.LARS 从2.9.0版本开始已被弃用，并将在未来版本中被移除。|r2.9.0: Ascend => r2.10.0: |优化器
[mindspore.nn.ProximalAdagrad](https://mindspore.cn/docs/zh-CN/r2.10.0/api_python/nn/mindspore.nn.ProximalAdagrad.html#mindspore.nn.ProximalAdagrad)|Changed|r2.9.0: ProximalAdagrad算法的实现，用于在线学习和随机优化。 => r2.10.0: mindspore.nn.ProximalAdagrad 从2.9.0版本开始已被弃用，并将在未来版本中被移除。|r2.9.0: Ascend/GPU => r2.10.0: |优化器
