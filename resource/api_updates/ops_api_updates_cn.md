# mindspore.ops.primitive API接口变更

与上一版本相比，MindSpore中`mindspore.ops.primitive`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.ops.Cummax](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.Cummax.html#mindspore.ops.Cummax)|Changed|更多详情请查看： mindspore.ops.cummax() 。|r2.3.1: GPU/CPU => r2.4.0: Ascend/GPU/CPU|Array操作
[mindspore.ops.UniqueWithPad](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.UniqueWithPad.html#mindspore.ops.UniqueWithPad)|Deprecated|r2.3.1: 对输入一维张量中元素去重，返回一维张量中的唯一元素（使用pad_num填充）和相对索引。 => r2.4.0: ops.UniqueWithPad 从2.4版本开始已被弃用，并将在未来版本中被移除。|r2.3.1: Ascend/GPU/CPU => r2.4.0: Deprecated|Array操作
[mindspore.ops.ForiLoop](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.ForiLoop.html#mindspore.ops.ForiLoop)|New|一段范围内的循环操作。|r2.4.0: Ascend/GPU/CPU|框架算子
[mindspore.ops.Scan](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.Scan.html#mindspore.ops.Scan)|New|将一个函数循环作用于一个数组，且对当前元素的处理依赖上一个元素的执行结果。|r2.4.0: Ascend/GPU/CPU|框架算子
[mindspore.ops.WhileLoop](https://mindspore.cn/docs/zh-CN/r2.4.0/api_python/ops/mindspore.ops.WhileLoop.html#mindspore.ops.WhileLoop)|New|在编译阶段不进行循环展开的循环算子。|r2.4.0: Ascend/GPU/CPU|框架算子
[mindspore.ops.silent_check.ASDBase](https://mindspore.cn/docs/zh-CN/r2.3.1/api_python/ops/mindspore.ops.silent_check.ASDBase.html#mindspore.ops.silent_check.ASDBase)|Deleted|ASDBase 是 Python 中具有特征值检测特性的算子的基类。|r2.3.1: Ascend|特征值检测
