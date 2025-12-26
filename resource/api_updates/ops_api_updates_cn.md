# mindspore.ops.primitive API接口变更

2.7.1版本与2.7.0版本相比，MindSpore中`mindspore.ops.primitive`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.ops.AlltoAllVC](https://mindspore.cn/docs/zh-CN/r2.7.1/api_python/ops/mindspore.ops.AlltoAllVC.html#mindspore.ops.AlltoAllVC)|New|AllToAllVC通过输入参数 send_count_matrix 传入所有rank的收发参数。|r2.7.1: Ascend|通信算子
