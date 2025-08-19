# mindspore.ops API接口变更

与上一版本 2.6.0 相比，MindSpore中`mindspore.ops`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别|
|:----|:----|:----|:----|:----|
[mindspore.ops.ring_attention_update](https://mindspore.cn/docs/zh-CN/r2.7.0/api_python/ops/mindspore.ops.ring_attention_update.html#mindspore.ops.ring_attention_update)|New|RingAttentionUpdate算子功能是将两次FlashAttention的输出根据其不同的softmax的max和sum更新。|r2.7.0: Ascend|神经网络