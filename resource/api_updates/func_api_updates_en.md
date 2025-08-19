# mindspore.ops API Interface Change

Compared with the previous version 2.6.0, the added, deleted and supported platforms change information of `mindspore.ops` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.ops.ring_attention_update](https://mindspore.cn/docs/en/r2.7.0/api_python/ops/mindspore.ops.ring_attention_update.html#mindspore.ops.ring_attention_update)|New|The RingAttentionUpdate operator updates the output of two FlashAttention operations based on their respective softmax max and softmax sum values.|r2.7.0: Ascend|Neural Network