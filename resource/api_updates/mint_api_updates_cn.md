# mindspore.mint API接口变更

与上一版本 2.6.0 相比，MindSpore中`mindspore.mint`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别|
|:----|:----|:----|:----|:----|
[mindspore.mint.distributed.TCPStore](https://mindspore.cn/docs/zh-CN/r2.7.0/api_python/mint/mindspore.mint.distributed.TCPStore.html#mindspore.mint.distributed.TCPStore)|New|一种基于传输控制协议（TCP）的分布式键值存储实现方法。|r2.7.0: Ascend|mindspore.mint.distributed
[mindspore.mint.distributed.all_gather_into_tensor_uneven](https://mindspore.cn/docs/zh-CN/r2.7.0/api_python/mint/mindspore.mint.distributed.all_gather_into_tensor_uneven.html#mindspore.mint.distributed.all_gather_into_tensor_uneven)|New|收集并拼接各设备上的张量，各设备上的张量第一维可以不一致。|r2.7.0: Ascend|mindspore.mint.distributed
[mindspore.mint.distributed.reduce_scatter_tensor_uneven](https://mindspore.cn/docs/zh-CN/r2.7.0/api_python/mint/mindspore.mint.distributed.reduce_scatter_tensor_uneven.html#mindspore.mint.distributed.reduce_scatter_tensor_uneven)|New|在指定通信组中执行归约分发操作，根据 input_split_sizes 将归约后的张量分散到各rank的输出张量中。|r2.7.0: Ascend|mindspore.mint.distributed
[mindspore.mint.floor_divide](https://mindspore.cn/docs/zh-CN/r2.7.0/api_python/mint/mindspore.mint.floor_divide.html#mindspore.mint.floor_divide)|New|按元素将第一个输入Tensor除以第二个输入Tensor，并向下取整。|r2.7.0: Ascend|逐元素运算
[mindspore.mint.nn.functional.threshold](https://mindspore.cn/docs/zh-CN/r2.7.0/api_python/mint/mindspore.mint.nn.functional.threshold.html#mindspore.mint.nn.functional.threshold)|New|逐元素计算Threshold激活函数。|r2.7.0: Ascend|非线性激活函数
[mindspore.mint.nn.functional.threshold_](https://mindspore.cn/docs/zh-CN/r2.7.0/api_python/mint/mindspore.mint.nn.functional.threshold_.html#mindspore.mint.nn.functional.threshold_)|New|通过逐元素计算 Threshold 激活函数，原地更新 input Tensor。|r2.7.0: Ascend|非线性激活函数
[mindspore.mint.nn.Threshold](https://mindspore.cn/docs/zh-CN/r2.7.0/api_python/mint/mindspore.mint.nn.Threshold.html#mindspore.mint.nn.Threshold)|New|逐元素计算Threshold激活函数。|r2.7.0: Ascend|非线性激活层 (加权和，非线性)