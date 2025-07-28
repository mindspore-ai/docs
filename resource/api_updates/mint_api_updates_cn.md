# mindspore.mint API接口变更

与上一版本2.6.0相比，MindSpore中`mindspore.mint`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
|[mindspore.mint.distributed.TCPStore](https://mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/mint/mindspore.mint.distributed.TCPStore.html#mindspore.mint.distributed.TCPStore)|New|一种基于传输控制协议（TCP）的分布式键值存储实现方法。|r2.7.0rc1: Ascend|mindspore.mint.distributed
|[mindspore.mint.floor_divide](https://mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/mint/mindspore.mint.floor_divide.html#mindspore.mint.floor_divide)|New|按元素将第一个输入Tensor除以第二个输入Tensor，并向下取整。|r2.7.0rc1: Ascend|逐元素运算
|[mindspore.mint.nn.functional.threshold](https://mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/mint/mindspore.mint.nn.functional.threshold.html#mindspore.mint.nn.functional.threshold)|New|逐元素计算Threshold激活函数。|r2.7.0rc1: Ascend|非线性激活函数
|[mindspore.mint.nn.functional.threshold_](https://mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/mint/mindspore.mint.nn.functional.threshold_.html#mindspore.mint.nn.functional.threshold_)|New|通过逐元素计算 Threshold 激活函数，原地更新 input Tensor。|r2.7.0rc1: Ascend|非线性激活函数
|[mindspore.mint.nn.Threshold](https://mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/mint/mindspore.mint.nn.Threshold.html#mindspore.mint.nn.Threshold)|New|逐元素计算Threshold激活函数。|r2.7.0rc1: Ascend|非线性激活层 (加权和，非线性)
