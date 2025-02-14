# mindspore.nn API接口变更

与上一版本2.4.10相比，MindSpore中`mindspore.nn`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.nn.AdamOffload](https://mindspore.cn/docs/zh-CN/r2.4.10/api_python/nn/mindspore.nn.AdamOffload.html#mindspore.nn.AdamOffload)|Deleted|此优化器在主机CPU上运行Adam优化算法，设备上仅执行网络参数的更新，最大限度地降低内存成本。|r2.4.10: Ascend/GPU/CPU|优化器
[mindspore.nn.thor](https://mindspore.cn/docs/zh-CN/r2.4.10/api_python/nn/mindspore.nn.thor.html#mindspore.nn.thor)|Deleted|通过二阶算法THOR更新参数。|r2.4.10: Ascend/GPU|优化器

