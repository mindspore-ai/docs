# mindspore.mint API接口变更

2.7.1版本与2.7.0版本相比，MindSpore中`mindspore.mint`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
[mindspore.mint.nn.functional.conv1d](https://mindspore.cn/docs/zh-CN/r2.7.1/api_python/mint/mindspore.mint.nn.functional.conv1d.html#mindspore.mint.nn.functional.conv1d)|New|对输入Tensor计算三维卷积。|r2.7.1: Ascend|卷积函数
[mindspore.mint.nn.Conv1d](https://mindspore.cn/docs/zh-CN/r2.7.1/api_python/mint/mindspore.mint.nn.Conv1d.html#mindspore.mint.nn.Conv1d)|New|一维卷积层。|r2.7.1: Ascend|卷积层
[mindspore.mint.imag](https://mindspore.cn/docs/zh-CN/r2.7.1/api_python/mint/mindspore.mint.imag.html#mindspore.mint.imag)|New|返回一个新tensor，包含输入tensor的虚部。|r2.7.1: Ascend|逐元素运算
[mindspore.mint.real](https://mindspore.cn/docs/zh-CN/r2.7.1/api_python/mint/mindspore.mint.real.html#mindspore.mint.real)|New|返回一个新tensor，包含输入tensor的实部。|r2.7.1: Ascend|逐元素运算
