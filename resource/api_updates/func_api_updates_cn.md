# mindspore.ops API接口变更

与上一版本相比，MindSpore中`mindspore.ops`API接口的添加、删除和支持平台的更改信息如下表所示。

|API|变更状态|概述|支持平台|类别|
|:----|:----|:----|:----|:----|
[mindspore.ops.eps](https://mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.eps.html#mindspore.ops.eps)|New|创建一个与输入数据类型和shape都相同的Tensor，元素值为对应数据类型能表达的最小值。|r2.2: Ascend/GPU/CPU|Tensor创建|
[mindspore.ops.clip_by_norm](https://mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.clip_by_norm.html#mindspore.ops.clip_by_norm)|New|对一组输入Tensor进行范数裁剪。|r2.2: Ascend/GPU/CPU|梯度剪裁|
[mindspore.ops.glu](https://mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.glu.html#mindspore.ops.glu)|Changed|门线性单元函数（Gated Linear Unit function）。|r2.1: Ascend/CPU => r2.2: Ascend/GPU/CPU|激活函数|
[mindspore.ops.lrn](https://mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.lrn.html#mindspore.ops.lrn)|Changed|局部响应归一化操作LRN(Local Response Normalization)。|r2.1: Ascend/GPU/CPU => r2.2: GPU/CPU|神经网络|
[mindspore.ops.accumulate_n](https://mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.accumulate_n.html#mindspore.ops.accumulate_n)|Changed|逐元素将所有输入的Tensor相加。|r2.1: Ascend => r2.2: Ascend/GPU|逐元素运算|
