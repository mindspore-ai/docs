# mindspore.ops.primitive API接口变更

与上一版本相比，MindSpore中`mindspore.ops.primitive`API接口的添加、删除和支持平台的更改信息如下表所示。

## 2.3.0版本与2.3.0rc2版本差异

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
|[mindspore.ops.Custom](https://mindspore.cn/docs/zh-CN/r2.3.0/api_python/ops/mindspore.ops.Custom.html#mindspore.ops.Custom)|Changed|Custom 算子是MindSpore自定义算子的统一接口。|r2.3.0rc2: GPU/CPU => r2.3.0: GPU/CPU/ASCEND|自定义算子
|[mindspore.ops.Barrier](https://mindspore.cn/docs/zh-CN/r2.3.0/api_python/ops/mindspore.ops.Barrier.html#mindspore.ops.Barrier)|New|同步通信域内的多个进程。|r2.3.0: Ascend|通信算子
|[mindspore.ops.CollectiveGather](https://mindspore.cn/docs/zh-CN/r2.3.0/api_python/ops/mindspore.ops.CollectiveGather.html#mindspore.ops.CollectiveGather)|New|对通信组的输入张量进行聚合。|r2.3.0: Ascend|通信算子
|[mindspore.ops.CollectiveScatter](https://mindspore.cn/docs/zh-CN/r2.3.0/api_python/ops/mindspore.ops.CollectiveScatter.html#mindspore.ops.CollectiveScatter)|New|对输入数据的数据进行均匀散射到通信域的卡上。|r2.3.0: Ascend|通信算子
|[mindspore.ops.Receive](https://mindspore.cn/docs/zh-CN/r2.3.0/api_python/ops/mindspore.ops.Receive.html#mindspore.ops.Receive)|New|接收来自 src_rank 线程的张量。|r2.3.0: Ascend/GPU|通信算子
|[mindspore.ops.Reduce](https://mindspore.cn/docs/zh-CN/r2.3.0/api_python/ops/mindspore.ops.Reduce.html#mindspore.ops.Reduce)|New|规约指定通信组中的张量，并将规约结果发送到目标为dest_rank的进程中，返回发送到目标进程的张量。|r2.3.0: Ascend|通信算子
|[mindspore.ops.Send](https://mindspore.cn/docs/zh-CN/r2.3.0/api_python/ops/mindspore.ops.Send.html#mindspore.ops.Send)|New|发送张量到指定线程。|r2.3.0: Ascend/GPU|通信算子

## 2.3.0rc2版本与2.3.0rc1版本差异

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
|[mindspore.ops.Zeros](https://mindspore.cn/docs/zh-CN/r2.3.0rc2/api_python/ops/mindspore.ops.Zeros.html#mindspore.ops.Zeros)|Changed|创建一个值全为0的Tensor。|r2.3.0rc1:  => r2.3.0rc2: Ascend/GPU/CPU|Tensor创建
|[mindspore.ops.MaxPoolWithArgmax](https://mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/ops/mindspore.ops.MaxPoolWithArgmax.html#mindspore.ops.MaxPoolWithArgmax)|Deleted|mindspore.ops.MaxPoolWithArgmax 从2.0版本开始已被弃用，并将在未来版本中被移除，建议使用 mindspore.ops.MaxPoolWithArgmaxV2 代替。|r2.3.0rc1: |神经网络
|[mindspore.ops.MaxPoolWithArgmaxV2](https://mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/ops/mindspore.ops.MaxPoolWithArgmaxV2.html#mindspore.ops.MaxPoolWithArgmaxV2)|Deleted|对输入Tensor执行最大池化运算，并返回最大值和索引。|r2.3.0rc1: Ascend/GPU/CPU|神经网络
|[mindspore.ops.Custom](https://mindspore.cn/docs/zh-CN/r2.3.0rc2/api_python/ops/mindspore.ops.Custom.html#mindspore.ops.Custom)|Changed|Custom 算子是MindSpore自定义算子的统一接口。|r2.3.0rc1: Ascend/GPU/CPU => r2.3.0rc2: GPU/CPU|自定义算子

## 2.3.0rc1版本与2.2.14版本差异

|API|变更状态|概述|支持平台|类别
|:----|:----|:----|:----|:----
|[mindspore.ops.BatchToSpaceND](https://mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/ops/mindspore.ops.BatchToSpaceND.html#mindspore.ops.BatchToSpaceND)|Changed|mindspore.ops.BatchToSpaceND 从2.0版本开始已被弃用，并将在未来版本中被移除，建议使用 mindspore.ops.batch_to_space_nd() 代替。|r2.2: Ascend/GPU/CPU => r2.3.0rc1: |Array操作
|[mindspore.ops.Identity](https://mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/ops/mindspore.ops.Identity.html#mindspore.ops.Identity)|Changed|r2.2: mindspore.ops.Identity 接口将废弃，请使用 mindspore.ops.deepcopy() 替代。 => r2.3.0rc1: 更多详情请查看： mindspore.ops.deepcopy() 。|r2.2:  => r2.3.0rc1: Ascend/GPU/CPU|Array操作
|[mindspore.ops.InplaceUpdate](https://mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/ops/mindspore.ops.InplaceUpdate.html#mindspore.ops.InplaceUpdate)|Changed|InplaceUpdate接口已废弃，请使用接口 mindspore.ops.InplaceUpdateV2 替换，废弃版本2.0。|r2.2: Ascend/GPU/CPU => r2.3.0rc1: |Array操作
|[mindspore.ops.ScatterNonAliasingAdd](https://mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/ops/mindspore.ops.ScatterNonAliasingAdd.html#mindspore.ops.ScatterNonAliasingAdd)|Changed|ScatterNonAliasingAdd接口已废弃，请使用接口 mindspore.ops.TensorScatterAdd 替换，废弃版本2.1。|r2.2: Ascend/GPU/CPU => r2.3.0rc1: |Parameter操作算子
|[mindspore.ops.ExtractVolumePatches](https://mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/ops/mindspore.ops.ExtractVolumePatches.html#mindspore.ops.ExtractVolumePatches)|Changed|r2.2: 从输入中提取数据，并将它放入"depth"输出维度中，"depth"为输出的第二维。 => r2.3.0rc1: ops.ExtractVolumePatches 从2.3版本开始已被弃用，并将在未来版本中被移除。|r2.2: Ascend/GPU/CPU => r2.3.0rc1: |图像处理
|[mindspore.ops.MaxPoolWithArgmax](https://mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/ops/mindspore.ops.MaxPoolWithArgmax.html#mindspore.ops.MaxPoolWithArgmax)|Changed|mindspore.ops.MaxPoolWithArgmax 从2.0版本开始已被弃用，并将在未来版本中被移除，建议使用 mindspore.ops.MaxPoolWithArgmaxV2 代替。|r2.2: Ascend/GPU/CPU => r2.3.0rc1: |神经网络
|[mindspore.ops.ResizeBilinear](https://mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.ResizeBilinear.html#mindspore.ops.ResizeBilinear)|Deleted|此接口已弃用，请使用 mindspore.ops.ResizeBilinearV2 。||神经网络
|[mindspore.ops.ScalarCast](https://mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/ops/mindspore.ops.ScalarCast.html#mindspore.ops.ScalarCast)|Changed|r2.2: 将输入Scalar转换为其他类型。 => r2.3.0rc1: ops.ScalarCast 从2.3版本开始已被弃用，并将在未来版本中被移除，建议使用 int(x) 或 float(x) 代替。|r2.2: Ascend/GPU/CPU => r2.3.0rc1: |类型转换
|[mindspore.ops.silent_check.ASDBase](https://mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/ops/mindspore.ops.silent_check.ASDBase.html#mindspore.ops.silent_check.ASDBase)|Changed|ASDBase 是 Python 中具有精度敏感检测特性的算子的基类。|r2.2: Ascend/GPU/CPU => r2.3.0rc1: Ascend|精度敏感检测
|[mindspore.ops.TensorDump](https://mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/ops/mindspore.ops.TensorDump.html#mindspore.ops.TensorDump)|New|将Tensor保存为numpy格式的npy文件。|r2.2: r2.3.0rc1: Ascend|调试算子
|[mindspore.ops.RandpermV2](https://mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/ops/mindspore.ops.RandpermV2.html#mindspore.ops.RandpermV2)|Changed|r2.2: 生成从0到n-1不重复的n个随机整数。 => r2.3.0rc1: 更多详情请查看： mindspore.ops.randperm() 。|r2.2: Ascend/CPU => r2.3.0rc1: CPU|随机生成算子
