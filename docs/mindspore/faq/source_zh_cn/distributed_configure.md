# 分布式配置

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/faq/source_zh_cn/distributed_configure.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

<font size=3>**Q: 进行HCCL分布式训练出错：`Init plugin so failed, ret = 1343225860`？**</font>

A: 初始化HCCL失败了，通常由于`rank json`没写对，可以用`mindspore/model_zoo/utils/hccl_tools`下面的工具生成一个试试。或者导入环境变量`export ASCEND_SLOG_PRINT_TO_STDOUT=1`打开HCCL的日志打印，然后检查日志信息。

<br/>

<font size=3>**Q: MindSpore执行GPU分布式训练报错如下，如何解决:**</font>

```text
Loading libgpu_collective.so failed. Many reasons could cause this:
1.libgpu_collective.so is not installed.
2.nccl is not installed or found.
3.mpi is not installed or found
```

A: 此问题为MindSpore动态加载集合通信库失败，可能原因如下:

- 执行环境未安装分布式训练依赖的OpenMPI以及NCCL。
- NCCL版本未更新至`v2.7.6`: MindSpore `v1.1.0`新增GPU P2P通信算子，该特性依赖于NCCL `v2.7.6`，若环境使用的NCCL未升级为此版本，则会引起加载失败错误。

<br/>

<font size=3>**Q：GPU分布式训练场景下，若错误设置环境变量CUDA_VISIBLE_DEVICES的个数小于执行的进程数时，可能导致进程阻塞问题。**</font>

A：此场景下，部分训练进程会提示如下报错：

```text
[ERROR] DEVICE [mindspore/ccsrc/runtime/device/gpu/cuda_driver.cc:245] SetDevice] SetDevice for id:7 failed, ret[101], invalid device ordinal. Please make sure that the 'device_id' set in context is in the range:[0, total number of GPU). If the environment variable 'CUDA_VISIBLE_DEVICES' is set, the total number of GPU will be the number set in the environment variable 'CUDA_VISIBLE_DEVICES'. For example, if export CUDA_VISIBLE_DEVICES=4,5,6, the 'device_id' can be 0,1,2 at the moment, 'device_id' starts from 0, and 'device_id'=0 means using GPU of number 4.
[ERROR] DEVICE [mindspore/ccsrc/runtime/device/gpu/gpu_device_manager.cc:27] InitDevice] Op Error: Failed to set current device id | Error Number: 0
```

其余进程由于GPU资源已分配成功，会正常执行到初始化`NCCL`步骤，日志如下：

```text
[INFO] DEVICE [mindspore/ccsrc/runtime/hardware/gpu/gpu_device_context.cc:90] Initialize] Start initializing NCCL communicator for device 1
```

此步骤中会调用`NCCL`接口`ncclCommInitRank`，该接口会阻塞，直到所有进程达成一致。因此如果某进程没有调用`ncclCommInitRank`，则会导致进程阻塞。
此问题我们已向`NCCL`社区反馈，社区开发者正在设计解决方案中，目前最新版本还未修复，详见[issue链接](https://github.com/NVIDIA/nccl/issues/593#issuecomment-965939279)。
解决方法：手动`kill`训练进程，根据报错日志，设置正确的卡号后，重启训练任务。

<br/>

<font size=3>**Q：GPU分布式训练场景下，若某进程异常退出，可能导致其余进程阻塞问题。**</font>

A：此场景下，异常进程由于各种问题退出，其余进程由于GPU资源已分配成功，会正常执行到初始化`NCCL`步骤，日志如下：

```text
[INFO] DEVICE [mindspore/ccsrc/runtime/hardware/gpu/gpu_device_context.cc:90] Initialize] Start initializing NCCL communicator for device 1
```

此步骤中会调用`NCCL`接口`ncclCommInitRank`，该接口会阻塞，直到所有进程达成一致。因此如果某进程没有调用`ncclCommInitRank`，则会导致进程阻塞。
此问题我们已向`NCCL`社区反馈，社区开发者正在设计解决方案中，目前最新版本还未修复，详见[issue链接](https://github.com/NVIDIA/nccl/issues/593#issuecomment-965939279)。
解决方法：手动`kill`训练进程后重启训练任务。

<br/>

<font size=3>**Q：在执行单机单卡的脚本时，调用communication.management.init方法可能会报错导致执行失败。**</font>

```text
[CRITICAL] DISTRIBUTED [mindspore/ccsrc/distributed/cluster/cluster_context.cc:130] InitNodeRole] Role name is invalid.
```

A：`communication.management.init`接口只有在执行分布式训练时建议调用，详细作用请参考[Python API文档](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.communication.html#mindspore.communication.init)。在单机单卡模式下，调用此接口会让MindSpore加载分布式相关的配置以及环境变量，导致报错。