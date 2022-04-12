# 分布式配置

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/faq/distributed_configure.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

<font size=3>**Q：在执行GPU单机单卡的脚本时，不使用mpirun启动进程时，调用mindspore.communication.init方法可能会报错,导致执行失败，该如何处理？**</font>

```text
[CRITICAL] DISTRIBUTED [mindspore/ccsrc/distributed/cluster/cluster_context.cc:130] InitNodeRole] Role name is invalid...
```

A：在用户不使用`mpirun`启动进程，但是依然调用了`init()`方法的情况下，MindSpore要求用户按照[不依赖OpenMPI进行训练](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_gpu.html#openmpi)配置若干环境变量并进行校验，若没有配置，MindSpore会给出以上报错提示。因此建议只有在执行分布式训练时调用`mindspore.communication.init`，并在不使用`mpirun`的场景下，根据文档配置正确的环境变量以启动分布式训练。

<br/>

<font size=3>**Q：在通过OpenMPI执行多机多卡训练时，提示由于MPI_Allgather失败。**</font>

```text
pml_ucx.c:175 Error: Failed to receive UCX worker address: Not found (-13)
pml_ucx.c:452 Error: Failed to resolve UCX endpoint for rank X
```

A：此问题是`OpenMPI`在Host侧通信时，无法和对端地址进行通信，一般是机器之间的网卡配置不同导致的，可以通过手动设置网卡名或者子网的方式解决：

```text
mpirun -n process_num --mca btl tcp --mca btl_tcp_if_include eth0 ./run.sh
```

以上指令启动了`process_num`个`run.sh`进程，并且选择Host侧通信方式为`tcp`，网卡选择了`eth0`，这样就能保证在每台机器上使用的网卡相同，进而解决通信异常问题。

还可以选择子网来进行匹配：

```text
mpirun -n process_num --mca btl tcp --mca btl_tcp_if_include 192.168.1.0/24 ./run.sh
```

子网范围需要包括所有机器所用的IP地址。

<br/>

<font size=3>**Q：在通过OpenMPI执行分布式训练时，单机多卡训练正常，但在多机多卡训练时，某些机器提示GPU device id设置失败。**</font>

```text
[ERROR] DEVICE [mindspore/ccsrc/runtime/device/gpu/cuda_driver.cc:245] SetDevice] SetDevice for id:7 failed, ret[101], invalid device ordinal. Please make sure that the 'device_id' set in context is in the range:[0, total number of GPU). If the environment variable 'CUDA_VISIBLE_DEVICES' is set, the total number of GPU will be the number set in the environment variable 'CUDA_VISIBLE_DEVICES'. For example, if export CUDA_VISIBLE_DEVICES=4,5,6, the 'device_id' can be 0,1,2 at the moment, 'device_id' starts from 0, and 'device_id'=0 means using GPU of number 4.
[ERROR] DEVICE [mindspore/ccsrc/runtime/device/gpu/gpu_device_manager.cc:27] InitDevice] Op Error: Failed to set current device id | Error Number: 0
```

A：在多机场景下，各进程卡号需要通过在Host侧`AllGather` `HOSTNAME`后计算得到，如果机器间有使用相同的`HOSTNAME`，则进程卡号会计算出错，导致卡号越界而设置失败。可以在执行脚本中设置每台机器的HOSTNAME为各自的IP地址来解决：

```text
export HOSTNAME=node_ip_address
```

<br/>

<font size=3>**Q：在通过OpenMPI执行多机多卡训练时，NCCL报错提示网络不通。**</font>

```text
include/socket.h:403 NCCL WARN Connect to XXX failed: Network is unreachable
```

A：此问题是`NCCL`在Host侧同步进程信息或者初始化通信域时，无法和对端地址进行通信，一般是机器之间的网卡配置不同导致的，可以通过设置`NCCL`环境变量`NCCL_SOCKET_IFNAME`，进行网卡选择：

```text
export NCCL_SOCKET_IFNAME=eth
```

以上指令设置了`NCCL`在Host侧选择网卡名中带有`eth`的网卡进行通信。

<br/>

<font size=3>**Q：多机多卡选择特定名称的RDMA网卡(通过NCCL_SOCKET_IFNAME设置)通信后，训练仍然报错：**</font>

```text
misc/ibvwrap.cc:284 NCCL WARN Call to ibv_modify_qp failed with error Invalid argument
...
include/socket.h:403 NCCL WARN Connect to XXX failed: Connection refused
```

A：一般此问题是多机之间RDMA网卡配置存在差异，需要具体情况具体分析。但常见原因是存在某些主机网卡存在IB协议和RoCE协议同时存在的情况，可能出现连接建立失败的情况。解决方案：

需要使用以下指令指定使用的RDMA网卡名为ib开头：

```text
export NCCL_IB_HCA=ib
```

<br/>

<font size=3>**Q：单机多卡训练能够成功，但是扩展脚本到多机多卡后，其他主机提示各类报错：**</font>

报错内容有多种，下面是几种典型的报错，可能有：
1.已经安装的whl包找不到。
2.IB网卡通信失败。
3.Cuda库加载失败。

A：这些问题，都是由于在`mpirun`启动其他主机时，其他主机的环境变量(包括NCCL的网卡选择配置)没有与本机同步，导致了单机多卡正常执行而多机多卡失败的现象。解决方法是通过mpirun的-x选项，导出特定的环境变量：

```text
mpirun --hostfile /path/to/hostfile -n 64 -x PYTHONPATH -x GLOG_v -x LD_LIBRARY_PATH -x NCCL_SOCKET_IFNAME -x NCCL_IB_HCA -x NCCL_DEBUG=INFO python train.py
```

以上指令导出了在本机已经设置的一些环境变量到其他主机，保证了在执行训练脚本前所有主机环境变量保持一致，达到多机多卡训练目标。