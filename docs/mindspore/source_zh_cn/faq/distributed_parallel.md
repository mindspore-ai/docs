# 分布式并行

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/faq/distributed_parallel.md)

## Q: 进行HCCL分布式训练出错：`Init plugin so failed, ret = 1343225860`？

A: 在Ascend进行分布式训练时初始化HCCL失败了，通常由于`rank_table.json`没写对，可以执行此文件[hccl_tools.py](https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/hccl_tools.py)生成一个新的`rank_table.json`。或者导入环境变量`export ASCEND_SLOG_PRINT_TO_STDOUT=1`打开HCCL的日志打印，根据日志中的ERROR信息来排查问题。

<br/>

## Q：GPU分布式训练场景下，若错误设置环境变量CUDA_VISIBLE_DEVICES的个数小于执行的进程数时，可能导致进程阻塞问题。

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

## Q：GPU分布式训练场景下，若某进程异常退出，可能导致其余进程阻塞问题。

A：此场景下，异常进程由于各种问题退出，其余进程由于GPU资源已分配成功，会正常执行到初始化`NCCL`步骤，日志如下：

```text
[INFO] DEVICE [mindspore/ccsrc/runtime/hardware/gpu/gpu_device_context.cc:90] Initialize] Start initializing NCCL communicator for device 1
```

此步骤中会调用`NCCL`接口`ncclCommInitRank`，该接口会阻塞，直到所有进程达成一致。因此如果某进程没有调用`ncclCommInitRank`，则会导致进程阻塞。
此问题我们已向`NCCL`社区反馈，社区开发者正在设计解决方案中，目前最新版本还未修复，详见[issue链接](https://github.com/NVIDIA/nccl/issues/593#issuecomment-965939279)。
解决方法：手动`kill`训练进程后重启训练任务。

<br/>

## Q：在执行GPU单机单卡的脚本时，不使用mpirun启动进程时，调用mindspore.communication.init方法可能会报错，导致执行失败，该如何处理？

```text
[CRITICAL] DISTRIBUTED [mindspore/ccsrc/distributed/cluster/cluster_context.cc:130] InitNodeRole] Role name is invalid...
```

A：在用户不使用`mpirun`启动进程，但是依然调用了`init()`方法的情况下，MindSpore要求用户按照[动态组网启动方式](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3/parallel/dynamic_cluster.html)配置若干环境变量并进行校验，若没有配置，MindSpore会给出以上报错提示。因此建议只有在执行分布式训练时调用`mindspore.communication.init`，并在不使用`mpirun`的场景下，根据文档配置正确的环境变量以启动分布式训练。

<br/>

## Q：在通过OpenMPI执行多机多卡训练时，提示由于MPI_Allgather失败。

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

## Q：在通过OpenMPI执行分布式训练时，单机多卡训练正常，但在多机多卡训练时，某些机器提示GPU device id设置失败。

```text
[ERROR] DEVICE [mindspore/ccsrc/runtime/device/gpu/cuda_driver.cc:245] SetDevice] SetDevice for id:7 failed, ret[101], invalid device ordinal. Please make sure that the 'device_id' set in context is in the range:[0, total number of GPU). If the environment variable 'CUDA_VISIBLE_DEVICES' is set, the total number of GPU will be the number set in the environment variable 'CUDA_VISIBLE_DEVICES'. For example, if export CUDA_VISIBLE_DEVICES=4,5,6, the 'device_id' can be 0,1,2 at the moment, 'device_id' starts from 0, and 'device_id'=0 means using GPU of number 4.
[ERROR] DEVICE [mindspore/ccsrc/runtime/device/gpu/gpu_device_manager.cc:27] InitDevice] Op Error: Failed to set current device id | Error Number: 0
```

A：在多机场景下，各进程卡号需要通过在Host侧`AllGather` `HOSTNAME`后计算得到，如果机器间有使用相同的`HOSTNAME`，则进程卡号会计算出错，导致卡号越界而设置失败。可以在执行脚本中设置每台机器的HOSTNAME为各自的IP地址来解决：

```text
export HOSTNAME=node_ip_address
```

<br/>

## Q：在通过OpenMPI执行多机多卡训练时，NCCL报错提示网络不通。

```text
include/socket.h:403 NCCL WARN Connect to XXX failed: Network is unreachable
```

A：此问题是`NCCL`在Host侧同步进程信息或者初始化通信域时，无法和对端地址进行通信，一般是机器之间的网卡配置不同导致的，可以通过设置`NCCL`环境变量`NCCL_SOCKET_IFNAME`，进行网卡选择：

```text
export NCCL_SOCKET_IFNAME=eth
```

以上指令设置了`NCCL`在Host侧选择网卡名中带有`eth`的网卡进行通信。

<br/>

## Q：多机多卡选择特定名称的RDMA网卡(通过NCCL_SOCKET_IFNAME设置)通信后，训练仍然报错。

```text
misc/ibvwrap.cc:284 NCCL WARN Call to ibv_modify_qp failed with error Invalid argument
...
include/socket.h:403 NCCL WARN Connect to XXX failed: Connection refused
```

A：一般此问题是多机之间RDMA网卡配置存在差异，需要具体情况具体分析。但常见原因是存在某些主机网卡存在IB协议和RoCE协议同时存在的情况，可能出现连接建立失败的情况。解决方案：

需要使用以下指令指定使用的RDMA网卡名为ib开头：

```text
export NCCL_IB_HCA=mlx
```

<br/>

## Q：单机多卡训练能够成功，但是扩展脚本到多机多卡后，其他主机提示各类报错。

报错内容有多种，下面是几种典型的报错，可能有：

1. 已经安装的whl包找不到。
2. IB网卡通信失败。
3. Cuda库加载失败。

A：这些问题，都是由于在`mpirun`启动其他主机时，其他主机的环境变量(包括NCCL的网卡选择配置)没有与本机同步，导致了单机多卡正常执行而多机多卡失败的现象。解决方法是通过mpirun的-x选项，导出特定的环境变量：

```text
mpirun --hostfile /path/to/hostfile -n 64 -x PYTHONPATH -x GLOG_v -x LD_LIBRARY_PATH -x NCCL_SOCKET_IFNAME -x NCCL_IB_HCA -x NCCL_DEBUG=INFO python train.py
```

以上指令导出了在本机已经设置的一些环境变量到其他主机，保证了在执行训练脚本前所有主机环境变量保持一致，达到多机多卡训练目标。

<br/>

## Q: 在Ascend上通过OpenMPI执行分布式训练时，`HcclCommInitRootInfo`报错。

```text
Ascend collective Error: "HcclCommInitRootInfo failed. | Error Number 2
```

A: OpenMPI启动时，当前版本的hccl下，创建通信域时，相应的卡需要分配大约300M的device内存，因此每张卡所在的通信域的数量越多，则额外需要的内存越多，因此会有内存不足的问题。
可以设置`context`中的`variable_memory_max_size`来减少Ascend进程可用的内存，从而为hccl预留足够的内存创建通信域。

<br/>

## Q: 在自动并行下执行分布式网络时，报张量无法被当前策略完整切分的错误如下，该怎么解决？

```text
np_tensor can not be split by strategy!
```

A: 该报错表明网络中有对参数配置了切分策略，但是参数的某个维度无法被切分策略整除。可能的问题有两个：1、该参数作为某个算子的输入，脚本中调用了shard接口对该算子设置了非法策略；2、在`auto_parallel_context`中设置了`dataset_strategy`="data_parallel"或`full_batch`=False时，框架会自动为网络输入设置数据并行策略，如果网络输入含有参数且其形状恰好不能被数据并行策略整除，也会报该错误。目前自动并行下仅支持网络输入为Tensor，需要对脚本进行调整。

<br/>

## Q: Linux环境上执行多卡训练过程中进程异常退出，通过ipcs命令看到有共享内存残留，该如何处理？

A: 在多卡训练并使能图算融合情况下，框架采用共享内存机制进行多卡间的算子统一编译，如果编译过程中遇到内外部异常导致进程意外结束，共享内存得不到有效释放。通过ipcs命令可以看到残留的共享内存的nattch为0，重新执行训练脚本时框架会重新接管共享内存，只要无异常出现就可以正常释放。也可以通过ipcrm命令释放共享内存，不会影响训练脚本执行。

<br/>

## Q: Ascend平台下训练较小规模网络，但在分布式模块初始化过程中，依然提示设备侧内存不足，如何解决？

A: 这是因为在Ascend平台下，MindSpore后端默认会预分配一块内存，约80%的NPU内存会被占用，剩余的20%内存则用于HCCL集合通信库的初始化。每个HCCL通信组会默认占用200MB的内存，那么在通信组较多的场景下，就容易出现设备侧内存不足的报错。解决方法是设置`HCCL_BUFFSIZE`环境变量修改通信域内存占用，具体配置方式可参考[HCCL官方文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha001/apiref/envref/envref_07_0081.html)。
