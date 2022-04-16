# Distributed Configuration

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/faq/distributed_configure.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

<font size=3>**Q: What do I do if the error message `Init plugin so failed, ret = 1343225860` is displayed during the HCCL distributed training?**</font>

A: HCCL fails to be initialized. The possible cause is that `rank json` is incorrect. You can use the tool in `mindspore/model_zoo/utils/hccl_tools` to generate one. Alternatively, import the environment variable `export ASCEND_SLOG_PRINT_TO_STDOUT=1` to enable the log printing function of HCCL and check the log information.

<br/>

<font size=3>**Q: How to fix the error below when running MindSpore distributed training with GPU:**</font>

```text
Loading libgpu_collective.so failed. Many reasons could cause this:
1.libgpu_collective.so is not installed.
2.nccl is not installed or found.
3.mpi is not installed or found
```

A: This message means that MindSpore is failed to dynamically load the collection communication library. The Possible causes are:

- OpenMPI or NCCL relied by the diatributed training is not installed in this environment.
- NCCL version is not updated to `v2.7.6`: MindSpore `v1.1.0` adds GPU P2P communication operator which relies on NCCL `v2.7.6`. The loading failure is caused if NCCL is not updated to this version.

<br/>

<font size=3>**Q: In the GPU distributed training scenario, if the number of environment variables CUDA_VISIBLE_DEVICES set incorrectly is less than the number of processes executed, the process blocking problem may occur.**</font>

A: In this scenario, some training processes will prompt the following error:

```text
[ERROR] DEVICE [mindspore/ccsrc/runtime/device/gpu/cuda_driver.cc:245] SetDevice] SetDevice for id:7 failed, ret[101], invalid device ordinal. Please make sure that the 'device_id' set in context is in the range:[0, total number of GPU). If the environment variable 'CUDA_VISIBLE_DEVICES' is set, the total number of GPU will be the number set in the environment variable 'CUDA_VISIBLE_DEVICES'. For example, if export CUDA_VISIBLE_DEVICES=4,5,6, the 'device_id' can be 0,1,2 at the moment, 'device_id' starts from 0, and 'device_id'=0 means using GPU of number 4.
[ERROR] DEVICE [mindspore/ccsrc/runtime/device/gpu/gpu_device_manager.cc:27] InitDevice] Op Error: Failed to set current device id | Error Number: 0
```

The remaining processes may normally execute to the initialization `NCCL` step due to the successful allocation of GPU resources, and the log is as follows:

```text
[INFO] DEVICE [mindspore/ccsrc/runtime/hardware/gpu/gpu_device_context.cc:90] Initialize] Start initializing NCCL communicator for device 1
```

In this step, the `NCCL` interface `ncclCommInitRank` is called, which blocks until all processes agree. So if a process doesn't call `ncclCommInitRank`, it will cause the process to block.

We have reported this issue to the `NCCL` community, and the community developers are designing a solution. The latest version has not been fixed, see [issue link](https://github.com/NVIDIA/nccl/issues/593#issuecomment-965939279).

Solution: Manually `kill` the training process. According to the error log, set the correct card number, and then restart the training task.

<br/>

<font size=3>**Q: What can we do when in the GPU distributed training scenario, if a process exits abnormally, it may cause other processes to block?**</font>

A: In this scenario, the abnormal process exits due to various problems, and the remaining processes are executed normally to the initialization `NCCL` step due to the successful allocation of GPU resources. The log is as follows:

```text
[INFO] DEVICE [mindspore/ccsrc/runtime/hardware/gpu/gpu_device_context.cc:90] Initialize] Start initializing NCCL communicator for device 1
```

In this step, the `NCCL` interface `ncclCommInitRank` is called, which blocks until all processes agree. So if a process doesn't call `ncclCommInitRank`, it will cause the process to block.

We have reported this issue to the `NCCL` community, and the community developers are designing a solution. The latest version has not been fixed, see [issue link](https://github.com/NVIDIA/nccl/issues/593#issuecomment-965939279).

Solution: Manually `kill` the training process. According to the error log, set the correct card number, and then restart the training task.

<br/>

<font size=3>**Q: When executing a GPU stand-alone single-card script, when the process is started without mpirun, calling the mindspore.communication.init method may report an error, resulting in execution failure, how to deal with it?**</font>

```text
[CRITICAL] DISTRIBUTED [mindspore/ccsrc/distributed/cluster/cluster_context.cc:130] InitNodeRole] Role name is invalid...
```

A: In the case where the user does not start the process using `mpirun` but still calls the `init()` method, MindSpore requires the user to configure several environment variables and verify according to [training and do not rely on OpenMPI for training]( https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/train_gpu.html#不依赖openmpi进行训练). If without configuring, MindSpore may display the above error message. Therefore, it is suggested that only when performing distributed training, `mindspore.communication.init` is called, and in the case of not using `mpirun`, it is configured the correct environment variables according to the documentation to start distributed training.

<br/>

<font size=3>**Q: What can we do when performing multi-machine multi-card training via OpenMPI, the prompt fails due to MPI_Allgather?**</font>

```text
pml_ucx.c:175 Error: Failed to receive UCX worker address: Not found (-13)
pml_ucx.c:452 Error: Failed to resolve UCX endpoint for rank X
```

A: This problem is that `OpenMPI` cannot communicate with the peer address when communicating on the Host side, which is generally caused by the different configuration of the NIC between the machines, and can be solved by manually setting the NIC name or subnet:

```text
mpirun -n process_num --mca btl tcp --mca btl_tcp_if_include eth0 ./run.sh
```

The above instruction starts the `process_num` of `run.sh` processes, and selects the Host side communication mode as `tcp`. The network card selects `eth0`, so that the network card used on each machine is the same, and then the communication abnormal problem is solved.

You can also select subnets for matching:

```text
mpirun -n process_num --mca btl tcp --mca btl_tcp_if_include 192.168.1.0/24 ./run.sh
```

The subnet range needs to include the IP addresses used by all machines.

<br/>

<font size=3>**Q: What can we do when performing distributed training via OpenMPI, stand-alone multi-card training is normal, but during the multi-machine multi-card training, some machines prompt the GPU device id setting to fail?**</font>

```text
[ERROR] DEVICE [mindspore/ccsrc/runtime/device/gpu/cuda_driver.cc:245] SetDevice] SetDevice for id:7 failed, ret[101], invalid device ordinal. Please make sure that the 'device_id' set in context is in the range:[0, total number of GPU). If the environment variable 'CUDA_VISIBLE_DEVICES' is set, the total number of GPU will be the number set in the environment variable 'CUDA_VISIBLE_DEVICES'. For example, if export CUDA_VISIBLE_DEVICES=4,5,6, the 'device_id' can be 0,1,2 at the moment, 'device_id' starts from 0, and 'device_id'=0 means using GPU of number 4.
[ERROR] DEVICE [mindspore/ccsrc/runtime/device/gpu/gpu_device_manager.cc:27] InitDevice] Op Error: Failed to set current device id | Error Number: 0
```

A: In the multi-machine scenario, each process card number needs to be calculated after the host side `AllGather`and `HOSTNAME`. If the same `HOSTNAME` is used between machines, the process card number will be calculated incorrectly, causing the card number to cross the boundary and the setting to fail. This can be resolved by setting the HOSTNAME of each machine to its respective IP address in the execution script:

```text
export HOSTNAME=node_ip_address
```

<br/>

<font size=3>**Q: What can we do when performing multi-machine multi-card training via OpenMPI, the NCCL error message displays that the network is not working.?**</font>

```text
include/socket.h:403 NCCL WARN Connect to XXX failed: Network is unreachable
```

A: This problem is that `NCCL` cannot communicate with the peer address when synchronizing process information or initializing the communication domain on the Host side, which is generally caused by the different configuration of the network card between the machines, and the network card can be selected by setting the `NCCL` environment variable `NCCL_SOCKET_IFNAME`:

```text
export NCCL_SOCKET_IFNAME=eth
```

The above command sets the `NCCL` to select the network card name with `eth` in the Host side to communicate.