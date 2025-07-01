# Distributed Parallel

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/faq/distributed_parallel.md)

## Q: What should I do if the error message `Init plugin so failed, ret = 1343225860` is displayed during the HCCL distributed training?

A: When the user starts distributed training on the Ascend and meets the error that HCCL fails to be initialized, the possible cause is that `rank_table.json` is incorrect. You can use the tool in [hccl_tools.py](https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/hccl_tools.py) to generate new `rank_table.json`. Alternatively, set the environment variable `export ASCEND_SLOG_PRINT_TO_STDOUT=1` to enable the log printing function of HCCL and check the ERROR log information.

<br/>

## Q: In the GPU distributed training scenario, what should I do if the number of environment variables CUDA_VISIBLE_DEVICES set incorrectly is less than the number of processes executed, the process blocking problem may occur?

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

## Q: What can we do when in the GPU distributed training scenario, if a process exits abnormally, it may cause other processes to block?

A: In this scenario, the abnormal process exits due to various problems, and the remaining processes are executed normally to the initialization `NCCL` step due to the successful allocation of GPU resources. The log is as follows:

```text
[INFO] DEVICE [mindspore/ccsrc/runtime/hardware/gpu/gpu_device_context.cc:90] Initialize] Start initializing NCCL communicator for device 1
```

In this step, the `NCCL` interface `ncclCommInitRank` is called, which blocks until all processes agree. So if a process doesn't call `ncclCommInitRank`, it will cause the process to block.
We have reported this issue to the `NCCL` community, and the community developers are designing a solution. The latest version has not been fixed, see [issue link](https://github.com/NVIDIA/nccl/issues/593#issuecomment-965939279).
Solution: Manually `kill` the training process and then restart the training task.

<br/>

## Q: When executing a GPU stand-alone single-card script, when the process is started without mpirun, calling the mindspore.communication.init method may report an error, resulting in execution failure, how to deal with it?

```text
[CRITICAL] DISTRIBUTED [mindspore/ccsrc/distributed/cluster/cluster_context.cc:130] InitNodeRole] Role name is invalid...
```

A: In the case where the user does not start the process using `mpirun` but still calls the `init()` method, MindSpore requires the user to configure several environment variables and verify according to training and [dynamic cluster startup methods](https://www.mindspore.cn/tutorials/en/master/parallel/dynamic_cluster.html). If without configuring, MindSpore may display the above error message. Therefore, it is suggested that only when performing distributed training, `mindspore.communication.init` is called, and in the case of not using `mpirun`, it is configured the correct environment variables according to the documentation to start distributed training.

<br/>

## Q: What can we do when performing multi-machine multi-card training via OpenMPI, the prompt fails due to MPI_Allgather?

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

## Q: What can we do when performing distributed training via OpenMPI, stand-alone multi-card training is normal, but during the multi-machine multi-card training, some machines prompt the GPU device id setting to fail?

```text
[ERROR] DEVICE [mindspore/ccsrc/runtime/device/gpu/cuda_driver.cc:245] SetDevice] SetDevice for id:7 failed, ret[101], invalid device ordinal. Please make sure that the 'device_id' set in context is in the range:[0, total number of GPU). If the environment variable 'CUDA_VISIBLE_DEVICES' is set, the total number of GPU will be the number set in the environment variable 'CUDA_VISIBLE_DEVICES'. For example, if export CUDA_VISIBLE_DEVICES=4,5,6, the 'device_id' can be 0,1,2 at the moment, 'device_id' starts from 0, and 'device_id'=0 means using GPU of number 4.
[ERROR] DEVICE [mindspore/ccsrc/runtime/device/gpu/gpu_device_manager.cc:27] InitDevice] Op Error: Failed to set current device id | Error Number: 0
```

A: In the multi-machine scenario, each process card number needs to be calculated after the host side `AllGather`and `HOSTNAME`. If the same `HOSTNAME` is used between machines, the process card number will be calculated incorrectly, causing the card number to cross the boundary and the setting to fail. This can be resolved by setting the HOSTNAME of each machine to its respective IP address in the execution script:

```text
export HOSTNAME=node_ip_address
```

<br/>

## Q: What can we do when performing multi-machine multi-card training via OpenMPI, the NCCL error message displays that the network is not working?

```text
include/socket.h:403 NCCL WARN Connect to XXX failed: Network is unreachable
```

A: This problem is that `NCCL` cannot communicate with the peer address when synchronizing process information or initializing the communication domain on the Host side, which is generally caused by the different configuration of the network card between the machines, and the network card can be selected by setting the `NCCL` environment variable `NCCL_SOCKET_IFNAME`:

```text
export NCCL_SOCKET_IFNAME=eth
```

The above command sets the `NCCL` to select the network card name with `eth` in the Host side to communicate.

<br/>

## Q: What can we do when after selecting RDMA NIC with a specific name (set via NCCL_SOCKET_IFNAME) for communication for multiple machines and multiple cards, the training still reports an error?

```text
misc/ibvwrap.cc:284 NCCL WARN Call to ibv_modify_qp failed with error Invalid argument
...
include/socket.h:403 NCCL WARN Connect to XXX failed: Connection refused
```

A: Generally this problem is the existence of differences in the configuration of the RDMA network card between multiple machines, which need to be analyzed on a case-by-case basis. However, the common reason is there are IB protocol and RoCE protocol on some host NICs at the same time, and there may be connection establishment failure. Solution:

The following command is required to specify the name of the RDMA NIC to be used as beginning with ib:

```text
export NCCL_IB_HCA=mlx
```

<br/>

## Q: What can we do when single-machine multi-card training is successful, but after expanding the script to multi-machine multi-card, other hosts prompt all kinds of errors?

There are various types of errors reported. Here are a few typical ones:

1. The installed whl package could not be found.
2. IB NIC communication failure.
3. Cuda library load failure.

A: These problems are caused by the fact that when `mpirun` starts other hosts, the environment variables of the other hosts (including the NCCL's NIC selection configurations) are not synchronized with the local machine, resulting in the phenomenon that single-machine multi-card executes normally while multi-machine multi-card fails. The solution is to export specific environment variables via mpirun -x option:

```text
mpirun --hostfile /path/to/hostfile -n 64 -x PYTHONPATH -x GLOG_v -x LD_LIBRARY_PATH -x NCCL_SOCKET_IFNAME -x NCCL_IB_HCA -x NCCL_DEBUG=INFO python train.py
```

The above command exports some environment variables that have been set on this machine to other hosts, ensuring that the environment variables of all hosts remain the same before executing the training script to achieve the goal of multi-machine multi-card training.

<br/>

## Q: Performing the distributed training via OpenMPI on Ascend, got `HcclCommInitRootInfo` error message. How can we deal it?

```text
Ascend collective Error: "HcclCommInitRootInfo failed. | Error Number 2
```

A: Currently, when training via OpenMPI, hccl needs to allocate about 300M device memory for each card within a communicator. The more communicators one card involved in, the more extra device memory needed. This probably cause memory issue.
You can set `max_size` in `set_memory`to reduce variable memory for Ascend processes, so that hccl will have enough memory to create communicators.

<br/>

## Q: When executing a distributed network under `auto_parallel`, an error is reported that the tensor cannot be split by the strategy. How can we solve it?

```text
np_tensor can not be split by strategy!
```

A: This error indicates that a strategy is configured for a parameter on the network, but a certain dimension of the parameter is not devisible by the strategy. There are two possible problems: 1. The parameter is used as the input of an operator, and the shard interface is called to set an illegel strategy for this operator. 2. When `dataset_strategy`="data_parallel" or `full_batch`=False is set in `auto_parallel_context`, the framework will automatically set a data-parallel strategy for network input. This error is also reported if the network input contains parameter whose shape cannot be divisible by the data-parallel strategy. However, auto-parallel only supports Tensor as network input, and you need to make adjustments to your script.

<br/>

## Q: What should I do if a process exits abnormally during the execution of multi-card training on a Linux environment and there is a shared memory residue through the ipcs command?

A: In the case of multi-card training and enabling graph operator fusion, the framework uses a shared memory mechanism for unified compilation of operators among multiple cards, and the shared memory is not effectively freed if the process ends unexpectedly due to internal or external exceptions during the compilation process. The ipcs command shows that nattch of the residual shared memory is 0. The framework will take over the shared memory again when the training script is re-executed, and it can be released normally as long as no exception occurs. You can also release the shared memory by ipcrm command, which will not affect the training script execution.

<br/>

## Q: I try to train a small network on Ascend platform but during initializing distributed module, `device memory not enough` exception is still thrown. How should I solve this issue?

A: This is because under the Ascend platform, the MindSpore backend defaults to pre-allocate a block of memory, with approximately 80% of NPU memory occupied and the remaining 20% used for initializing the HCCL collection communication library. Each HCCL communication group occupies 200 MB memory by default, so in scenarios with more communication groups, it is easy to encounter device side memory shortage errors. The solution is to set up `HCCL_BUFFSIZE` environment variable to change communication domain memory usage. Specific configuration method can refer to [HCCL official document](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/apiref/envvar/envref_07_0088.html).

<br/>

## Q: When using msrun to start a distributed framework, if I pass in the hostname as the master_addr but report an error that DNS resolution failed, how can I fix it?

```text
RuntimeError: DNS resolution failed: [Errno -2] Name or service not known. Please check whether the correct host name is input.
```

A: This is because when starting the distributed framework with msrun and specifying the master node by hostname, the DNS servers on the environment do not properly resolve the input hostname to an IP address. This can be due to the following:

1. The input hostname is incorrect, or the hostname does not exist in DNS. You can manually query the DNS information in Linux by using the command `nslookup <hostname>` or `dig <hostname>`, or you can query the static DNS resolution on the environment by using the command `cat /etc/hosts`.
2. DNS servers cannot be accessed normally. You can check the DNS server configuration in Linux with the command `cat /etc/resolv.conf`.
3. The firewall or security software organizes DNS queries. You can use the commands `systemctl status firewalld` and `service iptables status` in Linux to query the firewall and iptable status.

<br/>

## Q: When starting distributed framework using dynamic cluster or msrun in multi-machine scenario, an error is reported that device id is out of range. How can we solve it?

```text
RuntimeError: Ascend kernel runtime initialization failed, device id: 9. The details refer to 'Ascend Error Message'.

---------------------------------------------------
-Framework Error Message: (For framework developers)
---------------------------------------------------
Call aclrtSetDevice failed, ret[107001]. Got device count[8] and device id[9], please check if device id is valid.
```

A: This is because in a multi-machine scenario, the distributed framework automatically assigns a device_id/local_rank_id based on the hostname; if the hostname is the same, the assigned value will exceed the number of cards actually present on the machine. The hostname can be queried and modified in the following ways on a Linux system:

1. view the current hostname by the command `hostname` or `hostnamectl`.
2. You can modify the hostname by editing the file `/etc/hosts` or by using the command `hostnamectl set-hostname <hostname>`.
3. The hostname can be changed temporarily by using the command `hostname <hostname>`.

<br/>

## Q: How to resolve Unique ID retrieval timeout failures when using dynamic cluster or msrun to launch distributed tasks with HCCL backend in multi-cards scenarios?

```text
[WARNING] DEVICE(xxx,xxxxxx,python):xxxx-xx-xx-xx:xx:xx.xxx.xxx [mindspore/ccsrc/plugin/device/cpu/hal/hardware/ms_collective_comm_lib.cc:251] QueryUniqueID] Retry to lookup the unique id for group xxx from the meta server node...Retry time: 3/66, sleep 2
[WARNING] DEVICE(xxx,xxxxxx,python):xxxx-xx-xx-xx:xx:xx.xxx.xxx [mindspore/ccsrc/plugin/device/cpu/hal/hardware/ms_collective_comm_lib.cc:251] QueryUniqueID] Retry to lookup the unique id for group xxx from the meta server node...Retry time: 2/66, sleep 2
[WARNING] DEVICE(xxx,xxxxxx,python):xxxx-xx-xx-xx:xx:xx.xxx.xxx [mindspore/ccsrc/plugin/device/cpu/hal/hardware/ms_collective_comm_lib.cc:251] QueryUniqueID] Retry to lookup the unique id for group xxx from the meta server node...Retry time: 1/66, sleep 1
···
RuntimeError: Communicator of group xxx inited: failed. Result: Init communicator for group xxx exception info: Failed to fetch the unique id of the collective lib from the meta server node. Maybe the root rank process of this group has exited or has not executed to QueryUniqueID step. Please check root rank: 0's log.

--------------------------------------------
- C++ Call Stack: (For framework developers)
--------------------------------------------
mindspore/ccsrc/distributed/collective/collective_manager.cc:1123 WaitCommInitDone
mindspore/ccsrc/plugin/device/cpu/hal/hardware/ms_collective_comm_lib.cc:260 QueryUniqueID
```

A: The error occurs during communication group creation when non-rootrank processes fail to obtain rootinfo from the scheduler process within the timeout period. This typically happens in multi-card HCCL backend scenarios where neither the `RANK_TABLE_FILE` environment variable is set, nor the `--rank_table_file` parameter is specified in msrun. In such cases, the framework automatically uses HCCL's self-negotiation interface for communication group initialization. During the communication group creation phase, the rootrank process within the same communication group calls the HCCL interface to obtain rootinfo and then passes it to the scheduler process via a host-side TCP connection. Other rank processes within the communication group request this rootinfo from the scheduler process via TCP. To ensure all ranks within the same communication group obtain the rootinfo before proceeding with HCCL initialization, the framework provides the capability to retry QueryUniqueID with a timeout period, which defaults to 200s. Here are the solutions to this error:

1. Check `scheduler.log` to verify whether the scheduler process status is abnormal. Normally, the scheduler process waits for worker processes to complete their tasks, as shown below:

```text
[WARNING] DISTRIBUTED(xxx,xxxxxx,python):xxxx-xx-xx-xx:xx:xx.xxx.xxx [mindspore/ccsrc/distributed/cluster/cluster_context.cc:154] Finalize] This log means the cluster is successfully created. Retry to finalize the node and exit cluster...
[WARNING] DISTRIBUTED(xxx,xxxxxx,python):xxxx-xx-xx-xx:xx:xx.xxx.xxx [mindspore/ccsrc/distributed/cluster/topology/meta_server_node.cc:98] Finalize] The meta server node can not be finalized because there are still 256 alive nodes.
[WARNING] DISTRIBUTED(xxx,xxxxxx,python):xxxx-xx-xx-xx:xx:xx.xxx.xxx [mindspore/ccsrc/distributed/cluster/cluster_context.cc:154] Finalize] This log means the cluster is successfully created. Retry to finalize the node and exit cluster...
[WARNING] DISTRIBUTED(xxx,xxxxxx,python):xxxx-xx-xx-xx:xx:xx.xxx.xxx [mindspore/ccsrc/distributed/cluster/topology/meta_server_node.cc:98] Finalize] The meta server node can not be finalized because there are still 256 alive nodes.
[WARNING] DISTRIBUTED(xxx,xxxxxx,python):xxxx-xx-xx-xx:xx:xx.xxx.xxx [mindspore/ccsrc/distributed/cluster/cluster_context.cc:154] Finalize] This log means the cluster is successfully created. Retry to finalize the node and exit cluster...
[WARNING] DISTRIBUTED(xxx,xxxxxx,python):xxxx-xx-xx-xx:xx:xx.xxx.xxx [mindspore/ccsrc/distributed/cluster/topology/meta_server_node.cc:98] Finalize] The meta server node can not be finalized because there are still 256 alive nodes.
```

2. In large cluster scale, the number of communication groups may increase, and the number of ranks within a single communication group may also grow. In such cases, the default 200s timeout may be insufficient for all rank processes to request rootinfo information from the scheduler process. You can manually configure the environment variable MS_NODE_TIMEOUT to adjust the timeout period. For example:

```text
export MS_NODE_TIMEOUT=900
```

Since QueryUniqueID is a host-side operation, it can be affected by host-side network fluctuations, throughput, and CPU performance. It is recommended to appropriately increase the timeout period based on the cluster networking scale. For clusters with more than 128 devices, it is recommended to configure 900s~1800s.
