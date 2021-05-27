# Distributed Configure

`Linux` `Windows` `Ascend` `GPU` `CPU` `Environment Preparation` `Basic` `Intermediate`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_cn/distributed_configure.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

<font size=3>**Q: How to fix the error below when running MindSpore distributed training with GPU:**</font>

```text
Loading libgpu_collective.so failed. Many reasons could cause this:
1.libgpu_collective.so is not installed.
2.nccl is not installed or found.
3.mpi is not installed or found
```

A: This message means that MindSpore failed to load library `libgpu_collective.so`. The Possible causes are:

- OpenMPI or NCCL is not installed in this environment.
- NCCL version is not updated to `v2.7.6`: MindSpore `v1.1.0` supports GPU P2P communication operator which relies on NCCL `v2.7.6`. `libgpu_collective.so` can't be loaded successfully if NCCL is not updated to this version.

<br/>

<font size=3>**Q：The communication profile file needs to be configured on the Ascend environment, how should it be configured?**</font>

A：Please refer to the [Configuring Distributed Environment Variables](https://mindspore.cn/tutorial/training/en/master/advanced_use/distributed_training_ascend.html#configuring-distributed-environment-variables) section of Ascend-based distributed training in the MindSpore tutorial.

<br/>

<font size=3>**Q：How to perform distributed multi-machine multi-card training?**</font>

A：For Ascend environment, please refer to the [Multi-machine Training](https://mindspore.cn/tutorial/training/en/master/advanced_use/distributed_training_ascend.html#multi-machine-training) section of the MindSpore tutorial "distributed_training_ascend".
For GPU-based environments, please refer to the [Run Multi-Host Script](https://mindspore.cn/tutorial/training/en/master/advanced_use/distributed_training_gpu.html#running-the-multi-host-script) section of the MindSpore tutorial "distributed_training_gpu".
