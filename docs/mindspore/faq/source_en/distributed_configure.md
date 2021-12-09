# Distributed Configure

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/faq/source_en/distributed_configure.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

<font size=3>**Q: What do I do if the error `Init plugin so failed, ret = 1343225860` occurs during the HCCL distributed training?**</font>

A: HCCL fails to be initialized. The possible cause is that `rank json` is incorrect. You can use the tool in `mindspore/model_zoo/utils/hccl_tools` to generate one. Alternatively, import the environment variable `export ASCEND_SLOG_PRINT_TO_STDOUT=1` to enable the log printing function of HCCL and check the log information.

<br/>

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
