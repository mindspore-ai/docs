# 分布式配置

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/faq/source_zh_cn/distributed_configure.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

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

<font size=3>**Q: 基于Ascend环境需要配置通信配置文件，应该如何配置？**</font>

A: 请参考mindspore教程的基于Ascend分布式训练的[配置分布式环境变量](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.5/distributed_training_ascend.html#id4)部分。

<br/>

<font size=3>**Q: 如何进行分布式多机多卡训练？**</font>

A: 基于Ascend环境的，请参考mindspore教程的基于Ascend分布式训练的[多机多卡训练](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.5/distributed_training_ascend.html#id20) 部分。
基于GPU环境的，请参考mindspore教程的基于GPU分布式训练的[运行多机脚本](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.5/distributed_training_gpu.html#id8) 部分。
