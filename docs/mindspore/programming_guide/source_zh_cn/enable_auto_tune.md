# 使能算子调优工具

`Linux` `Ascend` `AutoTune` `算子编译` `中级` `高级`

<!-- TOC -->

- [使能算子调优工具](#使能算子调优工具)
    - [概述](#概述)
    - [调优模式](#调优模式)
    - [环境变量](#环境变量)
    - [开启调优](#开启调优)
    - [须知](#须知)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/enable_auto_tune.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>&nbsp;&nbsp;

## 概述

AutoTune是使用硬件资源对TBE算子性能进行自动调优的一个工具。相比于人工对算子性能调试，耗时更短、人力投入成本低，可以获得性能更优的模型。本文档主要介绍AutoTune的在线调优使用方法，该工具的架构，功能描述，使用指南以及常见故障处理方的法详细介绍请参考[AutoTune工具指南](https://support.huawei.com/enterprise/zh/doc/EDOC1100206690/31d1d888)。

## 调优模式

AutoTune工具包含RL和GA两种调优模式。其中, `RL`主要支持elewise、broadcast、reduce类的算子；`GA`主要支持cube类的算子。有关两种调优模式的定义和详细介绍，以及两种调优模式分别支持的算子列表请分别参考[调优模式](https://support.huawei.com/enterprise/zh/doc/EDOC1100206690/41bb2c07) 和 [算子列表](https://support.huawei.com/enterprise/zh/doc/EDOC1100206690/74e08a9c)。

## 环境变量

启用AutoTune工具时，需要配置相关环境变量（必选），具体的环境变量及其功能介绍请参考[环境变量](https://support.huawei.com/enterprise/zh/doc/EDOC1100206690/58a01d46)。

## 开启调优

MindSpore对接AutoTune工具接口，支持`在线调优`和`离线调优`。其中，通过在context接口中设置`auto_tune_mode`即可开启在线调优。离线调优则是使用训练脚本生成网络模型时的DUMP数据（包含算子输出描述文件、算子的二进制文件等）进行算子调优。下面主要介绍在线调优的使用方法，离线调优的使用方法以及注意事项请参考[离线调优](https://support.huawei.com/enterprise/zh/doc/EDOC1100206690/2fa72dd0)。

在context接口中设置`auto_tune_mode`，即可开启AutoTune工具进行在线调优。`auto_tune_mode`的取值应当在`["NO_TUNE", "RL", "GA", "RL,GA"]`中。其中：

NO_TUNE：不开启调优（关闭调优）。

RL：开启RL调优，针对支持RL调优的算子进行调优。

GA：开启GA调优，针对支持GA调优的算子进行调优。

RL,GA：同时开启RL和GA调优，工具会根据网络模型中不同类型的算子自动选择RL或者GA。不区分RL，GA的先后顺序。

举例在线调优的使用方法：

```python
import mindspore.context as context

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", auto_tune_mode="GA,RL")

...
```

## 须知

AutoTune调优工具在使用时，请注意以下几点：

1. AutoTune调优工具只支持在`Ascend`环境上使用。

2. 请确保运行环境中执行调优用户的home目录下磁盘可用空间>=20G.

3. AutoTune调优工具依赖部分第三方软件，如`TensorFlow`和`pciutils`等。具体依赖软件的版本等详细信息请参考[依赖](https://support.huawei.com/enterprise/zh/doc/EDOC1100206690/480d602c)。

4. AutoTune调优工具仅能对`GA`和`RL`所支持的算子进行调优，且无法保证所有的算子经过该工具调优之后都有性能收益（部分算子经过多个网络以及多次人工调试，性能已经达到最优）。

5. 开启该调优工具后，可以明显感知算子编译时间变长，属于正常现象。
