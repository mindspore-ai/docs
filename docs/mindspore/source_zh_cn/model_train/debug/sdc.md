# 特征值检测

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/model_train/debug/sdc.md)

## 概述

### 背景

模型训练过程中，处理器可能发生特征值检测异常，产生计算错误且无上报。特征值检测异常可能会造成对模型训练的严重负面影响。

### 解决方案

MindSpore框架提供了对Transformer结构模型的特征值检测方案。

对于默认的特征值检测点，用户可以设置环境变量 `NPU_ASD_ENABLE` 为`1`、`2`或`3`使能检测能力，并且通过配置环境变量 `NPU_ASD_UPPER_THRESH`, `NPU_ASD_SIGMA_THRESH`，调整检测强度。

此外，MindSpore框架支持用户根据需求，自定义特征值检测点，进一步提升对于特征值检测异常的检测能力。

关于相关环境变量的配置，见 **特性开关及配置**。

关于默认的特征值检测点的介绍，以及对于自定义特征值检测点的设计指导，见 **使用建议与检测原理** 。

### 使用建议与检测原理

处理器发生特征值检测异常时，计算得出错误结果。由于 Transformer 模型的结构，错误的计算结果会传播开来。

通过对实验结果进行统计，作以下经验性总结。

* 并非所有的特征值检测异常都一定影响模型的收敛和性能，事实上，大部分特征值检测异常对模型不产生可观测影响。 可见 [文献](https://dl.acm.org/doi/abs/10.1145/3579371.3589105)。
* 统计学意义上，反向传播计算过程中的特征值检测异常影响远大于正向计算过程中的影响。
* 在并行训练场景下，计算误差结果会由于并行计算而发生传播。
* 过多的检测点设置会影响模型训练性能。
* 根据计算错误检测敏感性实验结果，MindSpore框架默认选择反向传播计算过程中的`Norm`激活值梯度作为检测特征值，基于 **Llama 2 - 7B** 测试性能损失小于 2%。

开启检测开关后，针对Transformer结构模型训练的反向阶段，通过对修饰指定网络层从而调用检测算子，采集Norm层的激活值梯度，并通过算法判断是否异常。若出现异常，则终止训练，并将检测到异常的设备上的NPU状态置为Warning，上报故障事件。

特征值异常原因可分为两类：硬件错误与软件错误，可参考**故障处理**章节进行后续分析。

### 使用限制

目前本特性仅支持Atlas A2 训练系列产品，仅支持检测Transformer类模型，bfloat16数据类型，训练过程中出现的特征值检测异常。

## 特性开关及配置

环境变量`NPU_ASD_ENABLE`作为特性开关，`export NPU_ASD_ENABLE=1`、`export NPU_ASD_ENABLE=2`或`export NPU_ASD_ENABLE=3`开启本特性；不配置该环境变量或`export NPU_ASD_ENABLE=0`关闭本特性。

环境变量`NPU_ASD_UPPER_THRESH`控制检测的绝对数值阈值，格式为整型数据对，其中第一个元素控制绝对数值一级阈值，第二个元素控制绝对数值二级阈值；减小阈值可以检出波动更小的异常数据，增加检出率，增大阈值与之相反。在不配置该环境变量的默认情况下，`NPU_ASD_UPPER_THRESH=1000000,10000`。

环境变量`NPU_ASD_SIGMA_THRESH`控制检测的相对数值阈值，格式与上者相同，其中第一个元素控制数值跳变一级阈值，第二个元素控制数值跳变二级阈值；默认情况下，`NPU_ASD_SIGMA_THRESH=100000,5000`。

## 检测结果及处理

### 异常检测结果

未检测到数值异常时，对训练任务运行无影响。

当检测到数值异常后，训练任务失败并上报告警，请通过如下方法之一定位故障设备：

* 通过搜索应用类日志，查询**ERROR**级别错误日志，关键字"accuracy sensitivity feature abnormal"；
* 通过监控NPU健康状态：Health Status显示Warning，Error Code显示80818C00，Error Information显示node type=SoC, sensor type=Check Sensor, event state=check fail；
* 通过查看[Ascend Device Plugin](https://github.com/Ascend/ascend-device-plugin)事件，上报错误码80818C00，事件类型为故障事件，故障级别次要。

### 故障处理

将异常设备隔离，断点续训拉起继续训练；同时在异常设备上，通过Ascend-DMI工具执行AICore ERROR压测诊断，检测该设备上是否存在故障NPU。详情请查看[《ToolBox用户指南》](https://www.hiascend.com/document/detail/zh/mindx-dl/2046/dluserguide/toolboxug/toolboxug_000002.html) “ascend-dmi工具使用 > 故障诊断”章节。

若异常设备上检测到故障卡，请联系华为工程师维修更换；若异常设备上所有NPU均正常，则为软件类问题触发特征值溢出，建议排查程序和算子原因。