# 精度敏感检测

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/tutorials/experts/source_zh_cn/debug/sdc.md)

## 概述

### 背景

模型训练过程中，处理器可能发生精度敏感异常，产生计算错误且无上报。精度敏感异常可能会造成对模型训练的严重负面影响。

### 解决方案

MindSpore框架提供了对Transformer结构模型的精度敏感检测方案。

对于默认的特征值检测点，用户可以使用环境变量 `NPU_ASD_ENABLE=1` 使能检测能力，并且通过配置环境变量 `NPU_ASD_UPPER_THRESH`, `NPU_ASD_SIGMA_THRESH`，调整检测强度。

此外，MindSpore框架支持用户根据需求，自定义特征值检测点，进一步提升对于精度敏感异常的检测能力。

关于相关环境变量的配置，见 **特性开关及配置**。

关于默认的特征值检测点的介绍，以及对于自定义特征值检测点的设计指导，见 **使用建议与检测原理** 。

### 使用建议与检测原理

处理器发生精度敏感异常时，计算得出错误结果。由于 Transformer 模型的结构，错误的计算结果会传播开来。

通过对实验结果进行统计，作以下经验性总结。

* 并非所有的精度敏感异常都一定影响模型的收敛和性能，事实上，大部分精度敏感异常对模型不产生可观测影响。 可见 [文献](https://dl.acm.org/doi/abs/10.1145/3579371.3589105)。
* 统计学意义上，反向传播计算过程中的精度敏感异常影响远大于正向计算过程中的影响。
* 在并行训练场景下，计算误差结果会由于并行计算而发生传播。
* 过多的检测点设置会影响模型训练性能。
* 根据计算错误检测敏感性实验结果，MindSpore框架默认选择反向传播计算过程中的`Norm`激活值梯度作为检测特征值，基于 **Llama 2 - 7B** 测试性能损失小于 2%。

开启检测开关后，针对Transformer结构模型训练的反向阶段，通过对修饰指定网络层从而调用检测算子，采集Norm层的激活值梯度，并通过算法判断是否异常。若出现异常，则终止训练，并将检测到异常的设备上的NPU状态置为Warning，上报故障事件。

特征值异常原因可分为两类：硬件错误与软件错误，可参考**故障处理**章节进行后续分析。

### 使用限制

目前本特性仅支持Atlas A2 训练系列产品，仅支持检测Transformer类模型，bfloat16数据类型，训练过程中出现的精度异常。

## 特性开关及配置

环境变量`NPU_ASD_ENABLE`作为特性开关，`export NPU_ASD_ENABLE=1`开启本特性；不配置该环境变量或`export NPU_ASD_ENABLE=0`关闭本特性。

环境变量`NPU_ASD_UPPER_THRESH`控制检测的绝对数值阈值，格式为整型数据对，其中第一个元素控制绝对数值一级阈值，第二个元素控制绝对数值二级阈值；减小阈值可以检出波动更小的异常数据，增加检出率，增大阈值与之相反。在不配置该环境变量的默认情况下，`NPU_ASD_UPPER_THRESH=1000000,10000`。

环境变量`NPU_ASD_SIGMA_THRESH`控制检测的相对数值阈值，格式与上者相同，其中第一个元素控制数值跳变一级阈值，第二个元素控制数值跳变二级阈值；默认情况下，`NPU_ASD_SIGMA_THRESH=100000,5000`。

## 使用用例

> 本文档介绍精度敏感检测的使用方法以及用例。

### 模型与数据集准备

为了提供完整的体验，这里基于 MindSpore Transformers Llama2 网络实现精度敏感检测的使用用例。

模型与数据集准备流程可见 [Llama 2](https://mindformers.readthedocs.io/zh-cn/latest/docs/model_cards/llama2.html)。

如已准备好，可直接跳过本章节。

### 默认检测流程用例

在`mindspore.ops.silent_check`模块下，已实现了`LayerNormASD`，作为集成了ASD检测能力的算子。

如果开启了特性开关，`mindspore.ops.__init__`中，以上算子会自动替换`mindspore.ops.LayerNorm`，提供默认的检测能力。

### 自定义检测流程用例

如果需要在默认检测场景之外，对于自定义的特征值进行检测，除了开启特性开关`NPU_ASD_ENABLE`之外，还需要自行实现基于`ASDBase Jit Class`，集成ASD检测能力的自定义算子。

这里使用 MindSpore Transformers Llama2 作示例，实现对 Embedding 层特征值的精度敏感检测。

#### 特征值检测点确认

检查`llama.llama_layer.LlamaEmbedding`的实现，这里选择采集`Gather`算子的反向传播梯度作为检测特征值。

```python
class LlamaEmbedding(Cell):
    def construct(self, input_ids):
        """Forward of vocab embedding."""
        _check_input_dtype(F.dtype(input_ids), "input_ids", [mstype.int32, mstype.int64], self.cls_name)
        output = self.gather(self.embedding_weight, input_ids, 0)
        return output
```

#### ASD检测算子实现

对于本用例，需要实现集成ASD检测能力的自定义Gather算子。

在`llama.llama_layer`下，实现采集点对应算子，实现方法可参考如下用例以及`ops.silent_check.ASDBase` API方法注释。

```python
class GatherASD(ASDBase):
    def __init__(self, *args, **kwargs):
        super().__init__(P.Gather, *args, **kwargs)
        self.pre_val, self.min_val, self.max_val, self.cnt = self.generate_params()

    def __call__(self, input_params, input_indices, axis):
        if self.enable_check:
            input_params = self.check_op(
                input_params, self.pre_val, self.min_val, self.max_val, self.cnt, None)
            self.cnt += 1
        return self.op(input_params, input_indices, axis)
```

并用自定义GatherASD算子替换Embedding层的默认Gather算子。

```python
class LlamaEmbedding(Cell):
    def __init__(self, vocab_table_size, embedding_size, param_init_type=mstype.float32, param_init='normal',
                 parallel_optimizer=False):
        super().__init__()
        self.vocab_table_size = vocab_table_size
        self.embedding_size = embedding_size
        self.embedding_weight = Parameter(
            initializer(param_init, [self.vocab_table_size, self.embedding_size], dtype=param_init_type),
            name='embedding_weight', parallel_optimizer=parallel_optimizer)
        self.gather = GatherASD()# Gather()
```

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