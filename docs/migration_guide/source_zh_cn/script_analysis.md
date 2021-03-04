# 算子评估

<!-- TOC -->

- [算子评估](#算子评估)
    - [MindSpore算子设计](#mindspore算子设计)
    - [查询算子映射表](#查询算子映射表)
    - [缺失算子处理策略](#缺失算子处理策略)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/migration_guide/source_zh_cn/script_analysis.md" target="_blank"><img src="./_static/logo_source.png"></a>

## MindSpore算子设计

使用MindSpore框架搭建神经网络流程与其他框架（TensorFlow/PyTorch）类似，但支持的算子存在差异，需要在进行网络迁移（例如由TensorFlow迁移至MindSpore Ascend平台）时找出MindSpore框架缺失的算子。

MindSpore API由各种Python/C++ API算子组成，可以大致分为：

- 数据框架算子

  包括张量、基本数据类型、训练梯度、优化器算子，如`mindspore.int32`、`mindspore.nn.Cell`等。

- 数据预处理算子

  包括图片读取、数据类型转化算子，如`mindspore.dataset.MnistDataset`等。

- 网络结构算子

  包括网络构建中使用到的卷积、归一化算子，如`mindspore.nn.Conv2d`、`mindspore.nn.Dense`等。

  网络结构算子表层为ME算子，即用户调用的算子API（例如`mindspore.nn.Softmax`），ME算子底层调用TBE算子（C/C++）实现。

  统计缺失ME算子时，需要找出源码脚本中所有算子（含数据框架类、数据预处理、网络结构算子）在MindSpore框架的对应算子（例如`tf.nn.relu`对应MindSpore算子为`mindspore.nn.ReLU`）。如果MindSpore中没有对应算子，则计入缺失。

## 查询算子映射表

在代码库找到网络结构及实现训练功能的Python文件（名称一般为train.py model.py等等），在脚本文件中查找所有相关算子（含数据框架类、数据预处理、网络结构算子），并与[MindSpore算子API](https://www.mindspore.cn/doc/note/zh-CN/master/operator_list_ms.html)对比，查找“mindspore.nn”或者“mindspore.ops.operations”下算子的平台支持情况，目前支持Ascend、CPU与GPU。

若该网页均未能找到对应的ME算子，则可继续在[MindSpore API列表](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)中搜索算子名称。

若源码为PyTorch脚本，则可以直接查询[MindSpore与PyTorch的算子映射](https://www.mindspore.cn/doc/note/zh-CN/master/index.html#operator_api)找到对应的MindSpore算子。注意，针对相同功能的算子，MindSpore的命名可能与其他框架不同，同名算子参数与功能也可能与其他框架有区别，均以官方描述为准。

## 缺失算子处理策略

1. 考虑用其他算子替换：需要分析算子实现公式，审视是否可以用现有MindSpore算子叠加达到预期目标。
2. 考虑临时规避方案：比如某个loss不支持，可以替换为同类已支持的loss算子。
3. [在MindSpore社区](https://gitee.com/mindspore/mindspore/issues)提交建议开发缺失算子。
