# 网络脚本分析

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/script_analysis.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 算子评估

### MindSpore算子设计

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

### 查询算子映射表

在代码库找到网络结构及实现训练功能的Python文件（名称一般为train.py model.py等等），在脚本文件中查找所有相关算子（含数据框架类、数据预处理、网络结构算子），并与[MindSpore算子API](https://www.mindspore.cn/docs/zh-CN/master/note/operator_list_ms.html)对比，查找`mindspore.nn`或者`mindspore.ops`下算子的平台支持情况。

若该网页均未能找到对应的ME算子，则可继续在[MindSpore API列表](https://www.mindspore.cn/docs/zh-CN/master/index.html)中搜索算子名称。

若源码为PyTorch脚本，则可以直接查询[MindSpore与PyTorch的算子映射](https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_api_mapping.html)找到对应的MindSpore算子。其他框架算子的映射可以参考算子命名与功能描述。注意，针对相同功能的算子，MindSpore的命名可能与其他框架不同，同名算子参数与功能也可能与其他框架有区别，均以官方描述为准。

### 缺失算子处理策略

1. 考虑用其他算子替换：需要分析算子实现公式，审视是否可以用现有MindSpore算子叠加达到预期目标。
2. 考虑使用自定义算子实现：参考[Custom算子的使用指南](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/custom_operator_custom.html)。
3. 考虑通过自定义算子方式使用其他已有第三方算子库实现：参考[基于自定义算子接口调用第三方算子库](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/use_third_party_op.html)。
4. 考虑临时规避方案：比如某个loss不支持，可以替换为同类已支持的loss算子。
5. [在MindSpore社区](https://gitee.com/mindspore/mindspore/issues)提交建议开发缺失算子。

## 语法评估

MindSpore提供`GRAPH_MODE`和`PYNATIVE_MODE`两种模式。

PyNative模式下模型进行**推理**的行为与一般Python代码无异。

而在使用GRAPH_MODE时，或使用PYNATIVE_MODE进行**训练**时，通常会出现语法限制。在这两种情况下，需要对Python代码进行图编译操作，而这一步操作中MindSpore目前还未能支持完整的Python语法全集，所以`construct`函数的编写会存在部分限制。具体限制内容可以参考[MindSpore静态图语法](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html)。

### 常见限制原则

相较于详细的语法说明，常见的限制可以归结为以下几点：

- 构图时不要调用其他Python库，例如numpy、scipy，相关的处理应该前移到`__init__`阶段。
- 构图时不要使用自定义类型，而应该使用MindSpore提供的数据类型和Python基础类型，可以使用基于这些类型的tuple/list组合。
- 构图时不要对数据进行多线程或多进程处理。

### 常见处理策略

1. 使用MindSpore内部提供的算子替换其他Python库的功能。常量的处理可以前移到`__init__`阶段。
2. 使用基础类型进行组合，可以考虑增加函数参数量。函数入参数没有限制，并且可以使用不定长输入。
3. 避免网络中出现多线程处理。
