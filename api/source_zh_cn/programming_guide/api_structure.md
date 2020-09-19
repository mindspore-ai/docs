# MindSpore API概述

<!-- TOC -->

- [MindSpore API概述](#mindsporeapi概述)
    - [总体架构](#总体架构)
    - [设计理念](#设计理念)
    - [层次结构](#层次结构)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/api/source_zh_cn/programming_guide/api_structure.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 总体架构
MindSpore是一个全场景深度学习框架，旨在实现易开发、高效执行、全场景覆盖三大目标，其中易开发表现为API友好、调试难度低以及额外的自动化属性，高效执行包括计算效率、数据预处理效率和分布式训练效率，全场景则指框架同时支持云、边缘以及端侧场景。

MindSpore总体架构分为前端表示层（Mind Expression，ME）、计算图引擎（Graph Engine，GE）和后端运行时三个部分。ME提供了用户级应用软件编程接口（Application Programming Interface，API），用于构建和训练神经网络，并将用户的Python代码转换为数据流图。GE是算子和硬件资源的管理器，负责控制从ME接收的数据流图的执行。后端运行时包含云、边、端上不同环境中的高效运行环境，例如CPU、GPU、Ascend AI处理器、 Android/iOS等。更多总体架构的相关内容请参见[总体架构](https://www.mindspore.cn/docs/zh-CN/master/architecture.html)。

## 设计理念

MindSpore源于全产业的最佳实践，向数据科学家和算法工程师提供了统一的模型训练、推理和导出等接口，支持端、边、云等不同场景下的灵活部署，推动深度学习和科学计算等领域繁荣发展。

MindSpore提供了Python编程范式，用户使用Python原生控制逻辑即可构建复杂的神经网络模型，AI编程变得简单，具体示例请参见[实现一个图片分类应用](https://www.mindspore.cn/tutorial/zh-CN/master/quick_start/quick_start.html)。

目前主流的深度学习框架的执行模式有两种，分别为静态图模式和动态图模式。静态图模式拥有较高的训练性能，但难以调试。动态图模式相较于静态图模式虽然易于调试，但难以高效执行。MindSpore提供了动态图和静态图统一的编码方式，大大增加了静态图和动态图的可兼容性，用户无需开发多套代码，仅变更一行代码便可切换动态图/静态图模式，例如设置`context.set_context(mode=context.PYNATIVE_MODE)`切换成动态图模式，设置`context.set_context(mode=context.GRAPH_MODE)`即可切换成静态图模式，用户可拥有更轻松的开发调试及性能体验。

神经网络模型通常基于梯度下降算法进行训练，但手动求导过程复杂且梯度难以计算。MindSpore的基于源码转换（Source Code Transformation，SCT）的自动微分（Automatic Differentiation）机制采用函数式可微分编程架构，在接口层提供Python编程接口，包括控制流的表达。用户可聚焦于模型算法的数学原生表达，无需手动进行求导，在动态图模式下自动微分的样例代码如下所示。

```python
import mindspore as ms
from mindspore.ops import composite as C
from mindspore import context
from mindspore.common import Tensor


context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


def cost(x, y): return x * (x + y)


def test_grad(x, y):
    return C.GradOperation(get_all=True)(cost)(Tensor(x, dtype=ms.float32), Tensor(y, dtype=ms.float32))


def main():
    return test_grad(2, 1)

```

其中，第一步定义了一个函数（计算图），第二步利用MindSpore提供的反向接口进行自动微分，定义了一个反向函数（计算图），最后给定一些输入就能获取第一步定义的函数在指定处的导数，求导结果为`(5, 2)`。

此外，SCT能够将Python代码转换为函数中间表达（Intermediate Representation，IR），函数中间表达构造出能够在不同设备解析和执行的计算图，并且在执行该计算图前，应用了多种软硬件协同优化技术，端、边、云等不同场景下的性能和效率得到针对性的提升。

随着神经网络模型和数据集的规模不断增加，分布式并行训练成为了神经网络训练的常见做法，但分布式并行训练的策略选择和编写十分复杂，这严重制约着深度学习模型的训练效率，阻碍深度学习的发展。MindSpore统一了单机和分布式训练的编码方式，开发者无需编写复杂的分布式策略，在单机代码中添加少量代码即可实现分布式训练，例如设置`context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL)`便可自动建立代价模型，为用户选择一种较优的并行模式，提高神经网络训练效率，大大降低了AI开发门槛，使用户能够快速实现模型思路。

## 层次结构

MindSpore向用户提供了3个不同层次的API，支撑用户进行网络构建、整图执行、子图执行以及单算子执行，从低到高分别为Low-Level Python API、Medium-Level Python API以及High-Level Python API。

![img](./images/api_structure.png) 

- Low-Level Python API

  第一层为低阶API，主要包括张量定义、基础算子、自动微分等模块，用户可使用低阶API轻松实现张量定义和求导计算，例如用户可通过`Tensor`接口自定义张量，使用`ops.composite`模块下的`GradOperation`算子计算函数在指定处的导数。

- Medium-Level Python API

  第二层为中阶API，其封装了低价API，提供网络层、优化器、损失函数等模块，用户可通过中阶API灵活构建神经网络和控制执行流程，快速实现模型算法逻辑，例如用户可调用`Cell`接口构建神经网络模型和计算逻辑，通过使用`loss`模块和`Optimizer`接口为神经网络模型添加损失函数和优化方式。

- High-Level Python API

  第三层为高阶API，其在中阶API的基础上又提供了训练推理的管理、Callback、混合精度训练等高级接口，方便用户控制整网的执行流程和实现神经网络的训练及推理，例如用户使用`Model`接口，指定要训练的神经网络模型和相关的训练设置，即可对神经网络模型进行训练。
