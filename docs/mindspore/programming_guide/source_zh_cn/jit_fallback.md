# JIT Fallback

`Ascend` `GPU` `CPU` `模型运行`

<!-- TOC -->

- [JIT_Fallback](#JIT_Fallback)
    - [概述](#概述)
    - [实现原理](#实现原理)
    - [支持范围](#支持范围)
    - [使用须知](#使用须知)

<!-- /TOC -->

## 概述

MindSpore框架支持图模式和PyNative模式两种方式。图模式的语法要求严格，编译函数内原则上只支持算子调用和部分映射过的Python内置方法。而PyNative模式理论上支持使用任意Python语法。为实现动静统一，保障用户可以灵活地进行动静态图的切换，Fallback特性从静态图的角度出发，在编译过程中发现图模式不支持语法时，将相关语句Fallback到Python解释器进行解释执行。在图模式下尽量多地支持PyNative模式下的语法，从而实现动静统一，降低图模式编程门槛。

当前JIT Fallback有限支持常量场景。包括但不限于在construct/ms_function中创建和使用Tensor、调用NumPy第三方库中的对象和方法等。

本文档主要介绍JIT Fallback的使用方法和工作原理，以便您可以更有效地使用JIT Fallback功能。

## 实现原理

MindSpore图模式采用MindIR中间表示来表达图节点之间的关系，为表达方便，对在图模式下不支持需要Fallback到Python解释器进行解释执行的语法，引入解释节点进行表达。在Parser阶段，识别出不支持语法，使用解释节点记录相关信息。在解释节点的类型推导阶段，通过Python侧的eval()函数对不支持语法的表达式进行解释执行，返回执行结果作为表达式的类型推导结果。对于常量场景，可以在类型推导阶段，完成常量的推导和解释执行。

下面简单介绍解释节点的构成。

![interpretnode](./design/images/interpretnode.png)

解释节点是MindIR中间表示中一类特殊的CNode类型节点，其中primitive指prim::kPrimPyInterpret；script是string类型，用于记录图模式下不支持，需要Fallback解释执行的语句；globals和locals分别用于记录全局变量作用域信息和局部变量作用域信息，便于使用Python侧的eval()函数解释执行。

在编译过程中，会不断更新globals和locals信息，完成eval函数解释执行后，将执行结果转换成MindSpore支持的数据类型，用于后续图模式的流程。

## 支持范围

目前MindSpore图模式有条件地支持常量场景，即在construct/ms_function中创建和使用Tensor、调用NumPy第三方库中的对象和方法等。下面对各场景进行简单举例说明。

1. 在construct/ms_function中创建和使用Tensor

    ```python
    import numpy as np
    import mindspore.nn as nn
    from mindspore import context, Tensor

    class BinOpNet(nn.Cell):
        def __init__(self):
            super(BinOpNet, self).__init__()

        def construct(self):
            tensor_num = Tensor(np.array(9))
            res = tensor_num + tensor_num
            return res

    context.set_context(mode=context.GRAPH_MODE)
    net = BinOpNet()
    print(net())
    ```

    输出结果如下:

    ```text
    18
    ```

2. 在construct/ms_function中调用NumPy第三方库中的对象和方法

    ```python
    import numpy as np
    from mindspore import context, Tensor, ms_function

    @ms_function
    def np_binop():
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = a + b
        return Tensor(c)

    context.set_context(mode=context.GRAPH_MODE)
    res = np_binop()
    print(res)
    ```

    输出结果如下:

    ```text
    [5 7 9]
    ```

## 使用须知

在使用JIT Fallback在图模式下尽量多地支持PyNative模式下的语法，请注意以下几点：

1. JIT Fallback对标PyNative的支持能力，须在PyNative编译支持的语法范围内，包括但不限于数据类型等。

2. JIT Fallback使用解释节点来表达，解释节点仅用于Python eval()解释，不能够传递到后端执行。

3. 当前JIT Fallback仅支持常量场景，即值明确且保持不变，不因外界影响而修改，不以参数传入的场景。
