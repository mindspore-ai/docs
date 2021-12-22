# JIT_Fallback

`Ascend` `GPU` `CPU` `模型运行`

<!-- TOC -->

- [JIT_Fallback](#JIT_Fallback)
    - [概述](#概述)
    - [实现原理](#实现原理)
    - [支持范围](#支持范围)
    - [使用须知](#使用须知)

<!-- /TOC -->

## 概述

MindSpore框架支持静态图模式和动态图模式两种方式。图模式的语法要求严格，编译函数内原则上只支持算子调用和部分映射过的Python内置方法。而动态图模式理论上支持使用任意Python语法。为实现动静统一，保障用户可以灵活地进行动静态图的切换，JIT Fallback特性从静态图的角度出发，在编译过程中发现图模式不支持语法时，将相关语句JIT Fallback到Python解释器进行解释执行。在图模式下尽量多地支持动态图模式下的语法，从而实现动静统一，降低图模式编程门槛。

当前JIT Fallback有限支持常量场景。包括但不限于在construct/ms_function中创建和使用Tensor、调用NumPy等第三方库中的对象和方法、支持在construct/ms_function使用print打印等。

本文档主要介绍JIT Fallback的使用方法和工作原理，以便您可以更有效地使用JIT Fallback功能。

## 实现原理

MindSpore图模式采用MindIR中间表示来表达图节点之间的关系，为表达方便，对在图模式下不支持需要JIT Fallback到Python解释器进行解释执行的语法，引入解释节点进行表达。在Parser阶段，识别出不支持语法，使用解释节点记录相关信息。在解释节点的类型推导阶段，通过Python侧的eval()函数对不支持语法的表达式进行解释执行，返回执行结果作为表达式的类型推导结果。对于常量场景，可以在类型推导阶段，完成常量的推导和解释执行。

下面简单介绍解释节点的构成。

![interpretnode](./design/images/interpretnode.png)

解释节点是MindIR中间表示中一类特殊的CNode类型节点，其中primitive指prim::kPrimPyInterpret；script是string类型，用于记录图模式下不支持，需要JIT Fallback解释执行的语句；globals和locals分别用于记录全局变量作用域信息和局部变量作用域信息，便于使用Python侧的eval()函数解释执行。

在编译过程中，会不断更新globals和locals信息，完成eval函数解释执行后，将执行结果转换成MindSpore支持的数据类型，用于后续图模式的流程。

## 支持范围

目前MindSpore图模式有条件地支持常量场景，尽量多地支持动态图模式下的语法，包括但不限于在construct/ms_function中创建和使用Tensor、调用NumPy等第三方库中的对象和方法、支持在construct/ms_function使用print打印等。下面对各场景进行简单举例说明。

1. 在construct/ms_function中创建和使用Tensor。

    对于用例中的tensor_num = Tensor(np.array(9))是图模式下的不支持语法，会在编译过程中解析成解释节点。

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

    需要使用JIT Fallback特性来支持的语句，会打印相关提示信息，如下：

    ```text
    Found unsupported syntax in Graph mode, those codes would be fallen back to Python interpreter:
    tensor_num = Tensor(np.array(9))
    ```

    为了对比，我们可以通过关闭JIT Fallback特性的开关，来观察没有JIT Fallback特性时该用例的执行结果，即设置export MS_DEV_ENABLE_FALLBACK=0，用例执行结果如下：

    ```text
    Meet a exception from Python when get the type of '<built-in function array>'
    TypeError: Not support for this object with type '<class 'builtin_function_or_method'>' and value '<built-in function array>'
    ```

2. 在construct/ms_function中调用NumPy等第三方库中的对象和方法。

    对于用例中的a = np.array([1, 2, 3])和b = np.array([4, 5, 6])是图模式下的不支持语法，会在编译过程中解析成解释节点。

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

    需要使用JIT Fallback特性来支持的语句，会打印相关提示信息，如下：

    ```text
    Found unsupported syntax in Graph mode, those codes would be fallen back to Python interpreter:
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = a + b
    return Tensor(c)
    ```

    为了对比，我们可以通过关闭JIT Fallback特性的开关，来观察没有JIT Fallback特性时该用例的执行结果，即设置export MS_DEV_ENABLE_FALLBACK=0，用例执行结果如下：

    ```text
    Meet a exception from Python when get the type of '<built-in function array>'
    TypeError: Not support for this object with type '<class 'builtin_function_or_method'>' and value '<built-in function array>'
    ```

3. 支持在construct/ms_function使用print打印。

    在常量场景中，通过JIT Fallback特性使用Python原生的print来打印常量，与图模式中使用print算子来打印信息的时机有所不同。由于Python原生print是在编译过程中触发打印，而图模式调用算子打印是需要图中所有节点构图结束后下发到设备端运行才打印。

    为了便于理解，举例如下。tensor_sum是有两个Tensor变量相加得到结果，需要在运行阶段才可以得到结果，即需要使用图模式中的print算子打印信息；而np_sum是由两个NumPy常量对象相加得到结果，即在编译阶段使用Python原生print能力来打印信息。导致最终显示np_sum会在tensor_sum之前，这是编译时运行方式和运行时运行方式的区别。

    ```python
    import numpy as np
    from mindspore import context, Tensor, ms_function

    @ms_function
    def test_print():
        x = Tensor(np.array([1, 2, 3, 4, 5]))
        y = Tensor(np.array([1, 2, 3, 4, 5]))
        tensor_sum = x + y
        print("tensor_sum: ", tensor_sum)
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        np_sum = x + y
        print("np_sum: ", np_sum)
        return tensor_sum, Tensor(np_sum)

    context.set_context(mode=context.GRAPH_MODE)
    tensor_sum, np_sum = test_print()
    ```

    输出结果如下:

    ```text
    np_sum: [2 4 6 8 10]
    tensor_sum: (2, 4, 6, 8, 10)
    ```

## 使用须知

在使用JIT Fallback在图模式下尽量多地支持动态图语法，请注意以下几点：

1. 当前JIT Fallback仅支持常量场景，即值明确且保持不变，不因外界影响而修改，不以参数传入的场景。

2. JIT Fallback对标动态图的支持能力，须在动态图语法范围内，包括但不限于数据类型等。

3. 为了便于用户选择是否使用JIT Fallback特性的能力，提供了开关MS_DEV_ENABLE_FALLBACK，当前默认已经打开。如果需要关闭，可以使用命令：export MS_DEV_ENABLE_FALLBACK=0。

4. JIT Fallback使用解释节点来表达，解释节点仅用于Python eval()解释，不能够传递到后端执行。为了便于理解，举例如下。由于np.add(x, y)是图模式下不支持的语法，即会在编译阶段解析成为解释节点。如果解释节点作为函数的返回值，则将传递到后端执行，当前后端不支持解释节点。所以该类场景当前不支持。

    ```python
    import numpy as np
    from mindspore import context, Tensor, ms_function

    @ms_function
    def test_np_add():
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        return np.add(x, y)

    context.set_context(mode=context.GRAPH_MODE)
    np_add_res = test_np_add()
    ```

    输出结果如下:

    ```text
    Should not use Python object in runtime, node: ValueNode<InterpretedObject> InterpretedObject: '[2 4 6 8 10]'
    ```

5. 当前有限支持控制流场景，将逐步在后续版本中打通。

6. 当前暂不支持自定义Class的attr/method，将逐步在后续版本中打通。
