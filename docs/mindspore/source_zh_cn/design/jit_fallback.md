# JIT Fallback

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/design/jit_fallback.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

MindSpore框架支持静态图模式和动态图模式两种方式。在静态图模式下，先将Python代码编译成静态计算图，然后执行静态计算图。由于语法解析的限制，用户编写程序时需要遵循MindSpore[静态图语法支持](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/note/static_graph_syntax_support.html)，语法使用存在约束限制。在动态图模式下，Python代码会通过Python解释器执行，用户可以使用任意Python语法。可以看到，静态图和动态图的编译流程不一致，语法约束限制也不同。关于静态图和动态图的更多介绍，请参考[静态图和动态图](https://www.mindspore.cn/tutorials/zh-CN/r2.0.0-alpha/advanced/compute_graph.html)。

JIT Fallback是从静态图的角度出发考虑静态图和动态图的统一。通过JIT Fallback特性，静态图可以支持尽量多的动态图语法，使得静态图提供接近动态图的语法使用体验，从而实现动静统一。为了便于用户选择是否使用JIT Fallback特性的能力，提供了开关`MS_DEV_ENABLE_FALLBACK`，当前默认已经打开。如果需要关闭，可以使用命令：`export MS_DEV_ENABLE_FALLBACK=0`。

本文档主要介绍JIT Fallback的支持范围和使用须知，以便您可以更有效地使用JIT Fallback功能。

## 支持范围

当前JIT Fallback特性应用于常量场景，即要求在编译期间能够确定实际值。JIT Fallback特性还在持续完善中，下面列举出当前通过该特性已经支持的静态图编译语法。

### 创建和使用Tensor

JIT Fallback支持在静态图模式下创建和使用[Tensor](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/mindspore/mindspore.Tensor.html)，暂不支持Tensor.asnumpy()。

代码用例如下，用例中的`Tensor(1, dtype=mstype.int32)`是通过JIT Fallback支持的。

```python
import mindspore as ms
import mindspore.nn as nn

class Net(nn.Cell):
   def __init__(self):
      super(Net, self).__init__()

   def construct(self):
      return ms.Tensor(1, dtype=ms.int32)

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
print(net())
```

输出结果：

```text
1
```

### 调用第三方库

JIT Fallback支持在静态图模式下调用第三方库的对象和方法。

需要说明的是，对于具有返回值的方法，需要使用变量来保存其结果，否则可能出现报错。这个用法将在后续版本中支持。

调用第三方库的代码用例如下。用例调用了NumPy第三方库，其中`np.array([1, 2, 3])`和`np.array([4, 5, 6])`是通过JIT Fallback支持的。

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn

class Net(nn.Cell):
   def __init__(self):
      super(Net, self).__init__()

   def construct(self):
      a = np.array([1, 2, 3])
      b = np.array([4, 5, 6])
      c = a + b
      return ms.Tensor(c)

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
print(net())
```

输出结果：

```text
[5 7 9]
```

### 使用Python原生的print打印

JIT Fallback支持在静态图模式下使用Python原生的print来打印常量，它与[Print算子](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.Print.html)打印信息的时机有所不同。Python原生print是在编译过程中触发打印（编译时阶段打印），而Print算子是需要图编译完成后，下发到设备端运行才打印（运行时阶段打印）。

为了便于理解，举例如下。tensor_sum涉及Tensor相加，即运行时阶段才能得到结果，在调用print时，实际调用的是静态图模式中的Print算子，参考[静态图语法支持](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/note/static_graph_syntax_support.html)。而np_num是由两个NumPy常量相加得到的结果，即通过JIT Fallback支持的用法，因此在调用print时，使用的是Python原生print。由于两者的打印时机不同，最终导致显示np_sum在tensor_sum之前，即通过JIT Fallback支持的Python原生print的打印结果会在Print算子之前。

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn

class Net(nn.Cell):
   def __init__(self):
      super(Net, self).__init__()

   def construct(self):
      x = ms.Tensor(np.array([1, 2, 3, 4, 5]))
      y = ms.Tensor(np.array([1, 2, 3, 4, 5]))
      tensor_sum = x + y
      print("tensor_sum: ", tensor_sum)
      x = np.array([1, 2, 3, 4, 5])
      y = np.array([1, 2, 3, 4, 5])
      np_sum = x + y
      print("np_sum: ", np_sum)
      return tensor_sum, ms.Tensor(np_sum)

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
net()
```

输出结果：

```text
np_sum: [2 4 6 8 10]
tensor_sum: (2, 4, 6, 8, 10)
```

当前不支持使用同一个print同时打印编译时期和运行时期执行的信息，例如将np_sum和tensor_sum放在同一个print中将会报错。错误的代码用例如下：

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn

class Net(nn.Cell):
   def __init__(self):
      super(Net, self).__init__()

   def construct(self, input_x, input_y):
      tensor_sum = input_x + input_y
      x = np.array([1, 2, 3, 4, 5])
      y = np.array([1, 2, 3, 4, 5])
      np_sum = x + y
      print("np_sum: ", np_sum, "tensor_sum: ", tensor_sum)
      return tensor_sum, ms.Tensor(np_sum)

ms.set_context(mode=ms.GRAPH_MODE)
x = ms.Tensor(np.array([1, 2, 3, 4, 5]))
y = ms.Tensor(np.array([1, 2, 3, 4, 5]))
net = Net()
net(x, y)
```

报错信息如下：

```text
ValueError: When using JIT Fallback to handle script 'print("np_sum: ", np_sum, "tensor_sum: ", tensor_sum)', the inputs should be constant, but found variable 'tensor_sum' to be nonconstant.
```

### 使用raise和assert

JIT Fallback支持在静态图模式下使用raise和assert。

使用raise时，要求条件语句和抛出的异常语句符合常量场景的条件，否则可能出现不可预期的结果。正确的代码用例如下：

```python
import mindspore.nn as nn
import mindspore as ms
class Net(nn.Cell):
   def __init__(self):
      super(Net, self).__init__()

   def construct(self, x):
      if x <= 0:
         raise ValueError("x should be greater than 0.")
      else:
         x += 1
      return x

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
net(-1)
```

输出结果：

```text
ValueError: x should be greater than 0.
```

同理，使用assert时，也需要符合常量场景的条件。正确的代码用例如下：

```python
import mindspore.nn as nn
import mindspore as ms

class Net(nn.Cell):
   def __init__(self):
      super(Net, self).__init__()

   def construct(self):
      x = 1
      assert 1 in [2, 3, 4]
      return x

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
net()
```

输出结果中正常出现： `AssertionError`。

### 调用Python内置函数

MindSpore在静态图模式下已经支持了一些Python内置函数，包括但不限于len、isinstance、map、zip等，详情请参考[静态图语法支持](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/note/static_graph_syntax_support.html)。通过JIT Fallback，可以在常量场景中支持更多的Python内置函数的用法。下面简单举例支持的部分Python内置函数。

#### dict()

功能：用于创建一个字典。此外 dict 还可以返回对象的有效属性列表，暂不支持自定义类。

有效输入：字典的 Key 只支持 String 类型。

代码用例如下：

```python
import mindspore as ms

@ms.jit
def func():
   a = dict()                                          # 创建空字典
   b = dict(a='a', b='b', t='t')                       # 传入关键字
   c = dict(zip(['one', 'two', 'three'], [1, 2, 3]))   # 映射函数方式来构造字典
   d = dict([('one', 1), ('two', 2), ('three', 3)])    # 可迭代对象方式来构造字典
   return a, b, c, d

a, b, c, d = func()
print("a: ",a)
print("b: ",b)
print("c: ",c)
print("d: ",d)
```

输出结果：

```text
a: {}
b: {'a': 'a', 'b': 'b', 't': 't'}
c: {'one': 1, 'two': 2, 'three': 3}  
d: {'one': 1, 'two': 2, 'three': 3}
```

#### type()

功能：输出入参的类型。

有效输入：Number、list、tuple、dict、np.array、常量Tensor。

代码用例如下：

```python
import numpy as np
import mindspore as ms

@ms.jit
def func():
   a = type(1)
   b = type(1.0)
   c = type([1, 2, 3])
   d = type((1, 2, 3))
   e = type({'a': 1, 'b': 2})
   f = type(np.array([1, 2, 3]))
   g = type(ms.Tensor([1, 2, 3]))
   return a, b, c, d, e, f, g

a, b, c, d ,e, f, g = func()
print("a: ",a)
print("b: ",b)
print("c: ",c)
print("d: ",d)
print("e: ",e)
print("f: ",f)
print("g: ",g)
```

输出结果：

```text
a: <class 'int'>
b: <class 'float'>
c: <class 'list'>
d: <class 'tuple'>
e: <class 'dict'>
f: <class 'numpy.ndarray'>
g: <class 'mindspore.common.tensor.Tensor'>
```

> 注：type作为Python的原生函数还有另外一种使用方法，即type(name, bases, dict)返回name类型的类对象，由于该用法应用场景较少，因此暂不支持。

### 支持常量场景下控制流

为了提高Python标准语法支持度，在常量场景下实现动静统一，通过JIT Fallback实现常量场景下控制流语句的使用。控制流语句是指if、for、while等流程控制语句。JIT Fallback特性已经支持在静态图模式下创建和使用Tensor，支持调用Numpy等第三方库创建使用常量以及支持部分Python内置函数。理论上，通过JIT Fallback支持的常量语法，在常量控制流场景中也支持。
代码用例如下：

```python
import numpy as np
import mindspore as ms

@ms.jit
def func():
   x = np.array(1)
   if x <= 1:
     x += 1
   return ms.Tensor(x)

res = func()
print("res: ", res)
```

输出结果如下：

```text
res: 2
```

## 使用须知

在使用JIT Fallback时，请注意以下几点：

1. 当前JIT Fallback仅支持常量场景，即要求编译期间能够确定实际值。

2. JIT Fallback对标动态图的支持能力，须在动态图语法范围内，包括但不限于数据类型等。

3. 当前常量控制流场景中暂不支持对Numpy Array数据的取下标赋值，错误的代码用例如下：

   ```python
   import numpy as np
   import mindspore as ms

   @ms.jit
   def func():
       x = np.array([1, 2, 3])
       x[0] += 1
       return ms.Tensor(x)

   res = func()
   print("res: ", res)
   ```

   报错信息如下：

   ```text
   RuntimeError: The 'setitem' operation does not support the type [External, Int64, Int64].
   ```

4. 不支持运行时(Runtime)阶段的JIT Fallback。

   JIT Fallback处理不支持的语法表达式时，将会生成相应的节点，需要在编译时阶段完成推导和执行，否则这些节点传递到运行时后会引发报错。示例代码如下，`np.add(x, y)`会生成相应节点，作为函数的返回值将会传递到运行时，出现报错。在此用例中，可以将计算后的NumPy数据类型转换成Tensor类型，即调用Tensor()方法，使得程序能够正常执行。

    ```python
    import numpy as np
    import mindspore as ms

    @ms.jit
    def test_np_add():
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        return np.add(x, y)
        # return Tensor(np.add(x, y)) # 若调用Tensor()方法传递结果，则程序将能够正常执行。

    np_add_res = test_np_add()
    ```

   报错信息如下：

    ```text
    Should not use Python object in runtime, node: ValueNode<InterpretedObject> InterpretedObject: '[2 4 6 8 10]'
    ```

   值得注意的是，在常量场景中，NumPy整型数据、浮点型数据的运算结果将转换为常量进行保存，因此其运算结果可以作为函数返回值。例如：

    ```python
    import numpy as np
    import mindspore as ms

    @ms.jit
    def test_np_add_constant():
        x = 1.0
        y = 2.0
        return np.add(x, y)

    res = test_np_add_constant()
    print("res:", res)
    ```

   输出结果如下：

   ```text
   res: 3.0
   ```

5. 通过JIT Fallback支持的NumPy第三方库，与MindSpore提供的[mindspore.numpy](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/mindspore.numpy.html)不同。

   mindspore.numpy是通过MindSpore框架的算子能力实现的，涉及运行时阶段的算子计算，无法在编译期阶段推导其结果(变量的推导结果为None)。示例代码如下，对`mnp.average(x)`的结果使用Tensor()方法，不符合常量场景的条件，将会引发报错。

    ```python
    import mindspore as ms
    import mindspore.numpy as mnp

    @ms.jit
    def test_mnp_average():
        x = mnp.array(([[1., 2.], [3., 4.]]))
        x_average = mnp.average(x)
        return ms.Tensor(x_average)

    out = test_mnp_average()
    print(out)
    ```

   报错信息如下：

   ```text
   TypeError: For 'Tensor', the type of input_data should be one of '['Tensor', 'ndarray', 'str_', 'list', 'tuple', 'float', 'int', 'bool', 'complex']', but got 'None' with type 'NoneType'.
   ```
