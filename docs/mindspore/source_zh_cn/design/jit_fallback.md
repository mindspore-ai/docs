# JIT Fallback

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/design/jit_fallback.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

MindSpore框架支持静态图模式和动态图模式两种方式。在静态图模式下，先将Python代码编译成静态计算图，然后执行静态计算图。由于语法解析的限制，用户编写程序时需要遵循MindSpore[静态图语法支持](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html)，语法使用存在约束限制。在动态图模式下，Python代码会通过Python解释器执行，用户可以使用任意Python语法。可以看到，静态图和动态图的编译流程不一致，语法约束限制也不同。关于静态图和动态图的更多介绍，请参考[静态图和动态图](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/pynative_graph.html)。

JIT Fallback是从静态图的角度出发考虑静态图和动态图的统一。通过JIT Fallback特性，静态图可以支持尽量多的动态图语法，使得静态图提供接近动态图的语法使用体验，从而实现动静统一。为了便于用户选择是否使用JIT Fallback特性的能力，提供了开关`MS_DEV_ENABLE_FALLBACK`，当前默认已经打开。如果需要关闭，可以使用命令：`export MS_DEV_ENABLE_FALLBACK=0`。

本文档主要介绍JIT Fallback的支持范围和使用须知，以便您可以更有效地使用JIT Fallback功能。

## 支持范围

当前JIT Fallback特性应用于常量场景，即要求在编译期间能够确定实际值。JIT Fallback特性还在持续完善中，下面列举出当前通过该特性已经支持的静态图编译语法。

### 创建和使用Tensor

JIT Fallback支持在静态图模式下创建和使用[Tensor](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Tensor.html)，暂不支持Tensor.asnumpy()。

代码用例如下，用例中的`Tensor(1, dtype=mstype.int32)`是通过JIT Fallback支持的。

```python
import mindspore.nn as nn
from mindspore import set_context, GRAPH_MODE, Tensor
from mindspore import dtype as mstype


class Net(nn.Cell):
   def __init__(self):
      super(Net, self).__init__()

   def construct(self):
      return Tensor(1, dtype=mstype.int32)

set_context(mode=GRAPH_MODE)
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
import mindspore.nn as nn
from mindspore import set_context, GRAPH_MODE, Tensor

class Net(nn.Cell):
   def __init__(self):
      super(Net, self).__init__()

   def construct(self):
      a = np.array([1, 2, 3])
      b = np.array([4, 5, 6])
      c = a + b
      return Tensor(c)

set_context(mode=GRAPH_MODE)
net = Net()
print(net())
```

输出结果：

```text
[5 7 9]
```

### 使用Python原生的print打印

JIT Fallback支持在静态图模式下使用Python原生的print来打印常量，它与[Print算子](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Print.html)打印信息的时机有所不同。Python原生print是在编译过程中触发打印（编译时阶段打印），而Print算子是需要图编译完成后，下发到设备端运行才打印（运行时阶段打印）。

为了便于理解，举例如下。tensor_sum涉及Tensor相加，即运行时阶段才能得到结果，在调用print时，实际调用的是静态图模式中的Print算子，参考[静态图语法支持](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html)。而np_num是由两个NumPy常量相加得到的结果，即通过JIT Fallback支持的用法，因此在调用print时，使用的是Python原生print。由于两者的打印时机不同，最终导致显示np_sum在tensor_sum之前，即通过JIT Fallback支持的Python原生print的打印结果会在Print算子之前。

```python
import numpy as np
import mindspore.nn as nn
from mindspore import set_context, GRAPH_MODE, Tensor

class Net(nn.Cell):
   def __init__(self):
      super(Net, self).__init__()

   def construct(self):
      x = Tensor(np.array([1, 2, 3, 4, 5]))
      y = Tensor(np.array([1, 2, 3, 4, 5]))
      tensor_sum = x + y
      print("tensor_sum: ", tensor_sum)
      x = np.array([1, 2, 3, 4, 5])
      y = np.array([1, 2, 3, 4, 5])
      np_sum = x + y
      print("np_sum: ", np_sum)
      return tensor_sum, Tensor(np_sum)

set_context(mode=GRAPH_MODE)
net = Net()
net()
```

输出结果：

```text
np_sum: [2 4 6 8 10]
tensor_sum: (2, 4, 6, 8, 10)
```

当前不支持使用同一个print同时打印编译时期和运行时期执行的信息。例如将np_sum和tensor_sum放在同一个print中，将会报错：

```python
import numpy as np
import mindspore.nn as nn
from mindspore import set_context, GRAPH_MODE, Tensor

class Net(nn.Cell):
   def __init__(self):
      super(Net, self).__init__()

   def construct(self):
      x = Tensor(np.array([1, 2, 3, 4, 5]))
      y = Tensor(np.array([1, 2, 3, 4, 5]))
      tensor_sum = x + y
      x = np.array([1, 2, 3, 4, 5])
      y = np.array([1, 2, 3, 4, 5])
      np_sum = x + y
      print("np_sum: ", np_sum, "tensor_sum: ", tensor_sum)
      return tensor_sum, Tensor(np_sum)

set_context(mode=GRAPH_MODE)
net = Net()
net()
```

输出结果：

```text
TypeError: For 'Print', the type of 'input' should be one of Tensor, Int, Float, Bool, String, but got kMetaTypeExternal.
```

### 使用raise和assert

JIT Fallback支持在静态图模式下使用raise和assert。

使用raise时，要求条件语句和抛出的异常语句符合常量场景的条件，否则可能出现不可预期的结果。

```python
import mindspore.nn as nn
from mindspore import set_context, GRAPH_MODE

class Net(nn.Cell):
   def __init__(self):
      super(Net, self).__init__()

   def construct(self, x):
      if x <= 0:
         raise ValueError("x should be greater than 0.")
      else:
         x += 1
      return x

set_context(mode=GRAPH_MODE)
net = Net()
net(-1)
```

输出结果：

```text
ValueError: x should be greater than 0.
```

同理，使用assert时，也需要符合常量场景的条件。

```python
import mindspore.nn as nn
from mindspore import set_context, GRAPH_MODE

class Net(nn.Cell):
   def __init__(self):
      super(Net, self).__init__()

   def construct(self):
      x = 1
      assert 1 in [2, 3, 4]
      return x

set_context(mode=GRAPH_MODE)
net = Net()
net()
```

输出结果中出现： `AssertionError`。

### 调用Python内置函数

MindSpore在静态图模式下已经支持了一些Python内置函数，包括len、isinstance、map、zip等，详情请参考[静态图语法支持](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html)。通过JIT Fallback，可以在常量场景中支持更多的Python内置函数的用法。

#### abs()

功能：返回一个数的绝对值。参数可以是整数、浮点数或任何实现了 abs() 的对象。如果参数是一个复数，则返回它的模。

有效输入：int、float、complex、numpy.array、常量Tensor。其中，对于复数类型complex，只能通过函数方式创建如complex(1,2)，不支持通过表达式创建如 1+2j。

代码用例如下：

```python
from mindspore import ms_function

@ms_function
def func():
   a = abs(-45)
   b = abs(100.12)
   return a, b

a, b = func()
print("a: ",a)
print("b: {:.2f}".format(b))
```

输出结果：

```text
a: 45
b: 100.12
```

#### all()

功能：如果 all 的参数即 iterable 的所有元素均为真值，或 all 的参数缺省，返回 True。等价于：

```python
def all(iterable):
   for element in iterable:
      if not element:
         return False
    return True
```

有效输入：list、tuple、numpy.array、常量Tensor。

代码用例如下：

```python
from mindspore import ms_function

@ms_function
def func():
   a = all(['a', 'b', 'c', 'd'])  # 列表 list，元素都不为空或 0
   b = all(['a', 'b', '', 'd'])   # 列表 list，存在一个为空的元素
   c = all([0, 1, 2, 3])          # 列表 list，存在一个为 0 的元素
   d = all(('a', 'b', 'c', 'd'))  # 元组 tuple，元素都不为空或 0
   e = all(('a', 'b', '', 'd'))   # 元组 tuple，存在一个为空的元素
   f = all((0, 1, 2, 3))          # 元组 tuple，存在一个为 0 的元素
   g = all([])                    # 空列表
   h = all(())                    # 空元组
   return a, b, c, d, e, f, g, h

a, b, c, d, e, f, g, h = func()
print("a: ",a)
print("b: ",b)
print("c: ",c)
print("d: ",d)
print("e: ",e)
print("f: ",f)
print("g: ",g)
print("h: ",h)
```

输出结果：

```text
a: True
b: False
c: False
d: True
e: False
f: False
g: True
h: True
```

#### any()

功能：如果 any 的参数即 iterable 的任一元素为真值，返回 True。如果 any 的参数缺省，返回 False。等价于：

```python
def any(iterable):
   for element in iterable:
      if element:
         return True
   return False
```

有效输入：list、tuple、numpy.array、常量Tensor。

代码用例如下：

```python
from mindspore import ms_function

@ms_function
def func():
   a = any(['a', 'b', 'c', 'd'])  # 列表 list，元素都不为空或 0
   b = any(['a', 'b', '', 'd'])   # 列表 list，存在一个为空的元素
   c = any([0, '', False])        # 列表 list，元素为 0,'',false
   d = any(('a', 'b', 'c', 'd'))  # 元组 tuple，元素都不为空或 0
   e = any(('a', 'b', '', 'd'))   # 元组 tuple，存在一个为空的元素
   f = any((0, '', False))        # 元组 tuple，元素为 0,'',false
   g = any([])                    # 空列表
   h = any(())                    # 空元组
   return a, b, c, d, e, f, g, h

a, b, c, d, e, f, g, h = func()
print("a: ",a)
print("b: ",b)
print("c: ",c)
print("d: ",d)
print("e: ",e)
print("f: ",f)
print("g: ",g)
print("h: ",h)
```

输出结果：

```text
a: True
b: True
c: False
d: True
e: True
f: False
g: False
h: False
```

#### bool()

功能：返回布尔值，即 True或 False。假定参数为 x, x 是用标准的真值测试过程进行转换的。如果 x 为 False 或缺省，返回 False，否则返回 True。

有效输入：int、list、tuple、numpy.array、常量Tensor。

代码用例如下：

```python
from mindspore import ms_function

@ms_function
def func():
   a = bool()
   b = bool(0)
   c = bool(1)
   d = bool(2)
   return a, b, c, d

a, b, c, d = func()
print("a: ",a)
print("b: ",b)
print("c: ",c)
print("d: ",d)
```

输出结果：

```text
a: False
b: False
c: True
d: True
```

#### dict()

功能：用于创建一个字典。此外 dict 还可以返回对象的有效属性列表，暂不支持自定义类。

有效输入：字典的 Key 只支持 String 类型。

代码用例如下：

```python
from mindspore import ms_function

@ms_function
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

#### float()

功能：从数字或字符串 x 生成的浮点数。

有效输入：int、float、bool、str、numpy.array、常量Tensor。

对于字符串类型的参数，在去除前导和尾部的空白符后，输入参数必须符合以下语法：

```text
sign           ::=  "+" | "-"
infinity      ::=  "Infinity" | "inf"
nan           ::=  "nan"
numeric_value  ::=  floatnumber | infinity | nan
numeric_string  ::=  [sign] numeric_value
```

这里的 floatnumber 是指 Python 的浮点数格式，忽略大小写，即“inf”、“Inf”、“INFINITY”、“iNfINity”等都会被认定为正无穷的拼写形式。

对于整数或浮点数的参数，返回具有相同值（在 Python 浮点精度范围内）的浮点数，如果超出浮点精度范围，将会触发 OverflowError。

代码用例如下：

```python
from mindspore import ms_function

@ms_function
def func():
   a = float(1)
   b = float(112)
   c = float(-123.6)
   d = float('123')
   return a, b, c, d

a, b, c, d = func()
print("a: ",a)
print("b: ",b)
print("c: ",c)
print("d: ",d)
```

输出结果：

```text
a: 1.0
b: 112.0
c: -123.6
d: 123.0
```

#### int()

功能：返回一个基于数字或字符串构造的整数对象，如果参数缺省，返回0。对于浮点数，它将向零舍入。

有效输入：int、float、bool、str、numpy.array、常量Tensor。

如果 x 不是数字，或者带有 base 参数，则 x 必须是字符串、bytes，或者进制为 base 的整数字面值的 bytearray 实例。默认 base 为10即十进制，允许的进制有0、2-36。2、8、16 进制的数字可以在代码中用 0b/0B、 0o/0O、 0x/0X 前缀来表示。进制为 0 将按照代码的字面量来精确解释，最后的结果会是 2、8、10、16 进制中的一个。因此，int('010', 0) 是非法的，而 int('010') 和 int('010', 8) 是合法的。

代码用例如下：

```python
from mindspore import ms_function

@ms_function
def func():
   a = int(3)
   b = int(3.6)
   c = int('12',16)               # 如果是带参数 base 的话，12 要以字符串的形式进行输入，12 为 16 进制
   d = int('0xa',16)
   e = int('10',8)
   return a, b, c, d, e

a, b, c, d, e = func()
print("a: ",a)
print("b: ",b)
print("c: ",c)
print("d: ",d)
print("e: ",e)
```

输出结果：

```text
a: 3
b: 3
c: 18
d: 10
e: 8
```

#### list()

功能：将输入的对象转换为list。

有效输入：list、tuple、dict(只转换key值)、np.array、常量Tensor。

代码用例如下：

```python
from mindspore import Tensor, ms_function

@ms_function
def func():
   a = list((1, 2, 3))
   b = list([1, 2, 3])
   c = list({'a':1, 'b':2, 'c':3})
   d = list(Tensor([1, 2, 3]))
   return a, b, c, d
a, b, c, d = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
```

输出结果:

```text
a: [1, 2, 3]
b: [1, 2, 3]
c: ['a', 'b', 'c']
d: [Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 2), Tensor(shape=[], dtype=Int64, value= 3)]
```

#### max()

功能：输出最大值。若只有单个输入，则会比较单个输入内的各个元素，若存在多个输入，则比较每个输入。有些类型不支持比较，无法使用max()，比如tuple无法和list进行比较，故不支持 max(tuple, list)。

有效输入：Numbers(多个数字)、list、tuple、dict(只转换key值)、numpy.array、常量Tensor。

代码用例如下：

```python
import numpy as np
from mindspore import Tensor, ms_function

@ms_function
def func():
   a = max([0, 1, 2, 3])
   b = max((0, 1, 2, 3))
   c = max({1: 10, 2: 20, 3: 3})
   d = max(np.array([1, 2, 3, 4]))
   e = max(('a', 'b', 'c'))
   f = max((1, 2, 3), (1, 4))
   g = max(Tensor([1, 2, 3]))
   return a, b, c, Tensor(d), e, f, g

a, b, c, d, e, f, g = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
print("f: ", f)
print("g: ", g)
```

输出结果：

```text
a: 3
b: 3
c: 3
d: 4
e: c
f: (1, 4)
g: 3
```

#### min()

功能：输出最小值。若只有单个输入，则会比较单个输入内的各个元素，若存在多个输入，则比较每个输入。有些类型不支持比较，无法使用min()，比如tuple无法和list进行比较，故不支持 min(tuple, list)。

有效输入：Numbers(多个数字)、list、tuple、dict(只转换key值)、numpy.array、常量Tensor。

代码用例如下：

```python
import numpy as np
from mindspore import Tensor, ms_function

@ms_function
def func():
   a = min([0, 1, 2, 3])
   b = min((0, 1, 2, 3))
   c = min({1: 10, 2: 20, 3: 3})
   d = min(np.array([1, 2, 3, 4]))
   e = min(('a', 'b', 'c'))
   f = min((1, 2, 3), (1, 4))
   g = min(Tensor([1, 2, 3]))
   return a, b, c, Tensor(d), e, f, g

a, b, c, d, e, f, g = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
print("f: ", f)
print("g: ", g)
```

输出结果：

```text
a: 0
b: 0
c: 1
d: 1
e: a
f: (1, 2, 3)
g: 1
```

#### round()

功能：round(x, n=0) 返回输入的四舍五入值。

有效输入：x 表示待四舍五入的值，支持 int、float，n 表示四舍五入的小数点位数，支持 int。

对浮点数执行round()可能不符合预期。例如，round(2.675, 2)不一定得到期望的2.68。
这不是程序错误，这一结果是由于十进制小数实际上不能以浮点数精确表示。

代码用例如下：

```python
from mindspore import ms_function

@ms_function
def func():
   a = round(10)
   b = round(10.123)
   c = round(10.567)
   d = round(10, 0)
   e = round(10.72, -1)
   f = round(17.12, -1)
   g = round(10.17, 1)
   h = round(10.12, 1)
   return a, b, c, d, e, f, g, h

a, b, c, d, e, f, g, h = func()
print("a: ",a)
print("b: ",b)
print("c: ",c)
print("d: ",d)
print("e: {:.2f}".format(e))
print("f: {:.2f}".format(f))
print("g: {:.2f}".format(g))
print("h: {:.2f}".format(h))
```

输出结果：

```text
a: 10
b: 10
c: 11
d: 10
e: 10.00
f: 20.00
g: 10.20
h: 10.10
```

#### **sum()**

功能：sum(x, n=0) 对序列进行求和计算。

有效输入：x 表示可迭代对象，支持list、tuple、dict(只会转换key值)、numpy.array、常量Tensor。n 表示指定相加的参数，缺省值为0。

代码用例如下：

```python
import numpy as np
from mindspore import Tensor, ms_function

@ms_function
def func():
   a = sum([0, 1, 2])
   b = sum((0, 1, 2))
   c = sum({1: 10, 2: 20, 3: 30})
   d = sum(np.array([1, 2, 3]))
   e = sum([0, 1, 2], 10)
   f = sum((0, 1, 2), 10)
   g = sum({1: 10, 2: 20, 3: 30}, 10)
   h = sum(Tensor([1, 2, 3]), 10)
   return a, b, c, Tensor(d), e, f, g, h

a, b, c, d, e, f, g, h = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
print("f: ", f)
print("g: ", g)
print("h: ", h)
```

输出结果：

```text
a: 3
b: 3
c: 6
d: 6
e: 13
f: 13
g: 16
h: 16
```

#### tuple()

功能：将输入的对象转换为tuple。

有效输入：list、tuple、dict(只转换key值)、numpy.array、常量Tensor。

代码用例如下：

```python
from mindspore import Tensor,ms_function

@ms_function
def func():
   a = tuple((1, 2, 3))
   b = tuple([1, 2, 3])
   c = tuple({'a': 1, 'b': 2, 'c': 3})
   d = tuple(Tensor([1, 2, 3]))
   return a, b, c ,d

a, b, c ,d = func()
print("a: ",a)
print("b: ",b)
print("c: ",c)
print("d: ",d)
```

输出结果：

```text
a: (1, 2, 3)
b: (1, 2, 3)
c: ('a', 'b', 'c')
d: (Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 2), Tensor(shape=[], dtype=Int64, value= 3))
```

#### type()

功能：输出入参的类型。

有效输入：Number、list、tuple、dict、np.array、常量Tensor。

代码用例如下：

```python
import numpy as np
from mindspore import Tensor, ms_function

@ms_function
def func():
   a = type(1)
   b = type(1.0)
   c = type([1, 2, 3])
   d = type((1, 2, 3))
   e = type({'a': 1, 'b': 2})
   f = type(np.array([1, 2, 3]))
   g = type(Tensor([1, 2, 3]))
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

> 注： type作为Python的原生函数还有另外一种使用方法，即type(name, bases, dict)返回name类型的类对象，由于该用法应用场景较少，因此暂不支持。

## 使用须知

在使用JIT Fallback时，请注意以下几点：

1. 当前JIT Fallback仅支持常量场景，即要求编译期间能够确定实际值。

2. JIT Fallback对标动态图的支持能力，须在动态图语法范围内，包括但不限于数据类型等。

3. 当前有限支持控制流场景，将逐步在后续版本中支持。

4. 不支持运行时(Runtime)阶段的JIT Fallback。

   JIT Fallback处理不支持的语法表达式时，会生成相应的节点，称之为解释节点。这些不支持的语法表达式需要在编译时阶段完成解释执行，否则解释节点将会传递到运行时，从而引发报错。示例代码如下，`np.add(x, y)`会生成相应的解释节点，作为函数的返回值将会传递到运行时，出现报错。在此用例中，可以将计算后的NumPy数据类型转换成Tensor类型，即调用Tensor()方法，使得程序能够正常执行。

    ```python
    import numpy as np
    from mindspore import ms_function

    @ms_function
    def test_np_add():
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        return np.add(x, y)

    np_add_res = test_np_add()
    ```

   输出结果如下:

    ```text
    Should not use Python object in runtime, node: ValueNode<InterpretedObject> InterpretedObject: '[2 4 6 8 10]'
    ```

5. 通过JIT Fallback支持的NumPy第三方库，与MindSpore提供的[mindspore.numpy](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.numpy.html)不同。

   mindspore.numpy是通过MindSpore框架的算子能力实现的，涉及运行时阶段的算子计算，无法在编译期阶段推导其结果(变量的推导结果为None)。示例代码如下，对`mnp.average(x)`的结果使用Tensor()方法，不符合常量场景的条件，将会引发报错。

    ```python
    import mindspore.numpy as mnp
    from mindspore import Tensor, ms_function

    @ms_function
    def test_mnp_average():
        x = mnp.array(([[1., 2.], [3., 4.]]))
        x_average = mnp.average(x)
        return Tensor(x_average)

    out = test_mnp_average()
    print(out)
    ```

   输出结果如下:

   ```text
   TypeError: For 'Tensor', the type of input_data should be one of '['Tensor', 'ndarray', 'str_', 'list', 'tuple', 'float', 'int', 'bool', 'complex']', but got 'None' with type 'NoneType'.
   ```
