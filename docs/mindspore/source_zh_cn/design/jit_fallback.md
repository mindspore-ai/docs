# JIT Fallback

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/design/jit_fallback.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

MindSpore框架支持静态图模式和动态图模式两种方式。在静态图模式下，先将Python代码编译成静态计算图，然后执行静态计算图。由于语法解析的限制，用户编写程序时需要遵循MindSpore[静态图语法支持](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html)，语法使用存在约束限制。在动态图模式下，Python代码会通过Python解释器执行，用户可以使用任意Python语法。可以看到，静态图和动态图的编译流程不一致，语法约束限制也不同。关于静态图和动态图的更多介绍，请参考[静态图和动态图](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/pynative_graph.html)。

JIT Fallback是从静态图的角度出发考虑静态图和动态图的统一。通过JIT Fallback特性，静态图可以支持尽量多的动态图语法，使得静态图提供接近动态图的语法使用体验，从而实现动静统一。为了便于用户选择是否使用JIT Fallback特性的能力，提供了开关`MS_DEV_ENABLE_FALLBACK`，当前默认已经打开。如果需要关闭，可以使用命令：`export MS_DEV_ENABLE_FALLBACK=0`。

本文档主要介绍JIT Fallback的支持范围和使用须知，以便您可以更有效地使用JIT Fallback功能。

## 支持范围

当前JIT Fallback支持静态图模式的部分常量场景，包括在construct/ms_function中调用第三方库、创建及使用Tensor、调用Python的print打印等。下面对各场景进行简单举例说明。

### 支持在construct/ms_function中调用第三方库

JIT Fallback支持在construct/ms_function中调用NumPy等第三方库中的对象和方法。

代码用例如下。因为静态图模式不支持在construct/ms_function中调用numpy第三方库，用例中的`a = np.array([1, 2, 3])`和`b = np.array([4, 5, 6])`将会通过JIT Fallback使用Python解释器进行解释执行。

```python
import numpy as np
from mindspore import Tensor, ms_function

@ms_function
def np_binop():
   a = np.array([1, 2, 3])
   b = np.array([4, 5, 6])
   c = a + b
   return Tensor(c)

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

为了对比，我们可以通过关闭JIT Fallback特性的开关，来观察没有JIT Fallback特性时该用例的执行结果，即设置`export MS_DEV_ENABLE_FALLBACK=0`，用例执行结果如下：

```text
Meet a exception from Python when get the type of '<built-in function array>'
TypeError: Not support for this object with type '<class 'builtin_function_or_method'>' and value '<built-in function array>'
```

### 支持在construct/ms_function中创建和使用Tensor

JIT Fallback支持在construct/ms_function中创建和使用[Tensor](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Tensor.html)，暂不支持Tensor.asnumpy()。

代码用例如下。因为静态图模式不支持在construct/ms_function中创建Tensor对象，用例中的`tensor_num = Tensor(np.array(9))`将会通过JIT Fallback使用Python解释器进行解释执行。

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

为了对比，我们可以通过关闭JIT Fallback特性的开关，来观察没有JIT Fallback特性时该用例的执行结果，即设置`export MS_DEV_ENABLE_FALLBACK=0`，用例执行结果如下：

```text
Meet a exception from Python when get the type of '<built-in function array>'
TypeError: Not support for this object with type '<class 'builtin_function_or_method'>' and value '<built-in function array>'
```

### 支持在construct/ms_function使用print打印

在常量场景中，通过JIT Fallback特性使用Python原生的print来打印常量，与图模式中使用[print算子](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Print.html)来打印信息的时机有所不同。Python原生print是在编译过程中触发打印，而图模式调用算子打印是需要图中所有节点构图结束后下发到设备端运行才打印。

为了便于理解，举例如下。tensor_sum是由两个Tensor变量相加得到结果，需要在运行阶段才可以得到结果，即需要使用图模式中的print算子打印信息；而np_sum是由两个NumPy常量对象相加得到结果，即在编译阶段使用Python原生print能力来打印信息。导致最终显示np_sum会在tensor_sum之前，这是编译时运行方式和运行时运行方式的区别。

```python
import numpy as np
from mindspore import Tensor, ms_function

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

tensor_sum, np_sum = test_print()
```

输出结果如下:

```text
np_sum: [2 4 6 8 10]
tensor_sum: (2, 4, 6, 8, 10)
```

通过以上用例，不难理解，在同一个print中不能够同时含有编译时期和运行时期执行的信息，例如将np_sum和tensor_sum都在同一个print中打印，则会报错：

```python
import numpy as np
from mindspore import Tensor, ms_function

@ms_function
def test_print():
   x = Tensor(np.array([1, 2, 3, 4, 5]))
   y = Tensor(np.array([1, 2, 3, 4, 5]))
   tensor_sum = x + y
   x = np.array([1, 2, 3, 4, 5])
   y = np.array([1, 2, 3, 4, 5])
   np_sum = x + y
   print("np_sum: ", np_sum, "tensor_sum: ", tensor_sum)
   return tensor_sum, Tensor(np_sum)

tensor_sum, np_sum = test_print()
```

输出结果如下:

```text
TypeError: For 'Print', the type of 'input' should be one of Tensor, Int, Float, Bool, String, but got kMetaTypeExternal. The supported data types depend on the hardware that executes the operator, please refer the official api document to get more information about the data type.
```

### 支持在construct/ms_function常量场景下使用raise语句

在编译期间执行raise，则要求条件是在编译期间能够获取得到值，即常量场景。如果编译期间获取不到值，则属于变量场景。同时raise抛出的异常语句中也不能够含有变量，请在常量场景中使用raise语句，如果在变量场景下使用可能会存在不可预期的结果。例如下面例子，变量场景中入参的值在编译时期是不可知的，得到的结果是不符合预期的。

```python
import numpy as np
from mindspore import Tensor, ms_function

@ms_function
def raise_func(x):
   if x > 1:
     name = "MindSpore 1."
   else:
     name = "MindSpore 2."
   raise ValueError("I'm " + name)

raise_func(Tensor(1))
```

输出结果如下:

```text
ValueError: mindspore/ccsrc/pipeline/jit/static_analysis/prim.cc:2107 EvalPrim] I'm MindSpore 1.
```

### 支持Python的内置函数

在常量场景中，通过JIT Fallback特性可以支持Python的一些内置函数功能。

#### list()

**功能**： 将输入的对象转换为list

**有效输入**： list，tuple， dict，np.array， 常量Tensor

```python
import mindspore as ms
from mindspore import Tensor,ms_function
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

输出结果如下:

```text
a: [1, 2, 3]
b: [1, 2, 3]
c: ['a', 'b', 'c']
d: [Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 2), Tensor(shape=[], dtype=Int64, value= 3)]
```

MindSpore 支持程度：

| 类型                | 是否支持 |
| ------------------- | -------- |
| list                | √        |
| tuple               | √        |
| dict(只会转换key值) | √        |
| np.array            | √        |
| 常量Tensor          | √        |

#### tuple()

**功能**： 将输入的对象转换为tuple

**有效输入**： list，tuple，dict，np.array, 常量Tensor

```python
import mindspore as ms
from mindspore import Tensor,ms_function
@ms_function
def func():
   a = tuple((1, 2, 3))
   b = tuple([1, 2, 3])
   c = tuple({'a':1, 'b':2, 'c':3})
   d = tuple(Tensor([1, 2, 3]))
   return a, b, c ,d
a, b, c ,d = func()
print("a: ",a)
print("b: ",b)
print("c: ",c)
print("d: ",d)
```

输出结果如下：

```text
a: (1, 2, 3)
b: (1, 2, 3)
c: ('a', 'b', 'c')
d: (Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 2), Tensor(shape=[], dtype=Int64, value= 3))
```

MindSpore 支持程度：

| 类型                | 是否支持 |
| ------------------- | -------- |
| list                | √        |
| tuple               | √        |
| dict(只会转换key值) | √        |
| np.array            | √        |
| 常量Tensor          | √        |

#### round()

**功能**： round(x, n=0) 返回输入的四舍五入值

**有效输入**:

**x**: 待四舍五入的值， int, float

**n**: 表示四舍五入的小数点位数， int

```python
import mindspore as ms
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
   return a, b, c, d, e, f, g ,h
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

输出结果如下：

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

对浮点数执行round()行为可能会令人惊讶。例如，round(2.675, 2)不一定给出期望的 2.68。
这不是程序错误，这一结果是由于十进制小数实际上不能以浮点数精确表示。

MindSpore 支持程度：

| 类型(x, n) | 是否支持 |
| ---------- | -------- |
| int, int   | √        |
| float, int | √        |

#### **sum()**

**功能**： sum(x, n=0)对序列进行求和计算

**有效输入**：

**x**: 可迭代对象 list, tuple, dict, numpy.array, 常量Tensor

**n**: 指定相加的参数，如果没有设置这个值，默认为 0

```python
import mindspore as ms
from mindspore import Tensor,ms_function
import numpy as np
@ms_function
def func():
   a = sum([0,1,2])
   b = sum((0,1,2))
   c = sum({1:10, 2:20, 3:30})
   d = sum(np.array([1, 2, 3]))
   e = sum([0,1,2], 10)
   f = sum((0,1,2), 10)
   g = sum({1:10, 2:20, 3:30}, 10)
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

输出结果如下：

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

MindSpore 支持程度：

| 类型                | 是否支持 |
| ------------------- | -------- |
| list                | √        |
| tuple               | √        |
| dict(只会转换key值) | √        |
| np.array            | √        |
| 常量Tensor          | √        |

#### max(), min()

**功能**： max()输出最大值， min()输出最大值， 两者限制相同，以下均以max()为例。

**有效输入**：max(x, y, z, ...)， 其中每一项为待比较的对象。

若只有单输入则会比较单输入内的各个元素， 若存在多输入，则比较每个输入。 比较的元素必须是可以比较的， 例如：tuple和list就无法在一起比较。

输入可以为数字，list， tuple， dict，np.array, 常量Tensor。 其中dict以及np.array不支持比较，只支持其内部元素的比较， 例如：

```python
import mindspore as ms
from mindspore import Tensor,ms_function
import numpy as np
@ms_function
def func():
   a = max([0,1,2,3])
   b = max((0,1,2,3))
   c = max({1:10, 2:20, 3:3})
   d = max(np.array([1,2,3,4]))
   e = max(('a', 'b', 'c'))
   f = max((1,2,3), (1,4))
   g = max(Tensor([1, 2, 3]))
   return a, b, c , Tensor(d), e, f, g
a, b, c , d, e, f, g = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
print("f: ", f)
print("g: ", g)
```

输出结果如下：

```text
a: 3
b: 3
c: 3
d: 4
e: c
f: (1, 4)
g: 3
```

多输入与单输入的原则上是一致的，在这里不赘述。

MindSpore 支持程度：

| 类型                | 是否支持 |
| ------------------- | -------- |
| list                | √        |
| tuple               | √        |
| dict(只会转换key值) | √        |
| np.array            | √        |
| 常量Tensor          | √        |
| str                 | √        |
| Numbers(多个数字)   | √        |

#### type()

**功能**： type(x)输出x的类型

**有效输入：** 数字, list, tuple, dict, np.array, 常量Tensor

```python
import mindspore as ms
from mindspore import Tensor,ms_function
import numpy as np
@ms_function
def func():
   a = type(1)
   b = type(1.0)
   c = type([1, 2, 3])
   d = type((1, 2, 3))
   e = type({'a':1, 'b':2})
   f = type(np.array([1,2,3]))
   g = type(Tensor([1, 2, 3]))
   return a, b, c, d ,e, f, g
a, b, c, d ,e, f, g = func()
print("a: ",a)
print("b: ",b)
print("c: ",c)
print("d: ",d)
print("e: ",e)
print("f: ",f)
print("g: ",g)
```

输出结果如下：

```text
a: <class 'int'>
b: <class 'float'>
c: <class 'list'>
d: <class 'tuple'>
e: <class 'dict'>
f: <class 'numpy.ndarray'>
g: <class 'mindspore.common.tensor.Tensor'>
```

**<注>** type作为Python的原生函数还有另外一种使用方法， 因为该使用方法应用场景较少，因此暂不支持。

该方法为：type(name, bases, dict) 返回name类型的类对象。

#### abs()

**功能：**

返回一个数的绝对值。 参数可以是整数、浮点数或任何实现了 abs()的对象。 如果参数是一个复数，则返回它的模。

```python
import mindspore as ms
from mindspore import Tensor,ms_function
@ms_function
def func():
   a = abs(-45)
   b = abs(100.12)
   return a, b
a, b = func()
print("a: ",a)
print("b: {:.2f}".format(b))
```

输出结果如下：

```text
a: 45
b: 100.12
```

MindSpore 支持程度：

| 类型                                                         | 是否支持 |
| ------------------------------------------------------------ | -------- |
| int                                                          | √        |
| float                                                        | √        |
| complex[复数只能通过函数方式创建不能通过表达式创建比如 1+2j，需要改成 complex(1,2)] | √        |
| np.array                                                     | √        |
| 常量Tensor                                                   | √        |

#### all()/any()

**all()功能:**

如果 iterable 的所有元素均为真值（或可迭代对象为空）则返回 True 。 等价于：

```python
def all(iterable):
   for element in iterable:
      if not element:
         return False
    return True
```

**any功能：**

如果iterable 的任一元素为真值则返回 True。 如果可迭代对象为空，返回 False。 等价于:

```python
def any(iterable):
   for element in iterable:
      if element:
         return True
   return False
```

all():

```python
import mindspore as ms
from mindspore import ms_function
@ms_function
def func():
   a = all(['a', 'b', 'c', 'd'])  # 列表 list，元素都不为空或 0
   b = all(['a', 'b', '', 'd'])   # 列表 list，存在一个为空的元素
   c = all([0, 1, 2, 3])          # 列表 list，存在一个为 0 的元素
   d = all(('a', 'b', 'c', 'd'))  # 元组 tuple，元素都不为空或 0
   e = all(('a', 'b', '', 'd'))   # 元组 tuple，存在一个为空的元素
   f = all((0, 1, 2, 3))          # 元组 tuple，存在一个为 0 的元素
   g = all([])             # 空列表
   h = all(())             # 空元组
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

输出结果如下：

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

any():

```python
import mindspore as ms
from mindspore import ms_function
@ms_function
def func():
   a = any(['a', 'b', 'c', 'd'])# 列表 list，元素都不为空或 0
   b = any(['a', 'b', '', 'd'])# 列表 list，存在一个为空的元素**
   c = any([0, '', False])# 列表 list,元素全为 0,'',false**
   d = any(('a', 'b', 'c', 'd'))# 元组 tuple，元素都不为空或 0
   e = any(('a', 'b', '', 'd'))# 元组 tuple，存在一个为空的元素**
   f = any((0, '', False))# 元组 tuple，元素全为 0,'',false**
   g = any([])# 空列表**
   h = any(())# 空元组**
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

输出结果如下：

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

**支持类型**

| 类型       | 是否支持 |
| ---------- | -------- |
| List/Tuple | √        |
| np.array   | √        |
| Tensor     | √        |

#### bool()/float()/int()

**bool(x)** :

返回布尔值，True 或 False。

x 用标准的真值测试过程进行转换。如果 x 为 False 或省略，则返回 False；否则返回 True。 bool 类是 int 的子类。它不能再被继承。它唯一的实例就是 False 和 True。

```python
import mindspore as ms
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

输出结果如下：

```text
a: False
b: False
c: True
d: True
```

**float(x):**

从数字或字符串 x 生成的浮点数。

参数是个字符串，则应包含一个十进制数字，前面可选带上符号，也可选前后带有空白符。符号可以是 '+' 或 '-'；'+' 符号对值没有影响。参数也可以是一个代表 NaN（非数字）或正负无穷大的字符串。更确切地说，在去除前导和尾部的空白符后，输入参数必须符合以下语法：

```text
sign           ::=  "+" | "-"
infinity      ::=  "Infinity" | "inf"
nan           ::=  "nan"
numeric_value  ::=  floatnumber | infinity | nan
numeric_string  ::=  [sign] numeric_value
```

这里的 floatnumber 是指 Python 的浮点数格式。大小写没有关系，所以“inf”、“Inf”、“INFINITY”、“iNfINity”都可接受为正无穷的拼写形式。

另一方面，如果实参是整数或浮点数，则返回具有相同值（在 Python 浮点精度范围内）的浮点数。如果实参在 Python 浮点精度范围外，则会触发 OverflowError。

```python
import mindspore as ms
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

输出结果如下：

```text
a: 1.0
b: 112.0
c: -123.6
d: 123.0
```

**int()**:

返回一个基于数字或字符串 x 构造的整数对象，或者在未给出参数时返回 0。 对于浮点数，它将向零舍入。

如果 x 不是数字，或者有 base 参数，x 必须是字符串、bytes、表示进制为 base 的整数字面值的 bytearray 实例。该文字前可以有 + 或 - （中间不能有空格），前后可以有空格。一个进制为 n 的数字包含 0 到 n-1 的数，其中 a 到 z （或 A 到 Z ）表示 10 到 35。默认的 base 为 10 ，允许的进制有 0、2-36。2、8、16 进制的数字可以在代码中用 0b/0B、 0o/0O、 0x/0X 前缀来表示。进制为 0 将安照代码的字面量来精确解释，最后的结果会是 2、8、10、16 进制中的一个。所以 int('010', 0) 是非法的，但 int('010') 和 int('010', 8) 是合法的。

```python
import mindspore as ms
from mindspore import ms_function
@ms_function
def func():
   a = int(3)
   b = int(3.6)
   c = int('12',16)# 如果是带参数 base 的话，12 要以字符串的形式进行输入，12 为 16 进制
   d = int('0xa',16)
   e = int('10',8)
   return a, b ,c ,d ,e
a, b, c, d, e = func()
print("a: ",a)
print("b: ",b)
print("c: ",c)
print("d: ",d)
print("e: ",e)
```

输出结果如下：

```text
a: 3
b: 3
c: 18
d: 10
e: 8
```

MindSpore支持类型：

| 类型       | 是否支持                           |
| ---------- | ---------------------------------- |
| int        | √                                  |
| np.array   | √                                  |
| List/Tuple | bool ()支持，float()&&int() 不支持 |
| Tensor     | √                                  |

#### dict()

dict() 函数：用于创建一个字典。此外 dict 还可以返回对象的有效属性列表，由于涉及一些自定义类别，MindSpore 暂时不支持。

MindSpore 当前 dict 支支持 String 为 key，不支持其他类型为 key。

```python
import mindspore as ms
from mindspore import Tensor,ms_function
@ms_function
def func():
   a = dict()                        # 创建空字典
   b = dict(a='a', b='b', t='t')     # 传入关键字
   c = dict(zip(['one', 'two', 'three'], [1, 2, 3]))   # 映射函数方式来构造字典
   d = dict([('one', 1), ('two', 2), ('three', 3)])    # 可迭代对象方式来构造字典
   return a, b, c ,d
a, b, c ,d = func()
print("a: ",a)
print("b: ",b)
print("c: ",c)
print("d: ",d)
```

输出结果如下：

```text
a: {}
b: {'a': 'a', 'b': 'b', 't': 't'}
c: {'three': 3, 'two': 2, 'one': 1}  
d: {'three': 3, 'two': 2, 'one': 1}
```

## 使用须知

在使用JIT Fallback时，请注意以下几点：

1. 当前JIT Fallback仅支持常量场景，即值明确且保持不变，不以参数传入的场景。

2. JIT Fallback对标动态图的支持能力，须在动态图语法范围内，包括但不限于数据类型等。

3. 运行时(Runtime)阶段的JIT Fallback暂不支持。当前运行时不支持解释节点，如果JIT Fallback引入的解释节点传递到运行时，将会出现报错。示例代码如下，`np.add(x, y)`是静态图模式下不支持的语法，将会生成解释节点，作为函数的返回值传递到运行时，从而引发报错。

    ```python
    import numpy as np
    from mindspore import Tensor, ms_function

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

4. 当前有限支持控制流场景，将逐步在后续版本中支持。

5. 当前暂不支持自定义Class的attr/method，将逐步在后续版本中支持。

6. MindSpore提供的NumPy中的方法是由框架的算子能力实现，并不是通过JIT Fallback来支持的，在使用时需要注意该场景。使用Python解释器推导不出MindSpore提供的NumPy中的average方法结果，得到的值为None。例如下面的用例将报错。

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
   input_data and init can not be None at the same time.
   ```

7. 在静态图模式下，对于NumPy具有返回值的方法，需要使用变量来保存其结果。如果没有变量保存，当前不支持该语法，会在后续版本中支持。

    ```python
    import numpy as np
    from mindspore import Tensor, ms_function

    @ms_function
    def test_np_vdot():
        x = np.array([[1, 2], [3, 4]])
        y = x.T
        np.vdot(x, y)
        return Tensor(y)

    res = test_np_vdot()
    ```

   输出结果如下:

    ```text
    TypeError: module, class, method, function, traceback, frame, or code object was expected, got builtin_function_or_method.
    ```

8. 暂不支持在解释执行的语句中调用`self`的属性和方法，将逐步在后续版本中支持。

    ```python
    import numpy as np
    import mindspore.nn as nn
    from mindspore import Tensor

    class Network(nn.Cell):
        def __init__(self):
            super(Network, self).__init__()
            self.value = 1

        def construct(self):
            x = np.array([1, 2, 3])
            y = np.array([3, 4, 5])
            z = self.fn(x, y)
            out = Tensor(z)
            return out

        def fn(self, x, y):
            return x + y

    net = Network()
    out = net()
    ```

   输出结果如下：

    ```
    RuntimeError: The 'add' operation does not support the type [kMetaTypeExternal, kMetaTypeExternal]
    ```
