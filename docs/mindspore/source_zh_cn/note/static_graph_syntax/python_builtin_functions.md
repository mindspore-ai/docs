# 静态图语法——Python内置函数

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/static_graph_syntax/python_builtin_functions.md)

当前静态图模式支持的Python内置函数包括：`int`、`float`、`bool`、`str`、`tuple`、`list`、`dict`、`getattr`、`hasattr`、`len`、`isinstance`、`all`、`any`、`round`、`max`、`min`、`sum`、`abs`、`map`、`zip`、`range`、`enumerate`、`super`、`pow`、`print`、`filter`、`type`。图模式下内置函数的使用方法与对应的Python内置函数类似。

## int

功能：返回一个基于数字或字符串构造的整数对象。

调用：`int(x=0, base=10)`，默认转换成十进制。

入参：

- `x` -- 需要被转换为整数的对象，支持类型为`int`、`float`、`bool`、`str`、`Tensor`以及第三方对象（例如`numpy.ndarray`）。

- `base` -- 待转换进制，只有在`x`为常量`str`的时候，才可以设置该输入。

返回值：转换后的整数值。

代码用例如下：

```python
import mindspore as ms

@ms.jit
def func(x):
   a = int(3)
   b = int(3.6)
   c = int('12', 16)
   d = int('0xa', 16)
   e = int('10', 8)
   f = int(x)
   return a, b, c, d, e, f

x = ms.Tensor([-1.0], ms.float32)
a, b, c, d, e, f = func(x)
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
print("f: ", f)
```

输出结果：

```text
a: 3
b: 3
c: 18
d: 10
e: 8
f: -1
```

## float

功能：返回一个基于数字或字符串构造的浮点数对象。

调用：`float(x=0)`。

入参：`x` -- 需要被转换为浮点数的对象，支持类型为`int`、`float`、`bool`、`str`、`Tensor`以及第三方对象（例如`numpy.ndarray`）。

返回值：转换后的浮点数值。

代码用例如下：

```python
import mindspore as ms

@ms.jit
def func(x):
   a = float(1)
   b = float(112)
   c = float(-123.6)
   d = float('123')
   e = float(x.asnumpy())
   return a, b, c, d, e

x = ms.Tensor([-1], ms.int32)
a, b, c, d, e = func(x)
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
```

输出结果：

```text
a: 1.0
b: 112.0
c: -123.6
d: 123.0
e: -1.0
```

## bool

功能：返回一个基于输入构造的布尔值的对象。

调用：`bool(x=false)`。

入参：`x` -- 需要被转换为布尔值的对象，支持类型为`int`、`float`、`bool`、`str`、`list`、 `tuple`、 `dict`、`Tensor`以及第三方对象（例如`numpy.ndarray`）。

返回值：转换后的布尔值。

代码用例如下：

```python
import mindspore as ms

@ms.jit
def func():
   a = bool()
   b = bool(0)
   c = bool("abc")
   d = bool([1, 2, 3, 4])
   e = bool(ms.Tensor([10]).asnumpy())
   return a, b, c, d, e

a, b, c, d, e = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
```

输出结果：

```text
a: False
b: False
c: True
d: True
e: True
```

## str

功能：返回一个基于输入构造的字符串的对象。

调用：`str(x='')`。

入参：`x` -- 需要被转换为字符串的对象，支持类型为`int`、`float`、`bool`、`str`、`list`、 `tuple`、 `dict`、`Tensor`以及第三方对象（例如`numpy.ndarray`）。

返回值：输入`x`转换后的字符串。

代码用例如下：

```python
import numpy as np
import mindspore as ms

@ms.jit
def func():
   a = str()
   b = str(0)
   c = str([1, 2, 3, 4])
   d = str(ms.Tensor([10]))
   e = str(np.array([1, 2, 3, 4]))
   return a, b, c, d, e

a, b, c, d, e = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
```

输出结果：

```text
a:                                             # a 为空字符串
b: 0
c: [1, 2, 3, 4]
d: Tensor(shape=[1], dtype=Int64, value=[10])
e: [1 2 3 4]
```

## tuple

功能：返回一个基于输入构造的元组。

调用：`tuple(x=())`。

入参：`x` -- 需要被转换为元组的对象，支持类型为`list`、 `tuple`、 `dict`、`Tensor`以及第三方对象（例如`numpy.ndarray`）。

返回值：按照`x`的第零纬度拆分得到的元组。

代码用例如下：

```python
import numpy as np
import mindspore as ms

@ms.jit
def func():
   a = tuple((1, 2, 3))
   b = tuple(np.array([1, 2, 3]))
   c = tuple({'a': 1, 'b': 2, 'c': 3})
   d = tuple(ms.Tensor([1, 2, 3]))
   return a, b, c, d

a, b, c, d = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
```

输出结果：

```text
a: (1, 2, 3)
b: (1, 2, 3)
c: ('a', 'b', 'c')
d: (Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 2), Tensor(shape=[], dtype=Int64, value= 3))
```

## list

功能：返回一个基于输入构造的列表。

调用：`list(x=())`。

入参：`x` -- 需要被转换为列表的对象，支持类型为`list`、 `tuple`、 `dict`、`Tensor`以及第三方对象（例如`numpy.ndarray`）。

返回值：按照`x`的第零纬度拆分得到的列表。

代码用例如下：

```python
import numpy as np
import mindspore as ms

@ms.jit
def func():
   a = list((1, 2, 3))
   b = list(np.array([1, 2, 3]))
   c = list({'a':1, 'b':2, 'c':3})
   d = list(ms.Tensor([1, 2, 3]))
   return a, b, c, d
a_t, b_t, c_t, d_t = func()
print("a_t: ", a_t)
print("b_t: ", b_t)
print("c_t: ", c_t)
print("d_t: ", d_t)
```

输出结果:

```text
a_t: [1, 2, 3]
b_t: [1, 2, 3]
c_t: ['a', 'b', 'c']
d_t: [Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 2), Tensor(shape=[], dtype=Int64, value= 3)]
```

## dict

功能：用于创建一个字典。

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
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
```

```text
a: {}
b: {'a': 'a', 'b': 'b', 't': 't'}
c: {'one': 1, 'two': 2, 'three': 3}
d: {'one': 1, 'two': 2, 'three': 3}
```

## getattr

功能：获取对象的属性。

调用：`getattr(x, attr, default)`。

入参：

- `x` -- 需要被获取属性的对象，可以为任意的图模式支持类型；在JIT语法支持级别选项为`Lax`时，也支持第三方库类型。

- `attr` -- 需要获取的属性，需要为`str`。

- `default` -- 可选参数。若`x`没有`attr`，则返回`default`，可以为任意的图模式支持类型；在JIT语法支持级别选项为`Lax`时，也支持第三方库类型。若未输入`default`，且`x`没有属性`attr`，则会抛出AttributeError。

返回值：目标属性或者`default`。

代码用例如下：

```python
import numpy as np
import mindspore as ms

@ms.jit_class
class MSClass1:
  def __init__(self):
    self.num0 = 0

ms_obj = MSClass1()

@ms.jit
def func(x):
  a = getattr(ms_obj, 'num0')
  b = getattr(ms_obj, 'num1', 2)
  c = getattr(x.asnumpy(), "shape", np.array([0, 1, 2, 3, 4]))
  return a, b, c

x = ms.Tensor([-1.0], ms.float32)
a, b, c = func(x)
print("a: ", a)
print("b: ", b)
print("c: ", c)
```

输出结果:

```text
a:  0
b:  2
c:  (1,)
```

在静态图模式下对象的属性可能会和动态图模式下有区别，建议使用`default`输入，或者在使用`getattr`前先使用`hasattr`进行校验。

其中`getattr(x.asnumpy(), "shape", np.array([0, 1, 2, 3, 4]))`属于高阶用法，更多介绍可见[扩展语法（LAX级别）](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html#扩展语法lax级别)章节。

## hasattr

功能：判断对象是否具有该属性。

调用：`hasattr(x, attr)`。

入参：

- `x` -- 需要被判断是否具有某属性的对象，可以为任意的图模式支持类型；在JIT语法支持级别选项为`Lax`时，也支持第三方库类型。

- `attr` -- 属性名，需要为`str`。

返回值：布尔值，表示是否具有该属性。

代码用例如下：

```python
import numpy as np
import mindspore as ms
from mindspore import Tensor

@ms.jit_class
class MSClass1:
  def __init__(self):
    self.num0 = 0

ms_obj = MSClass1()

@ms.jit
def func():
   a = hasattr(ms_obj, 'num0')
   b = hasattr(ms_obj, 'num1')
   c = hasattr(Tensor(np.array([1, 2, 3, 4])).asnumpy(), "__len__")
   return a, b, c

a, b, c = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
```

输出结果:

```text
a: True
b: False
c: True
```

其中`hasattr(Tensor(np.array([1, 2, 3, 4])).asnumpy(), "__len__")`属于高阶用法，更多介绍可见[扩展语法（LAX级别）](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html#扩展语法lax级别)章节。

## len

功能：获取对象（字符串或者其他可迭代对象）的长度。

调用：`len(sequence)`。

入参：`sequence` -- `Tuple`、`List`、`Dictionary`、`Tensor`、`String`以及第三方对象（例如numpy.ndarray）。

返回值：序列的长度，类型为`int`。当入参是`Tensor`时，返回的是`Tensor`第零维的长度。

示例如下：

```python
import numpy as np
import mindspore as ms

z = ms.Tensor(np.ones((6, 4, 5)))

@ms.jit()
def test(w):
    x = (2, 3, 4)
    y = [2, 3, 4]
    d = {"a": 2, "b": 3}
    n = np.array([1, 2, 3, 4])
    x_len = len(x)
    y_len = len(y)
    d_len = len(d)
    z_len = len(z)
    n_len = len(n)
    w_len = len(w.asnumpy())
    return x_len, y_len, d_len, z_len, n_len, w_len

input_x = ms.Tensor([1, 2, 3, 4])
x_len, y_len, d_len, z_len, n_len, w_len = test(input_x)
print('x_len:{}'.format(x_len))
print('y_len:{}'.format(y_len))
print('d_len:{}'.format(d_len))
print('z_len:{}'.format(z_len))
print('n_len:{}'.format(n_len))
print('w_len:{}'.format(w_len))
```

结果如下：

```text
x_len:3
y_len:3
d_len:2
z_len:6
z_len:4
w_len:4
```

其中`len(w.asnumpy())`属于高阶用法，更多介绍可见[扩展语法（LAX级别）](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html#扩展语法lax级别)章节。

## isinstance

功能：判断对象是否为一个已知的类型。

调用：`isinstance(obj, type)`。

入参：

- `obj` -- MindSpore支持类型的一个实例。

- `type` -- `bool`、`int`、`float`、`str`、`list`、`tuple`、`dict`、`Tensor`、`Parameter`，或者第三方库的类型（例如numpy.ndarray）或者是一个只包含这些类型的`tuple`。

返回值：`obj`为`type`的实例，返回`True`，否则返回`False`。

示例如下：

```python
import mindspore as ms
import numpy as np

z = ms.Tensor(np.ones((6, 4, 5)))

@ms.jit()
def test(w):
    x = (2, 3, 4)
    y = [2, 3, 4]
    x_is_tuple = isinstance(x, tuple)
    y_is_list = isinstance(y, list)
    z_is_tensor = isinstance(z, ms.Tensor)
    w_is_ndarray = isinstance(w.asnumpy(), np.ndarray)
    return x_is_tuple, y_is_list, z_is_tensor, w_is_ndarray

w = ms.Tensor(np.array([-1, 2, 4]))
x_is_tuple, y_is_list, z_is_tensor, w_is_ndarray = test(w)
print('x_is_tuple:{}'.format(x_is_tuple))
print('y_is_list:{}'.format(y_is_list))
print('z_is_tensor:{}'.format(z_is_tensor))
print('w_is_ndarray:{}'.format(w_is_ndarray))
```

结果如下：

```text
x_is_tuple:True
y_is_list:True
z_is_tensor:True
w_is_ndarray:True
```

其中`isinstance(w.asnumpy(), np.ndarray)`属于高阶用法，更多介绍可见[扩展语法（LAX级别）](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html#扩展语法lax级别)章节。

## all

功能：判断输入中的元素是否均为真值。

调用：`all(x)`。

入参：`x` -- 可迭代对象，支持类型包括`tuple`、`list`、`dict`、`Tensor`以及第三方对象（例如`numpy.ndarray`）。

返回值：布尔值，如果所有元素都为`True`，则返回`True`，否则返回`False`。

代码用例如下：

```python
import numpy as np
import mindspore as ms
from mindspore import Tensor

@ms.jit
def func():
   a = all(['a', 'b', 'c', 'd'])
   b = all(['a', 'b', '', 'd'])
   c = all([0, 1, 2, 3])
   d = all(('a', 'b', 'c', 'd'))
   e = all(('a', 'b', '', 'd'))
   f = all((0, 1, 2, 3))
   g = all([])
   h = all(())
   x = Tensor(np.array([0, 1, 2, 3]))
   i = all(x.asnumpy())
   return a, b, c, d, e, f, g, h, i

a, b, c, d, e, f, g, h, i = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
print("f: ", f)
print("g: ", g)
print("h: ", h)
print("i: ", i)
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
i: False
```

其中`all(x.asnumpy())`属于高阶用法，更多介绍可见[扩展语法（LAX级别）](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html#扩展语法lax级别)章节。

## any

功能：判断输入中的元素是存在为真值。

调用：`any(x)`。

入参：`x` -- 可迭代对象，支持类型包括`tuple`、`list`、`dict`、`Tensor`以及第三方对象（例如`numpy.ndarray`）。

返回值：布尔值，如果所有元素都为`False`，则返回`False`，否则返回`True`。元素除了0，空，`False`外都算`True`。

代码用例如下：

```python
import numpy as np
import mindspore as ms
from mindspore import Tensor

@ms.jit
def func():
   a = any(['a', 'b', 'c', 'd'])
   b = any(['a', 'b', '', 'd'])
   c = any([0, '', False])
   d = any(('a', 'b', 'c', 'd'))
   e = any(('a', 'b', '', 'd'))
   f = any((0, '', False))
   g = any([])
   h = any(())
   x = Tensor(np.array([0, 1, 2, 3]))
   i = any(x.asnumpy())
   return a, b, c, d, e, f, g, h, i

a, b, c, d, e, f, g, h, i = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
print("f: ", f)
print("g: ", g)
print("h: ", h)
print("i: ", i)
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
i: True
```

## round

功能：返回输入的四舍五入。

调用：`round(x, digit=0)`。

入参：

- `x` -- 需要四舍五入的值，有效类型为 `int`、`float`、`bool`、`Tensor`以及定义了魔术方法`__round__()`第三方对象。

- `digit` -- 表示进行四舍五入的小数点位数，默认值为0，支持`int`类型以及`None`。若`x`为`Tensor`类型，则不支持输入`digit`。

返回值：四舍五入后的值。

代码用例如下：

```python
import mindspore as ms

@ms.jit
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
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
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

## max

功能：返回给定参数的最大值。

调用：`max(*data)`。

入参： - `*data` -- 若`*data`为单输入，则会比较单个输入内的各个元素，此时`data`必须为可迭代对象。若存在多个输入，则比较每个输入。`data`有效类型为`int`、`float`、`bool`、`list`、`tuple`、`dict`、`Tensor`以及第三方对象（例如`numpy.ndarray`）。

返回值：最大值。

代码用例如下：

```python
import numpy as np
import mindspore as ms

@ms.jit
def func():
   a = max([0, 1, 2, 3])
   b = max((0, 1, 2, 3))
   c = max({1: 10, 2: 20, 3: 3})
   d = max(np.array([1, 2, 3, 4]))
   e = max(('a', 'b', 'c'))
   f = max((1, 2, 3), (1, 4))
   g = max(ms.Tensor([1, 2, 3]))
   return a, b, c, ms.Tensor(d), e, f, g

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

## min

功能：返回给定参数的最小值。

调用：`min(*data)`。

入参： - `*data` -- 若`*data`为单输入，则会比较单个输入内的各个元素，此时`data`必须为可迭代对象。若存在多个输入，则比较每个输入。`data`有效类型为`int`、`float`、`bool`、`list`、`tuple`、`dict`、`Tensor`以及第三方对象（例如`numpy.ndarray`）。

返回值：最小值。

代码用例如下：

```python
import numpy as np
import mindspore as ms

@ms.jit
def func():
  a = min([0, 1, 2, 3])
  b = min((0, 1, 2, 3))
  c = min({1: 10, 2: 20, 3: 3})
  d = min(np.array([1, 2, 3, 4]))
  e = min(('a', 'b', 'c'))
  f = min((1, 2, 3), (1, 4))
  g = min(ms.Tensor([1, 2, 3]))
  return a, b, c, ms.Tensor(d), e, f, g

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

## sum

功能：对输入序列进行求和计算。

调用：`sum(x, n=0)`。

入参：

- `x` -- 表示可迭代对象，有效类型为`list`、`tuple`、`Tensor`以及第三方对象（例如`numpy.ndarray`）。

- `n` -- 表示指定相加的参数，缺省值为0。

返回值：对`x`求和后与`n`相加得到的值。

代码用例如下：

```python
import numpy as np
import mindspore as ms

@ms.jit
def func():
  a = sum([0, 1, 2])
  b = sum((0, 1, 2), 10)
  c = sum(np.array([1, 2, 3]))
  d = sum(ms.Tensor([1, 2, 3]), 10)
  e = sum(ms.Tensor([[1, 2], [3, 4]]))
  f = sum([1, ms.Tensor([[1, 2], [3, 4]]), ms.Tensor([[1, 2], [3, 4]])], ms.Tensor([[1, 1], [1, 1]]))
  return a, b, ms.Tensor(c), d, e, f

a, b, c, d, e, f = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
print("f: ", f)
```

输出结果：

```text
a:  3
b:  13
c:  6
d:  16
e:  [4 6]
f:  [[ 4  6]
     [ 8 10]]
```

## abs

功能：返回给定参数的绝对值。

调用：`abs(x)`。

入参： - `x` -- 有效类型为`int`、`float`、`bool`、`Tensor`以及第三方对象（例如`numpy.ndarray`）。

返回值：绝对值。

代码用例如下：

```python
import mindspore as ms
from mindspore import Tensor

@ms.jit
def func():
   a = abs(-45)
   b = abs(100.12)
   c = abs(Tensor([-1, 2]).asnumpy())
   return a, b, c

a, b, c = func()
print("a: ", a)
print("b: {:.2f}".format(b))
print("c: ", c)
```

输出结果：

```text
a: 45
b: 100.12
c: [1 2]
```

其中`abs(Tensor([-1, 2]).asnumpy())`属于高阶用法，更多介绍可见[扩展语法（LAX级别）](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html#扩展语法lax级别)章节。

## map

功能：根据提供的函数对一个或者多个序列做映射，由映射的结果生成一个新的序列。当前要求多个序列中的元素个数一致。

调用：`map(func, sequence, ...)`。

入参：

- `func` -- 函数。

- `sequence` -- 一个或多个序列（`Tuple`或者`List`）。

返回值：返回一个新的序列。

示例如下：

```python
import mindspore as ms

def add(x, y):
    return x + y

@ms.jit()
def test():
    elements_a = (1, 2, 3)
    elements_b = (4, 5, 6)
    ret1 = map(add, elements_a, elements_b)
    elements_c = [0, 1, 2]
    elements_d = [6, 7, 8]
    ret2 = map(add, elements_c, elements_d)
    return ret1, ret2

ret1, ret2 = test()
print('ret1:{}'.format(ret1))
print('ret2:{}'.format(ret2))
```

结果如下：

```text
ret1: (5, 7, 9)
ret2: [6, 8, 10]
```

## zip

功能：将多个序列中对应位置的元素打包成一个个元组，然后由这些元组组成一个新序列，如果各个序列中的元素个数不一致，则生成的新序列与最短的那个长度相同。

调用：`zip(sequence, ...)`。

入参：`sequence` -- 一个或多个序列(`Tuple`或`List`)。

返回值：返回一个新的序列。

示例如下：

```python
import mindspore as ms

@ms.jit()
def test():
    elements_a = (1, 2, 3)
    elements_b = (4, 5, 6, 7)
    ret = zip(elements_a, elements_b)
    return ret

ret = test()
print('ret:{}'.format(ret))
```

结果如下：

```text
ret:((1, 4), (2, 5), (3, 6))
```

## range

功能：根据起始值、结束值和步长创建一个`Tuple`。

调用：

- `range(start, stop, step)`

- `range(start, stop)`

- `range(stop)`

入参：

- `start` -- 计数起始值，类型为`int`，默认为0。

- `stop` -- 计数结束值，但不包括在内，类型为`int`。

- `step` -- 步长，类型为`int`，默认为1。

返回值：返回一个`Tuple`。

示例如下：

```python
import mindspore as ms

@ms.jit()
def test():
    x = range(0, 6, 2)
    y = range(0, 5)
    z = range(3)
    return x, y, z

x, y, z = test()
print('x:{}'.format(x))
print('y:{}'.format(y))
print('z:{}'.format(z))
```

结果如下：

```text
x:(0, 2, 4)
y:(0, 1, 2, 3, 4)
z:(0, 1, 2)
```

## enumerate

功能：生成一个序列的索引序列，索引序列包含数据和对应下标。

调用：

- `enumerate(sequence, start=0)`

- `enumerate(sequence)`

入参：

- `sequence` -- 一个序列（`Tuple`、`List`、`Tensor`）。

- `start` -- 下标起始位置，类型为`int`，默认为0。

返回值：返回一个`Tuple`。

示例如下：

```python
import mindspore as ms
import numpy as np

y = ms.Tensor(np.array([[1, 2], [3, 4], [5, 6]]))

@ms.jit()
def test():
    x = (100, 200, 300, 400)
    m = enumerate(x, 3)
    n = enumerate(y)
    return m, n

m, n = test()
print('m:{}'.format(m))
print('n:{}'.format(n))
```

结果如下：

```text
m:((3, 100), (4, 200), (5, 300), (6, 400))
n:((0, Tensor(shape=[2], dtype=Int64, value= [1, 2])), (1, Tensor(shape=[2], dtype=Int64, value= [3, 4])), (2, Tensor(shape=[2], dtype=Int64, value= [5, 6])))
```

## super

功能：用于调用父类(超类)的一个方法，一般在`super`之后调用父类的方法。

调用：

- `super().xxx()`

- `super(type, self).xxx()`

入参：

- `type` -- 类。

- `self` -- 对象。

返回值：返回父类的方法。

示例如下：

```python
import mindspore as ms
from mindspore import nn, set_context

set_context(mode=ms.GRAPH_MODE)

class FatherNet(nn.Cell):
    def __init__(self, x):
        super(FatherNet, self).__init__(x)
        self.x = x

    def construct(self, x, y):
        return self.x * x

    def test_father(self, x):
        return self.x + x

class SingleSubNet(FatherNet):
    def __init__(self, x, z):
        super(SingleSubNet, self).__init__(x)
        self.z = z

    def construct(self, x, y):
        ret_father_construct = super().construct(x, y)
        ret_father_test = super(SingleSubNet, self).test_father(x)
        return ret_father_construct, ret_father_test

x = 3
y = 6
z = 9
f_net = FatherNet(x)
net = SingleSubNet(x, z)
out = net(x, y)
print("out:", out)
```

结果如下：

```text
out: (9, 6)
```

## pow

功能：求幂。

调用：`pow(x, y)`

入参：

- `x` -- 底数， `Number`或`Tensor`。

- `y` -- 幂指数， `Number`或`Tensor`。

返回值：返回`x`的`y`次幂，`Number`或`Tensor`。

示例如下：

```python
import mindspore as ms
import numpy as np

x = ms.Tensor(np.array([1, 2, 3]))
y = ms.Tensor(np.array([1, 2, 3]))

@ms.jit()
def test(x, y):
    return pow(x, y)

ret = test(x, y)

print('ret:{}'.format(ret))
```

结果如下：

```text
ret:[ 1  4 27]
```

## print

功能：用于打印。

调用：`print(arg, ...)`

入参：`arg` -- 要打印的信息(`int` 、`float`、`bool`、`String`或`Tensor`，或者第三方库的数据类型)。

返回值：无返回值。

示例如下：

```python
import mindspore as ms
import numpy as np

x = ms.Tensor(np.array([1, 2, 3]), ms.int32)
y = ms.Tensor(3, ms.int32)

@ms.jit()
def test(x, y):
    print(x)
    print(y)
    return x, y

ret = test(x, y)
```

结果如下：

```text
Tensor(shape=[3], dtype=Int32, value= [1 2 3])
Tensor(shape=[], dtype=Int32, value=3)
```

## filter

功能：根据提供的函数对一个序列的元素做判断，每个元素依次作为参数传入函数中，将返回结果不为0或False的元素组成新的序列。

调用：`filter(func, sequence)`

入参：

- `func` -- 函数。

- `sequence` -- 序列（`Tuple`或`List`）。

返回值：返回一个新的序列。

示例如下：

```python
import mindspore as ms

def is_odd(x):
    if x % 2:
        return True
    return False

@ms.jit()
def test():
    elements1 = (1, 2, 3, 4, 5)
    ret1 = filter(is_odd, elements1)
    elements2 = [6, 7, 8, 9, 10]
    ret2 = filter(is_odd, elements2)
    return ret1, ret2

ret1, ret2 = test()
print('ret1:{}'.format(ret1))
print('ret2:{}'.format(ret2))
```

结果如下：

```text
ret1:[1, 3, 5]
ret2:[7, 9]
```

## type

功能：输出入参的类型。

有效输入：Number、list、tuple、dict、numpy.ndarray、常量Tensor。

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

a, b, c, d, e, f, g = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
print("f: ", f)
print("g: ", g)
```

```text
a: <class 'int'>
b: <class 'float'>
c: <class 'list'>
d: <class 'tuple'>
e: <class 'dict'>
f: <class 'numpy.ndarray'>
g: <class 'mindspore.common.tensor.Tensor'>
```

> type作为Python的原生函数还有另外一种使用方法，即type(name, bases, dict)返回name类型的类对象，由于该用法应用场景较少，因此暂不支持。
