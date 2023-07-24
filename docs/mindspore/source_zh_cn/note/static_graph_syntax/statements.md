# 静态图语法——Python语句

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/static_graph_syntax/statements.md)

## 简单语句

### raise语句

支持使用`raise`触发异常，`raise`语法格式：`raise[Exception [, args]]`。语句中的`Exception`是异常的类型，`args`是用户提供的异常参数，通常可以是字符串或者其他对象。目前支持的异常类型有：NoExceptionType、UnknownError、ArgumentError、NotSupportError、NotExistsError、DeviceProcessError、AbortedError、IndexError、ValueError、TypeError、KeyError、AttributeError、NameError、AssertionError、BaseException、KeyboardInterrupt、Exception、StopIteration、OverflowError、ZeroDivisionError、EnvironmentError、IOError、OSError、ImportError、MemoryError、UnboundLocalError、RuntimeError、NotImplementedError、IndentationError、RuntimeWarning。

例如：

```python
import mindspore.nn as nn
import mindspore as ms

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()

    def construct(self, x, y):
        if x <= y:
            raise ValueError("x should be greater than y.")
        else:
            x += 1
        return x

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
net(ms.Tensor(-2), ms.Tensor(-1))
```

输出结果:

```text
ValueError: x should be greater than y.
```

### assert语句

支持使用assert来做异常检查，`assert`语法格式：`assert[Expression [, args]]`。其中`Expression`是判断条件，如果条件为真，就不做任何事情；条件为假时，则将抛出`AssertError`类型的异常信息。`args`是用户提供的异常参数，通常可以是字符串或者其他对象。

```python
import mindspore.nn as nn
import mindspore as ms

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()

    def construct(self, x):
        assert x in [2, 3, 4]
        return x

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
net(ms.Tensor(-1))
```

输出结果中正常出现:

```text
AssertionError.
```

### pass语句

`pass`语句不做任何事情，通常用于占位，保持结构的完整性。例如：

```python
import mindspore as ms
from mindspore import nn, set_context

set_context(mode=ms.GRAPH_MODE)

class Net(nn.Cell):
  def construct(self, x):
    i = 0
    while i < 5:
      if i > 3:
        pass
      else:
        x = x * 1.5
      i += 1
    return x

net = Net()
ret = net(10)
print("ret:", ret)
```

结果如下：

```text
ret: 50.625
```

### return语句

`return`语句通常是将结果返回调用的地方，`return`语句之后的语句不被执行。如果返回语句没有任何表达式或者函数没有`return`语句，则默认返回一个`None`对象。一个函数体内可以根据不同的情况有多个`return`语句。例如：

```python
import mindspore as ms
from mindspore import nn, set_context

set_context(mode=ms.GRAPH_MODE)

class Net(nn.Cell):
  def construct(self, x):
      if x > 0:
        return x
      else:
        return 0

net = Net()
ret = net(10)
print("ret:", ret)
```

输出结果:

```text
ret: 10
```

如上，在控制流场景语句中，可以有多个`return`语句。如果一个函数中没有`return`语句，则默认返回None对象，如下用例：

```python
from mindspore import jit, context

context.set_context(mode=context.GRAPH_MODE)

@jit
def foo():
  x = 3
  print("x:", x)

res = foo()
assert res is None
```

### break语句

`break`语句用来终止循环语句，即循环条件没有`False`条件或者序列还没完全递归完时，也会停止执行循环语句，通常用在`while`和`for`循环中。在嵌套循环中，`break`语句将停止执行最内层的循环。

```python
import mindspore as ms
from mindspore import nn, set_context

set_context(mode=ms.GRAPH_MODE)

class Net(nn.Cell):
  def construct(self, x):
    for i in range(8):
      if i > 5:
        x *= 3
        break
      x = x * 2
    return x

net = Net()
ret = net(10)
print("ret:", ret)
```

得到结果：

```text
ret: 1920
```

### continue语句

`continue`语句用来跳出当前的循环语句，进入下一轮的循环。与`break`语句有所不同，`break`语句用来终止整个循环语句。`continue`也用在`while`和`for`循环中。例如：

```python
import mindspore as ms
from mindspore import nn, set_context

set_context(mode=ms.GRAPH_MODE)

class Net(nn.Cell):
  def construct(self, x):
    for i in range(4):
      if i > 2:
        x *= 3
        continue
    return x


net = Net()
ret = net(3)
print("ret:", ret)
```

得到结果：

```text
ret: 9
```

## 复合语句

### 条件控制语句

#### if语句

使用方式：

- `if (cond): statements...`

- `x = y if (cond) else z`

参数：`cond` -- 支持`Bool`类型的变量，也支持类型为`Number`、`List`、`Tuple`、`Dict`、`String`类型的常量以及`None`对象。

限制：

- 如果`cond`不为常量，在不同分支中同一符号被赋予的变量或者常量的数据类型应一致，如果是被赋予变量或者常量数据类型是`Tensor`，则要求`Tensor`的type和shape也应一致。shape一致性约束详见[ShapeJoin规则](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.1/network/control_flow.html#shapejoin规则)。

示例1：

```python
import mindspore as ms

x = ms.Tensor([1, 4], ms.int32)
y = ms.Tensor([0, 3], ms.int32)
m = 1
n = 2

@ms.jit()
def test_if_cond(x, y):
    if (x > y).any():
        return m
    else:
        return n

ret = test_if_cond(x, y)
print('ret:{}'.format(ret))
```

`if`分支返回的`m`和`else`分支返回的`n`，二者数据类型必须一致。

结果如下:

```text
ret:1
```

示例2：

```python
import mindspore as ms

x = ms.Tensor([1, 4], ms.int32)
y = ms.Tensor([0, 3], ms.int32)
m = 1
n = 2

@ms.jit()
def test_if_cond(x, y):
    out = 3
    if (x > y).any():
        out = m
    else:
        out = n
    return out

ret = test_if_cond(x, y)
print('ret:{}'.format(ret))
```

`if`分支中`out`被赋值的变量或者常量`m`与`else`分支中`out`被赋值的变量或者常量`n`的数据类型必须一致。

结果如下:

```text
ret:1
```

示例3：

```python
import mindspore as ms

x = ms.Tensor([1, 4], ms.int32)
y = ms.Tensor([0, 3], ms.int32)
m = 1

@ms.jit()
def test_if_cond(x, y):
    out = 2
    if (x > y).any():
        out = m
    return out

ret = test_if_cond(x, y)
print('ret:{}'.format(ret))
```

`if`分支中`out`被赋值的变量或者常量`m`与`out`初始赋值的数据类型必须一致。

结果如下:

```text
ret:1
```

### 循环语句

#### for语句

使用方式：

- `for i in sequence  statements...`

- `for i in sequence  statements... if (cond) break`

- `for i in sequence  statements... if (cond) continue`

参数：`sequence` -- 遍历序列(`Tuple`、`List`、`range`等)

限制：

- 图的算子数量和`for`循环的迭代次数成倍数关系，`for`循环迭代次数过大可能会导致图占用内存超过使用限制。

- 不支持`for...else...`语句。

示例：

```python
import numpy as np
import mindspore as ms

z = ms.Tensor(np.ones((2, 3)))

@ms.jit()
def test_for_cond():
    x = (1, 2, 3)
    for i in x:
        z += i
    return z

ret = test_for_cond()
print('ret:{}'.format(ret))
```

结果如下：

```text
ret:[[7. 7. 7.]
 [7. 7. 7.]]
```

#### while语句

使用方式：

- `while (cond)  statements...`

- `while (cond)  statements... if (cond1) break`

- `while (cond)  statements... if (cond1) continue`

参数：`cond` -- 支持`Bool`类型的变量，也支持类型为`Number`、`List`、`Tuple`、`Dict`、`String`类型的常量以及`None`对象。

限制：

- 如果`cond`不为常量，在循环体内外同一符号被赋值的变量或者常量的数据类型应一致，如果是被赋予数据类型`Tensor`，则要求`Tensor`的type和shape也应一致。shape一致性约束详见[ShapeJoin规则](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.1/network/control_flow.html#shapejoin规则)。

- 不支持`while...else...`语句。

示例1：

```python
import mindspore as ms

m = 1
n = 2

@ms.jit()
def test_cond(x, y):
    while x < y:
        x += 1
        return m
    return n

ret = test_cond(1, 5)
print('ret:{}'.format(ret))
```

`while`循环内返回的`m`和`while`外返回的`n`数据类型必须一致。

结果如下：

```text
ret:1
```

示例2：

```python
import mindspore as ms

m = 1
n = 2

def ops1(a, b):
    return a + b

@ms.jit()
def test_cond(x, y):
    out = m
    while x < y:
        x += 1
        out = ops1(out, x)
    return out

ret = test_cond(1, 5)
print('ret:{}'.format(ret))
```

`while`内，`out`在循环体内被赋值的变量`op1`的输出类型和初始类型`m`必须一致。

结果如下：

```text
ret:15
```

### 函数定义语句

#### def关键字

`def`用于定义函数，后接函数标识符名称和原括号`()`，括号中可以包含函数的参数。
使用方式：`def function_name(args): statements...`。

示例如下：

```python
import mindspore as ms

def number_add(x, y):
    return x + y

@ms.jit()
def test(x, y):
    return number_add(x, y)

ret = test(1, 5)
print('ret:{}'.format(ret))
```

结果如下：

```text
ret:6
```

说明：

- 函数可以支持不写返回值，不写返回值默认函数的返回值为None。
- 支持最外层网络模型的`construct`函数和内层网络函数输入kwargs，即支持 `def construct(**kwargs):`。
- 支持变参和非变参的混合使用，即支持 `def function(x, y, *args):`和 `def function(x = 1, y = 1, **kwargs):`。

#### lambda表达式

`lambda`表达式用于生成匿名函数。与普通函数不同，它只计算并返回一个表达式。使用方式：`lambda x, y: x + y`。

示例如下：

```python
import mindspore as ms

@ms.jit()
def test(x, y):
    number_add = lambda x, y: x + y
    return number_add(x, y)

ret = test(1, 5)
print('ret:{}'.format(ret))
```

结果如下：

```text
ret:6
```

#### 偏函数partial

功能：偏函数，固定函数入参。使用方式：`partial(func, arg, ...)`。

入参：

- `func` -- 函数。

- `arg` -- 一个或多个要固定的参数，支持位置参数和键值对传参。

返回值：返回某些入参固定了值的函数。

示例如下：

```python
import mindspore as ms
from mindspore import ops

def add(x, y):
    return x + y

@ms.jit()
def test():
    add_ = ops.partial(add, x=2)
    m = add_(y=3)
    n = add_(y=5)
    return m, n

m, n = test()
print('m:{}'.format(m))
print('n:{}'.format(n))
```

结果如下：

```text
m:5
n:7
```

#### 函数参数

- 参数默认值：目前不支持默认值设为`Tensor`类型数据，支持`int`、`float`、`bool`、`None`、`str`、`tuple`、`list`、`dict`类型数据。
- 可变参数：支持带可变参数网络的推理和训练。
- 键值对参数：目前不支持带键值对参数的函数求反向。
- 可变键值对参数：目前不支持带可变键值对的函数求反向。

### 列表生成式和生成器表达式

支持列表生成式（List Comprehension）和生成器表达式（Generator Expression）。支持构建一个新的序列。

#### 列表生成式

列表生成式用于生成列表。使用方式：`[arg for loop if statements]`。

示例如下：

```python
import mindspore as ms

@ms.jit()
def test():
    l = [x * x for x in range(1, 11) if x % 2 == 0]
    return l

ret = test()
print('ret:{}'.format(ret))
```

结果如下：

```text
ret:[4, 16, 36, 64, 100]
```

限制：

图模式下不支持多层嵌套迭代器的使用方式。

限制用法示例如下（使用了两层迭代器）：

```python
l = [y for x in ((1, 2), (3, 4), (5, 6)) for y in x]
```

会提示错误：

```text
TypeError:  The `generators` supports one `comprehension` in ListComp/GeneratorExp, but got 2 comprehensions.
```

#### 生成器表达式

生成器表达式用于生成列表。使用方式：`(arg for loop if statements)`。

示例如下：

```python
import mindspore as ms

@ms.jit()
def test():
    l = (x * x for x in range(1, 11) if x % 2 == 0)
    return l

ret = test()
print('ret:{}'.format(ret))
```

结果如下：

```text
ret:[4, 16, 36, 64, 100]
```

使用限制同列表生成式。即：图模式下不支持多层嵌套迭代器的使用方式。

### with语句

在图模式下，有限制地支持`with`语句。`with`语句要求对象必须有两个魔术方法：`__enter__()`和`__exit__()`。

示例如下：

```python
import mindspore as ms
import mindspore.nn as nn
from mindspore import set_context

set_context(mode=ms.GRAPH_MODE)

@ms.jit_class
class Sample:
    def __init__(self):
        super(Sample, self).__init__()
        self.num = ms.Tensor([2])

    def __enter__(self):
        return self.num * 2

    def __exit__(self, exc_type, exc_value, traceback):
        return self.num * 4

class TestNet(nn.Cell):
    def construct(self):
        res = 1
        obj = Sample()
        with obj as sample:
            res += sample
        return res, obj.num

test_net = TestNet()
out1, out2 = test_net()
print("out1:", out1)
print("out2:", out2)
```

结果如下：

```text
out1: [5]
out2: [2]
```
