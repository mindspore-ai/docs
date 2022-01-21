# 静态图语法支持

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/note/source_zh_cn/static_graph_syntax_support.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## 概述

在Graph模式下，Python代码并不是由Python解释器去执行，而是将代码编译成静态计算图，然后执行静态计算图。

当前仅支持编译`@ms_function`装饰器修饰的函数、Cell及其子类的实例。
对于函数，则编译函数定义；对于网络，则编译`construct`方法及其调用的其他方法或者函数。

`ms_function`使用规则可参考文档：<https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore/mindspore.ms_function.html#mindspore.ms_function>

`Cell`定义可参考文档：<https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/cell.html>

由于语法解析的限制，当前在编译构图时，支持的数据类型、语法以及相关操作并没有完全与Python语法保持一致，部分使用受限。

本文主要介绍，在编译静态图时，支持的数据类型、语法以及相关操作，这些规则仅适用于Graph模式。

> 以下所有示例都运行在Graph模式下的网络中，为了简洁，并未将网络的定义都写出来。
>

## 数据类型

### Python内置数据类型

当前支持的`Python`内置数据类型包括：`Number`、`String`、`List`、`Tuple`和`Dictionary`。

#### Number

支持`int`、`float`、`bool`，不支持complex（复数）。

支持在网络里定义`Number`，即支持语法：`y = 1`、`y = 1.2`、 `y = True`。

不支持在网络里强转`Number`，即不支持语法：`y = int(x)`、`y = float(x)`、`y = bool(x)`。

#### String

支持在网络里构造`String`，即支持语法`y = "abcd"`。

不支持在网络里强转`String`，即不支持语法 `y = str(x)`。

#### List

支持在网络里构造`List`，即支持语法`y = [1, 2, 3]`。

不支持在网络里强转`List`，即不支持语法`y = list(x)`。

计算图中最终需要输出的`List`会转换为`Tuple`输出。

- 支持接口

  `append`: 向`list`里追加元素。

  示例如下：

  ```python
  from mindspore import ms_function

  @ms_function()
  def test_list():
      x = [1, 2, 3]
      x.append(4)
      return x

  x = test_list()
  print('x:{}'.format(x))
  ```

  结果如下：

  ```text
  x: (1, 2, 3, 4)
  ```

- 支持索引取值和赋值

  支持单层和多层索引取值以及赋值。

  取值和赋值的索引值仅支持`int`。

  赋值时，所赋的值支持`Number`、`String`、`Tuple`、`List`、`Tensor`。

  示例如下：

  ```python
  from mindspore import ms_function, Tensor
  import numpy as np

  t = Tensor(np.array([1, 2, 3]))

  @ms_function()
  def test_index():
      x = [[1, 2], 2, 3, 4]
      m = x[0][1]
      x[1] = t
      x[2] = "ok"
      x[3] = (1, 2, 3)
      x[0][1] = 88
      n = x[-3]
      return m, x, n

  m, x, n = test_index()
  print('m:{}'.format(m))
  print('x:{}'.format(x))
  print('n:{}'.format(n))
  ```

  结果如下：

  ```text
  m:2
  x:[[1, 88], Tensor(shape=[3], dtype=Int64, value= [1, 2, 3]), 'ok', (1, 2, 3)]
  n:[1 2 3]
  ```

#### Tuple

支持在网络里构造`Tuple`，即支持语法`y = (1, 2, 3)`。

不支持在网络里强转`Tuple`，即不支持语法`y = tuple(x)`。

- 支持索引取值

  索引值支持`int`、`slice`、`Tensor`，也支持多层索引取值，即支持语法`data = tuple_x[index0][index1]...`。

  索引值为`Tensor`有如下限制：

    - `tuple`里存放的都是`Cell`，每个`Cell`要在tuple定义之前完成定义，每个`Cell`的入参个数、入参类型和入参`shape`要求一致，每个`Cell`的输出个数、输出类型和输出`shape`也要求一致。

    - 索引`Tensor`是一个`dtype`为`int32`的标量`Tensor`，取值范围在`[-tuple_len, tuple_len)`，`Ascend`后端不支持负数索引。

    - 该语法不支持`if`、`while`、`for`控制流条件为变量的运行分支，仅支持控制流条件为常量。

    - 支持`GPU`和`Ascend`后端。

  `int`、`slice`索引示例如下：

  ```python
  from mindspore import ms_function, Tensor
  import numpy as np

  t = Tensor(np.array([1, 2, 3]))

  @ms_function()
  def test_index():
      x = (1, (2, 3, 4), 3, 4, t)
      y = x[1][1]
      z = x[4]
      m = x[1:4]
      n = x[-4]
      return y, z, m, n

  y, z, m, n = test_index()
  print('y:{}'.format(y))
  print('z:{}'.format(z))
  print('m:{}'.format(m))
  print('n:{}'.format(n))
  ```

  结果如下：

  ```text
  y:3
  z:[1 2 3]
  m:((2, 3, 4), 3, 4)
  n:(2, 3, 4)
  ```

  `Tensor`索引示例如下：

  ```python
  from mindspore import Tensor, nn, dtype

  class Net(nn.Cell):
      def __init__(self):
          super(Net, self).__init__()
          self.relu = nn.ReLU()
          self.softmax = nn.Softmax()
          self.layers = (self.relu, self.softmax)

      def construct(self, x, index):
          ret = self.layers[index](x)
          return ret

  x = Tensor([-1.0], dtype.float32)

  net = Net()
  ret = net(x, 0)
  print('ret:{}'.format(ret))
  ```

  结果如下：

  ```text
  ret:[0.]
  ```

#### Dictionary

支持在网络里构造`Dictionary`，即支持语法`y = {"a": 1, "b": 2}`，当前仅支持`String`作为`key`值。

计算图中最终需要输出的`Dictionary`，会取出所有的`value`组成`Tuple`输出。

- 支持接口

  `keys`：取出`dict`里所有的`key`值，组成`Tuple`返回。

  `values`：取出`dict`里所有的`value`值，组成`Tuple`返回。

  `items`：取出`dict`里每一对`key`和`value`组成的`Tuple`，组成`Tuple`返回。

  示例如下：

  ```python
  from mindspore import ms_function, Tensor
  import numpy as np

  x = {"a": Tensor(np.array([1, 2, 3])), "b": Tensor(np.array([4, 5, 6])), "c": Tensor(np.array([7, 8, 9]))}

  @ms_function()
  def test_dict():
      y = x.keys()
      z = x.values()
      q = x.items()
      return y, z, q

  y, z, q = test_dict()
  print('y:{}'.format(y))
  print('z:{}'.format(z))
  print('q:{}'.format(q))
  ```

  结果如下：

  ```text
  y:('a', 'b', 'c')
  z:(Tensor(shape=[3], dtype=Int64, value= [1, 2, 3]), Tensor(shape=[3], dtype=Int64, value= [4, 5, 6]), Tensor(shape=[3], dtype=Int64, value= [7, 8, 9]))
  q:[('a', Tensor(shape=[3], dtype=Int64, value= [1, 2, 3])), ('b', Tensor(shape=[3], dtype=Int64, value= [4, 5, 6])), ('c', Tensor(shape=[3], dtype=Int64, value= [7, 8, 9]))]
  ```

- 支持索引取值和赋值

  取值和赋值的索引值都仅支持`String`。赋值时，所赋的值支持`Number`、`Tuple`、`Tensor`。

  示例如下：

  ```python
  from mindspore import ms_function, Tensor
  import numpy as np

  x = {"a": Tensor(np.array([1, 2, 3])), "b": Tensor(np.array([4, 5, 6])), "c": Tensor(np.array([7, 8, 9]))}

  @ms_function()
  def test_dict():
      y = x["b"]
      x["a"] = (2, 3, 4)
      return x, y

  x, y = test_dict()
  print('x:{}'.format(x))
  print('y:{}'.format(y))
  ```

  结果如下：

  ```text
  x:{'a': (2, 3, 4), 'b': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6]), 'c': Tensor(shape=[3], dtype=Int64, value= [7, 8, 9])}
  y:[4 5 6]
  ```

### MindSpore自定义数据类型

当前MindSpore自定义数据类型包括：`Tensor`、`Primitive`、`Cell`和`Parameter`。

#### Tensor

当前不支持在网络里构造Tensor，即不支持语法`x = Tensor(args...)`。

可以通过`@constexpr`装饰器修饰函数，在函数里生成`Tensor`。

关于`@constexpr`的用法可参考：<https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.constexpr.html>

对于网络中需要用到的常量`Tensor`，可以作为网络的属性，在`init`的时候定义，即`self.x = Tensor(args...)`，然后在`construct`里使用。

如下示例，通过`@constexpr`生成一个`shape = (3, 4), dtype = int64`的`Tensor`。

```python
from mindspore import Tensor
from mindspore.ops import constexpr

@constexpr
def generate_tensor():
    return Tensor(np.ones((3, 4)))

x = generate_tensor()
print('x:{}'.format(x))
```

结果如下：

```Text
x:[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
```

下面将介绍下`Tensor`支持的属性和接口。

- 支持属性：

  `shape`：获取`Tensor`的shape，返回一个`Tuple`。

  `dtype`：获取`Tensor`的数据类型，返回一个`MindSpore`定义的数据类型。

- 支持接口：

  `all`：对`Tensor`通过`all`操作进行归约，仅支持`Bool`类型的`Tensor`。

  `any`：对`Tensor`通过`any`操作进行归约，仅支持`Bool`类型的`Tensor`。

  `view`：将`Tensor`reshape成输入的`shape`。

  `expand_as`：将`Tensor`按照广播规则扩展成与另一个`Tensor`相同的`shape`。

  示例如下：

  ```python
  from mindspore import ms_function, Tensor
  import numpy as np

  x = Tensor(np.array([[True, False, True], [False, True, False]]))
  y = Tensor(np.ones((2, 3), np.float32))
  z = Tensor(np.ones((2, 2, 3)))

  x_shape = x.shape
  x_dtype = x.dtype
  x_all = x.all()
  x_any = x.any()
  x_view = x.view((1, 6))
  y_as_z = y.expand_as(z)

  print('x_shape:{}'.format(x_shape))
  print('x_dtype:{}'.format(x_dtype))
  print('x_all:{}'.format(x_all))
  print('x_any:{}'.format(x_any))
  print('x_view:{}'.format(x_view))
  print('y_as_z:{}'.format(y_as_z))
  ```

  结果如下:

  ```text
  x_shape:(2, 3)
  x_dtype:Bool
  x_all:False
  x_any:True
  x_view:[[ True False  True False  True False]]
  y_as_z:[[[1. 1. 1.]
    [1. 1. 1.]]

   [[1. 1. 1.]
    [1. 1. 1.]]]
  ```

#### Primitive

当前支持在网络里构造`Primitive`及其子类的实例，即支持语法`reduce_sum = ReduceSum(True)`。

但在构造时，参数只能通过位置参数方式传入，不支持通过键值对方式传入，即不支持语法`reduce_sum = ReduceSum(keep_dims=True)`。

当前不支持在网络调用`Primitive`及其子类相关属性和接口。

`Primitive`定义可参考文档：<https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/operators_classification.html>

当前已定义的`Primitive`可参考文档：<https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore.ops.html>

#### Cell

当前支持在网络里构造`Cell`及其子类的实例，即支持语法`cell = Cell(args...)`。

但在构造时，参数只能通过位置参数方式传入，不支持通过键值对方式传入，即不支持在语法`cell = Cell(arg_name=value)`。

当前不支持在网络调用`Cell`及其子类相关属性和接口，除非是在`Cell`自己的`construct`中通过`self`调用。

`Cell`定义可参考文档：<https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/cell.html>

当前已定义的`Cell`可参考文档：<https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore.nn.html>

#### Parameter

`Parameter`是变量张量，代表在训练网络时，需要被更新的参数。

`Parameter`的定义和使用参考：<https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/parameter.html>

## 原型

原型代表编程语言中最紧密绑定的操作。

### 属性引用

属性引用是后面带有一个句点加一个名称的原型。

在MindSpore的Cell 实例中使用属性引用作为左值需满足如下要求：

- 被修改的属性属于本`cell`对象，即必须为`self.xxx`。
- 该属性在Cell的`__init__`函数中完成初始化且其为Parameter类型。

示例如下：

```python
from mindspore import ms_function, Tensor, nn, dtype, Parameter
import numpy as np
from mindspore.ops import constexpr

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(Tensor(3, dtype.float32), name="w")
        self.m = 2

    def construct(self, x, y):
        self.weight = x  # 满足条件可以修改
        # self.m = 3  # self.m 非Parameter类型禁止修改
        # y.weight = x  # y不是self，禁止修改
        return x

net = Net()
ret = net(1, 2)
print('ret:{}'.format(ret))
```

结果如下:

```text
ret:1
```

### 索引取值

对序列`Tuple`、`List`、`Dictionary`、`Tensor`的索引取值操作(Python称为抽取)。

`Tuple`的索引取值请参考本文的[Tuple](#tuple)章节。

`List`的索引取值请参考本文的[List](#list)章节。

`Dictionary`的索引取值请参考本文的[Dictionary](#dictionary)章节。

`Tensor`的索引取请参考:<https://www.mindspore.cn/docs/note/zh-CN/r1.6/index_support.html>

### 调用

所谓调用就是附带可能为空的一系列参数来执行一个可调用对象(例如：`Cell`、`Primitive`)。

示例如下：

```python
from mindspore import Tensor, nn, dtype, ops
import numpy as np

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = ops.MatMul()

    def construct(self, x, y):
        out = self.matmul(x, y)  # Primitive调用
        return out

x = Tensor(np.ones(shape=[1, 3]), dtype.float32)
y = Tensor(np.ones(shape=[3, 4]), dtype.float32)
net = Net()
ret = net(x, y)
print('ret:{}'.format(ret))
```

结果如下:

```text
ret:[[3. 3. 3. 3.]]
```

## 运算符

算术运算符和赋值运算符支持`Number`和`Tensor`运算，也支持不同`dtype`的`Tensor`运算。

之所以支持，是因为这些运算符会转换成同名算子进行运算，这些算子支持了隐式类型转换。

规则可参考文档：<https://www.mindspore.cn/docs/note/zh-CN/r1.6/operator_list_implicit.html>

### 单目算术运算符

| 单目算术运算符 | 支持类型                                        |
| :------------- | :---------------------------------------------- |
| `+`            | `Number`、`Tensor`，取正值。                    |
| `-`            | `Number`、`Tensor`，取负值。                    |
| ~              | `Tensor`， 且其数据类型为`Bool`。成员逐个取反。 |

说明：

- 在Python中`~`操作符对输入的整数安位取反; MindSpore对`~`的功能重新定义为对`Tensor(Bool)`的逻辑取反。

### 二元算术运算符

| 二元算术运算符 | 支持类型                                                     |
| :------------- | :----------------------------------------------------------- |
| `+`            | `Number` + `Number`、`String` + `String`、`Number` + `Tensor`、`Tensor` + `Number`、`Tuple` + `Tensor`、`Tensor` + `Tuple`、`List` + `Tensor`、`Tensor`+`List`、`List`+`List`、`Tensor` + `Tensor`、`Tuple` + `Tuple`。 |
| `-`            | `Number` - `Number`、`Tensor` - `Tensor`、`Number` - `Tensor`、`Tensor` - `Number`、`Tuple` - `Tensor`、`Tensor` - `Tuple`、`List` - `Tensor`、`Tensor` - `List`。 |
| `*`            | `Number` \* `Number`、`Tensor` \* `Tensor`、`Number` \* `Tensor`、`Tensor` \* `Number`、`List` \* `Number`、`Number` \* `List`、`Tuple` \* `Number`、`Number` \* `Tuple`、`Tuple` \* `Tensor`、`Tensor` \* `Tuple`、 `List` \* `Tensor`、`Tensor` \* `List`。 |
| `/`            | `Number` / `Number`、`Tensor` / `Tensor`、`Number` / `Tensor`、`Tensor` / `Number`、`Tuple` / `Tensor`、`Tensor` / `Tuple`、`List` / `Tensor`、`Tensor` / `List`。 |
| `%`            | `Number` % `Number`、`Tensor` % `Tensor`、`Number` % `Tensor`、`Tensor` % `Number`、`Tuple` % `Tensor`、`Tensor` % `Tuple`、`List` % `Tensor`、`Tensor` % `List`。 |
| `**`           | `Number` \*\* `Number`、`Tensor` \*\* `Tensor`、`Number` \*\* `Tensor`、`Tensor` \*\* `Number`、`Tuple` \*\* `Tensor`、`Tensor` \*\* `Tuple`、 `List` \*\* `Tensor`、`Tensor` \*\* `List`。 |
| `//`           | `Number` // `Number`、`Tensor` // `Tensor`、`Number` // `Tensor`、`Tensor` // `Number`、`Tuple` // `Tensor`、`Tensor` // `Tuple`、`List` // `Tensor`、`Tensor` // `List`。 |

限制：

- 当左右操作数都为`Number`类型时，`Number`的值不可为`Bool` 类型。
- 当左右操作数都为`Number`类型时，不支持`Float64` 和 `Int32`间的运算。
- 当任一操作数为`Tensor`类型时，左右操作数的值不可同时为`Bool`。
- `List/Tuple`和`Number`进行`*`运算时表示将`List/Tuple`复制`Number`份后串联起来，`List/Tuple`内的数据类型必须为`Number`、`String`、`None`或由以上类型构成的`List/Tuple`。

### 赋值运算符

| 赋值运算符 | 支持类型                                                     |
| :--------- | :----------------------------------------------------------- |
| `=`        | MindSpore支持的Python内置数据类型和MindSpore自定义数据类型   |
| `+=`       | `Number` += `Number`、`String` += `String`、`Number` += `Tensor`、`Tensor` += `Number`、`Tuple` += `Tensor`、`Tensor` += `Tuple`、`List` += `Tensor`、`Tensor` += `List`、`List` += `List`、`Tensor` += `Tensor`、`Tuple` += `Tuple`。 |
| `-=`       | `Number` -= `Number`、`Tensor` -= `Tensor`、`Number` -= `Tensor`、`Tensor` -= `Number`、`Tuple` -= `Tensor`、`Tensor` -= `Tuple`、`List` -= `Tensor`、`Tensor` -= `List`。 |
| `*=`       | `Number` \*= `Number`、`Tensor` \*= `Tensor`、`Number` \*= `Tensor`、`Tensor` \*= `Number`、`List` \*= `Number`、`Number` \*= `List`、`Tuple` \*= `Number`、`Number` \*= `Tuple`、`Tuple` \*= `Tensor`、`Tensor` \*= `Tuple`、 `List` \*= `Tensor`、`Tensor` \*= `List`。 |
| `/=`       | `Number` /= `Number`、`Tensor` /= `Tensor`、`Number` /= `Tensor`、`Tensor` /= `Number`、`Tuple` /= `Tensor`、`Tensor` /= `Tuple`、`List` /= `Tensor`、`Tensor` /= `List`。 |
| `%=`       | `Number` %= `Number`、`Tensor` %= `Tensor`、`Number` %= `Tensor`、`Tensor` %= `Number`、`Tuple` %= `Tensor`、`Tensor` %= `Tuple`、`List` %= `Tensor`、`Tensor` %= `List`。 |
| `**=`      | `Number` \*\*= `Number`、`Tensor` \*\*= `Tensor`、`Number` \*\*= `Tensor`、`Tensor` \*\*= `Number`、`Tuple` \*\*= `Tensor`、`Tensor` \*\*= `Tuple`、 `List` \*\*= `Tensor`、`Tensor` \*\*= `List`。 |
| `//=`      | `Number` //= `Number`、`Tensor` //= `Tensor`、`Number` //= `Tensor`、`Tensor` //= `Number`、`Tuple` //= `Tensor`、`Tensor` //= `Tuple`、`List` //= `Tensor`、`Tensor` //= `List`。 |

限制：

- 对于 `=`来说，不支持下列场景:

  在`construct`函数中仅支持创建`Cell`和`Primitive`类型对象，使用`xx = Tensor(...)`的方式创建`Tensor`会失败。

  在`construct`函数中仅支持为self 的`Parameter`类型的属性赋值, 详情参考：[属性引用](#属性引用)。

- 当`AugAssign`的左右操作数都为`Number`类型时，`Number`的值不可为`Bool` 类型。

- 当`AugAssign`的左右操作数都为`Number`类型时，不支持`Float64` 和 `Int32`间的运算。

- 当`AugAssign`的任一操作数为`Tensor`类型时，左右操作数的值不可同时为`Bool`。

- `List/Tuple`和`Number`进行`*=`运算时表示将`List/Tuple`复制`Number`份后串联起来，`List/Tuple`内的数据类型必须为`Number`、`String`、`None`或由以上类型构成的`List/Tuple`。

### 逻辑运算符

| 逻辑运算符 | 支持类型                                                     |
| :--------- | :----------------------------------------------------------- |
| `and`      | `String`、 `Number`、 `Tuple`、`List` 、`Dict`、`None`、标量、Tensor。 |
| `or`       | `String`、 `Number`、 `Tuple`、`List` 、`Dict`、`None`、标量、Tensor。 |
| `not`      | `Number`、`Tuple`、`List`、只有一个成员的Tensor。            |

限制：

- 当and/or的左操作数是Tensor类型时，左右操作数类型必须保持一致且Tensor成员个数只能有一个。

- 当and/or的左操作数不是Tensor类型时，右操作数可以为支持的任意类型。

### 比较运算符

| 比较运算符 | 支持类型                                                     |
| :--------- | :----------------------------------------------------------- |
| `in`       | `Number` in `tuple`、`String` in `tuple`、`Tensor` in `Tuple`、`Number` in `List`、`String` in `List`、`Tensor` in `List`、`String` in `Dictionary`。 |
| `not in`   | 与`in`相同。                                                 |
| `is`       | 仅支持判断是`None`、 `True`或者`False`。                     |
| `is not`   | 仅支持判断不是`None`、 `True`或者`False`。                   |
| <          | `Number` < `Number`、`Number` < `Tensor`、`Tensor` < `Tensor`、`Tensor` < `Number`。 |
| <=         | `Number` <= `Number`、`Number` <= `Tensor`、`Tensor` <= `Tensor`、`Tensor` <= `Number`。 |
| >          | `Number` > `Number`、`Number` > `Tensor`、`Tensor` > `Tensor`、`Tensor` > `Number`。 |
| >=         | `Number` >= `Number`、`Number` >= `Tensor`、`Tensor` >= `Tensor`、`Tensor` >= `Number`。 |
| !=         | `Number` != `Number`、`Number` != `Tensor`、`Tensor` != `Tensor`、`Tensor` != `Number`、`mstype` != `mstype`、`String` != `String`、`Tuple !` = `Tuple`、`List` != `List`。 |
| ==         | `Number` == `Number`、`Number` == `Tensor`、`Tensor` == `Tensor`、`Tensor` == `Number`、`mstype` == `mstype`、`String` == `String`、`Tuple` == `Tuple`、`List` == `List`。 |

限制：

- 对于`<`、`<=`、`>`、`>=`、`!=`来说，当左右操作数都为`Number`类型时，`Number`的值不可为`Bool` 类型。
- 对于`<`、`<=`、`>`、`>=`、`!=`、`==`来说，当左右操作数都为`Number`类型时，不支持`Float64` 和 `Int32`间的运算。
- 对于`<`、`<=`、`>`、`>=`、`!=`、`==`来说，当左右任一操作数为`Tensor`类型时，左右操作数的值不可同时为`Bool`。
- 对于`==`来说，当左右操作数都为`Number`类型时，支持左右操作数同时为`Bool`，不支持只有一个操作数为`Bool`。
- 对于`!=`、`==`来说除`mstype`外，其他取值均可和`None`进行比较来判空。
- 不支持链式比较，如: `a>b>c`。

## 复合语句

### 条件控制语句

#### if语句

使用方式：

- `if (cond): statements...`

- `x = y if (cond) else z`

参数：`cond` -- 支持`Bool`类型的变量，也支持类型为`Number`、`List`、`Tuple`、`Dict`、`String`类型的常量。

限制：

- 如果`cond`不为常量，在不同分支中同一符号被赋予的变量或者常量的数据类型应一致，如果是被赋予变量或者常量数据类型是`Tensor`，则要求`Tensor`的type和shape也应一致。

- `if`的使用数量不能超过100个。

示例1：

```python
from mindspore import ms_function, Tensor, dtype

x = Tensor([1, 2], dtype.int32)
y = Tensor([0, 3], dtype.int32)
m = 'xx'
n = 'yy'

@ms_function()
def test_cond(x, y):
    if (x > y).any():
        return m
    else:
        return n

ret = test_cond(x, y)
print('ret:{}'.format(ret))
```

`if`分支返回的`m`和`else`分支返回的`n`，二者数据类型必须一致。

结果如下:

  ```text
ret:xx
  ```

示例2：

```python
from mindspore import ms_function, Tensor, dtype

x = Tensor([1, 2], dtype.int32)
y = Tensor([0, 3], dtype.int32)
m = 'xx'
n = 'yy'

@ms_function()
def test_cond(x, y):
    out = 'init'
    if (x > y).any():
        out = m
    else:
        out = n
    return out

ret = test_cond(x, y)
print('ret:{}'.format(ret))
````

`if`分支中`out`被赋值的变量或者常量`m`与`else`分支中`out`被赋值的变量或者常量`n`的数据类型必须一致。

结果如下:

```text
ret:xx
```

示例3：

```python
from mindspore import ms_function, Tensor, dtype

x = Tensor([1, 2], dtype.int32)
y = Tensor([0, 3], dtype.int32)
m = 'xx'

@ms_function()
def test_cond(x, y):
    out = 'init'
    if (x > y).any():
        out = m
    return out

ret = test_cond(x, y)
print('ret:{}'.format(ret))
```

`if`分支中`out`被赋值的变量或者常量`m`与`out`初始赋值的变量或者常量`init`的数据类型必须一致。

结果如下:

```text
ret:xx
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

- `for`的循环体内部不能包含`while`语句。

示例：

```python
from mindspore import ms_function, Tensor
import numpy as np

z = Tensor(np.ones((2, 3)))

@ms_function()
def test_cond():
    x = (1, 2, 3)
    for i in x:
        z += i
    return z

ret = test_cond()
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

参数：`cond` -- 支持`Bool`类型的变量，也支持类型为`Number`、`List`、`Tuple`、`Dict`、`String`类型的常量。

限制：

- 如果`cond`不为常量，在循环体内外同一符号被赋值的变量或者常量的数据类型应一致，如果是被赋予数据类型`Tensor`，则要求`Tensor`的type和shape也应一致。

- 不支持`while...else...`语句

- 如果`cond`不为常量， 循环体内部不能更新循环体外的`Number`、`List`、`Tuple`类型数据， 不能更改`Tensor`类型数据的shape。

- while的数量不能超过100个。

示例1：

```python
from mindspore import ms_function

m = 1
n = 2

@ms_function()
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
from mindspore import ms_function

m = 1
n = 2

def ops1(a, b):
    return a + b

@ms_function()
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

用于定义函数。

使用方式：

`def function_name(args): statements...`

示例如下：

```python
from mindspore import ms_function

def number_add(x, y):
    return x + y

@ms_function()
def test(x, y):
    return number_add(x, y)

ret = test(1, 5)
print('ret:{}'.format(ret))
```

结果如下：

```text
ret: 6
```

限制：

- 函数必须有返回语句。
- 最外层网络模型的`construct`函数不支持kwargs，即不支持 `def  construct(**kwargs):`。
- 不支持变参和非变参的混合使用，即不支持 `def function(x, y, *args):`和 `def function(x = 1, y = 1, **kwargs):`。

#### lambda表达式

用于生成函数。

使用方式：`lambda x, y: x + y`

示例如下：

```python
from mindspore import ms_function

@ms_function()
def test(x, y):
    number_add = lambda x, y: x + y
    return number_add(x, y)

ret = test(1, 5)
print('ret:{}'.format(ret))
```

结果如下：

```text
ret: 6
```

### 列表生成式和生成器表达式

支持列表生成式（List Comprehension）和生成器表达式（Generator Expression）。

#### 列表生成式

用于生成列表。由于编译器会自动把List类型转换成Tuple类型，经过编译后最终输出类型为Tuple。

使用方式：参考Python语法说明。

示例如下：

```python
from mindspore import ms_function

@ms_function()
def test(x, y):
    l = [x * x for x in range(1, 11) if x % 2 == 0]
    return l

ret = test(1, 5)
print('ret:{}'.format(ret))
```

结果如下：

```text
ret:(4, 16, 36, 64, 100)
```

限制：

不支持多层嵌套迭代器的使用方式。

限制用法示例如下（使用了两层迭代器）：

```python
l = [y for x in ((1, 2), (3, 4), (5, 6)) for y in x]
```

会提示错误：

```text
TypeError:  The `generators` supports one `comprehension` in ListComp/GeneratorExp, but got 2 comprehensions.
```

#### 生成器表达式

用于生成列表，与列表生成式动作完全一致，最终的输出类型同样是Tuple。此表达式即刻产生List值，与Python解释器中列表生成式的动作有所差异。

使用方式：同列表生成式。

示例如下：

```python
from mindspore import ms_function

@ms_function()
def test(x, y):
    l = (x * x for x in range(1, 11) if x % 2 == 0)
    return l

ret = test(1, 5)
print('ret:{}'.format(ret))
```

结果如下：

```text
ret:(4, 16, 36, 64, 100)
```

使用限制同列表生成式。

## 函数

### Python内置函数

当前支持的Python内置函数包括：`len`、`isinstance`、`partial`、`map`、`range`、`enumerate`、`super`、`pow`和`filter`。

#### len

功能：求序列的长度。

调用：`len(sequence)`

入参：`sequence` -- `Tuple`、`List`、`Dictionary`或者`Tensor`。

返回值：序列的长度，类型为`int`。当入参是`Tensor`时，返回的是`Tensor`第0维的长度。

示例如下：

```python
from mindspore import ms_function, Tensor
import numpy as np

z = Tensor(np.ones((6, 4, 5)))

@ms_function()
def test():
    x = (2, 3, 4)
    y = [2, 3, 4]
    d = {"a": 2, "b": 3}
    x_len = len(x)
    y_len = len(y)
    d_len = len(d)
    z_len = len(z)
    return x_len, y_len, d_len, z_len

x_len, y_len, d_len, z_len = test()
print('x_len:{}'.format(x_len))
print('y_len:{}'.format(y_len))
print('d_len:{}'.format(d_len))
print('z_len:{}'.format(z_len))
```

结果如下：

```text
x_len:3
y_len:3
d_len:2
z_len:6
```

#### isinstance

功能：判断对象是否为类的实例。区别于算子Isinstance，该算子的第二个入参是MindSpore的dtype模块下定义的类型。

调用：`isinstance(obj, type)`

入参：

- `obj` -- MindSpore支持类型的一个实例。

- `type` -- `bool`、`int`、`float`、`str`、`list`、`tuple`、`dict`、`Tensor`、`Parameter`，或者是一个只包含这些类型的`tuple`。

返回值：`obj`为`type`的实例，返回`True`，否则返回`False`。

示例如下：

```python
from mindspore import ms_function, Tensor
import numpy as np

z = Tensor(np.ones((6, 4, 5)))

@ms_function()
def test():
    x = (2, 3, 4)
    y = [2, 3, 4]
    x_is_tuple = isinstance(x, tuple)
    y_is_list = isinstance(y, list)
    z_is_tensor = isinstance(z, Tensor)
    return x_is_tuple, y_is_list, z_is_tensor

x_is_tuple, y_is_list, z_is_tensor = test()
print('x_is_tuple:{}'.format(x_is_tuple))
print('y_is_list:{}'.format(y_is_list))
print('z_is_tensor:{}'.format(z_is_tensor))
```

结果如下：

```text
x_is_tuple:True
y_is_list:True
z_is_tensor:True
```

#### partial

功能：偏函数，固定函数入参。

调用：`partial(func, arg, ...)`

入参：

- `func` -- 函数。

- `arg` -- 一个或多个要固定的参数，支持位置参数和键值对传参。

返回值：返回某些入参固定了值的函数。

示例如下：

```python
from mindspore import ms_function, ops

def add(x, y):
    return x + y

@ms_function()
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

#### map

功能：根据提供的函数对一个或者多个序列做映射，由映射的结果生成一个新的序列。
如果多个序列中的元素个数不一致，则生成的新序列与最短的那个长度相同。

调用：`map(func, sequence, ...)`

入参：

- `func` -- 函数。

- `sequence` -- 一个或多个序列（`Tuple`或者`List`）。

返回值：返回一个`Tuple`。

示例如下：

```python
from mindspore import ms_function

def add(x, y):
    return x + y

@ms_function()
def test():
    elements_a = (1, 2, 3)
    elements_b = (4, 5, 6)
    ret = map(add, elements_a, elements_b)
    return ret

ret = test()
print('ret:{}'.format(ret))
```

结果如下：

```text
ret: (5, 7, 9)
```

#### zip

功能：将多个序列中对应位置的元素打包成一个个元组，然后由这些元组组成一个新序列，
如果各个序列中的元素个数不一致，则生成的新序列与最短的那个长度相同。

调用：`zip(sequence, ...)`

入参：`sequence` -- 一个或多个序列(`Tuple`或`List`)`。

返回值：返回一个`Tuple`。

示例如下：

```python
from mindspore import ms_function

@ms_function()
def test():
    elements_a = (1, 2, 3)
    elements_b = (4, 5, 6)
    ret = zip(elements_a, elements_b)
    return ret

ret = test()
print('ret:{}'.format(ret))
```

结果如下：

```text
ret:((1, 4), (2, 5), (3, 6))
```

#### range

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
from mindspore import ms_function

@ms_function()
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

#### enumerate

功能：生成一个序列的索引序列，索引序列包含数据和对应下标。

调用：

- `enumerate(sequence, start)`

- `enumerate(sequence)`

入参：

- `sequence` -- 一个序列（`Tuple`、`List`、`Tensor`）。

- `start` -- 下标起始位置，类型为`int`，默认为0。

返回值：返回一个`Tuple`。

示例如下：

```python
from mindspore import ms_function, Tensor
import numpy as np

y = Tensor(np.array([[1, 2], [3, 4], [5, 6]]))

@ms_function()
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

#### super

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
from mindspore import nn

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
```

#### pow

功能：求幂。

调用：`pow(x, y)`

入参：

- `x` -- 底数， `Number`或`Tensor`。

- `y` -- 幂指数， `Number`或`Tensor`。

返回值：返回`x`的`y`次幂，`Number`或`Tensor`。

示例如下：

```python
from mindspore import ms_function, Tensor
import numpy as np

x = Tensor(np.array([1, 2, 3]))
y = Tensor(np.array([1, 2, 3]))

@ms_function()
def test(x, y):
    return pow(x, y)

ret = test(x, y)

print('ret:{}'.format(ret))
```

结果如下：

```text
ret:[ 1  4 27]
```

#### print

功能：用于打印。

调用：`print(arg, ...)`

入参：`arg` -- 要打印的信息(`int` 、`float`、`bool`、`String`或`Tensor`)。
当打印的数据是`int`，`float`或者`bool`时，会将其包成一个`0-D`的tensor打印出来。

返回值：无返回值。

示例如下：

```python
from mindspore import ms_function, Tensor, dtype
import numpy as np

x = Tensor(np.array([1, 2, 3]), dtype.int32)
y = Tensor(3, dtype.int32)

@ms_function()
def test(x, y):
    print(x)
    print(y)
    return x, y

ret = test(x, y)
```

结果如下：

```text
Tensor(shape=[3], dtype=Int32, value= [1 2 3])
3
```

#### filter

功能：根据提供的函数对一个序列的元素做判断，每个元素依次作为参数传入函数中，将返回结果不为0或False的元素组成新的序列。

调用：`filter(func, sequence)`

入参：

- `func` -- 函数。

- `sequence` -- 序列（`Tuple`或`List`）。

返回值：返回一个`Tuple`。

示例如下：

```python
from mindspore import ms_function

def is_odd(x):
    if x % 2:
        return True
    return False

@ms_function()
def test():
    elements = (1, 2, 3, 4, 5)
    ret = filter(is_odd, elements)
    return ret

ret = test()
print('ret:{}'.format(ret))
```

结果如下：

```text
ret:(1, 3, 5)
```

### 函数参数

- 参数默认值：目前不支持默认值设为`Tensor`类型数据，支持`int`、`float`、`bool`、`None`、`str`、`tuple`、`list`、`dict`类型数据。
- 可变参数：支持带可变参数网络的推理和训练。
- 键值对参数：目前不支持带键值对参数的函数求反向。
- 可变键值对参数：目前不支持带可变键值对的函数求反向。

## 网络定义

### 网络入参

整网（最外层网络）入参仅支持`bool`、`int`、`float`、`Tensor`、`None`、`mstype.number(mstype.bool_、mstype.int、mstype.float、mstype.uint)`，以及只包含这些类型对象的`list`或者`tuple`，和`value`值是这些类型的`Dictionary`。

在对整网入参求梯度的时候，会忽略非`Tensor`的入参，只计算`Tensor`入参的梯度。例如整网入参`(x, y, z)`中，`x`和`z`是`Tensor`，`y`是非`Tensor`时，在对整网入参求梯度的时候，只会计算`x`和`z`的梯度，返回`(grad_x, grad_z)`。

如果网络里要使用其他类型，可在初始化网络的时候，传入该类型对象，作为网络属性保存起来，然后在`construct`里使用。
内层调用的网络入参无此限制。

示例如下：

```python
from mindspore import nn, ops, Tensor
import numpy as np

class Net(nn.Cell):
    def __init__(self, flag):
        super(Net, self).__init__()
        self.flag = flag

    def construct(self, x, y, z):
        if self.flag == "ok":
            return x + y + z
        return x - y - z

class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.grad_all = ops.GradOperation(get_all=True)
        self.forward_net = net

    def construct(self, x, y, z):
        return self.grad_all(self.forward_net)(x, y, z)

flag = "ok"
input_x = Tensor(np.ones((2, 3)).astype(np.float32))
input_y = 2
input_z = Tensor(np.ones((2, 3)).astype(np.float32) * 2)

net = Net(flag)
grad_net = GradNet(net)
ret = grad_net(input_x, input_y, input_z)

print('ret:{}'.format(ret))
```

结果如下：

```text
ret:(Tensor(shape=[2, 3], dtype=Float32, value=
[[ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00],
 [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00]]), Tensor(shape=[2, 3], dtype=Float32, value=
[[ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00],
 [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00]]))
```

上面定义的Net网络里，在初始化时传入一个`string` flag，作为网络的属性保存起来，然后在`construct`里使用`self.flag`这个属性。

整网入参`x`和`z`是`Tensor`，`y`是`int`数，`grad_net`在对整网入参`(x, y, z)`求梯度时，会自动忽略`y`的梯度，只计算`x`和`z`的梯度，`ret = (grad_x, grad_z)`。

### 网络实例类型

- 带[@ms_function](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore/mindspore.ms_function.html)装饰器的普通Python函数。

- 继承自[nn.Cell](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.Cell.html)的Cell子类。

### 网络构造组件

| 类别                 | 内容                                                                                                                                                                                                         |
| :------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Cell`实例           | [mindspore/nn/*](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore.nn.html)、自定义[Cell](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.Cell.html)。 |
| `Cell`实例的成员函数 | Cell的construct中可以调用其他类成员函数。                                                                                                                                                                    |
| `dataclass`实例      | 使用@dataclass装饰的类。                                                                                                                                                                                     |
| `Primitive`算子      | [mindspore/ops/operations/*](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore.ops.html)                                                                                              |
| `Composite`算子      | [mindspore/ops/composite/*](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore.ops.html)                                                                                               |
| `constexpr`生成算子  | 使用[@constexpr](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.constexpr.html)生成的值计算算子。                                                                          |
| 函数                 | 自定义Python函数、前文中列举的系统函数。                                                                                                                                                                     |

### 网络使用约束

1. 不允许修改网络的非`Parameter`类型数据成员。

   示例如下：

   ```python
   from mindspore import nn, Tensor, Parameter
   import numpy as np

   class Net(nn.Cell):
       def __init__(self):
           super(Net, self).__init__()
           self.num = 2
           self.par = Parameter(Tensor(np.ones((2, 3, 4))), name="par")

       def construct(self, x, y):
           self.par[0] = y
           self.x = x
           return x + y

   net = Net()
   net(1, 2)
   ```

   上面所定义的网络里，`self.num`不是一个`Parameter`，不允许被修改，而`self.par`是一个`Parameter`，可以被修改。

   结果报错如下：

   ```Text
   TypeError: mindspore/ccsrc/pipeline/jit/parse/parse.cc:1740 HandleAssignClassMember] 'self.x' should be initialized as a 'Parameter' in the '__init__' function before assigning.
   ```

2. 当`construct`函数里，使用未定义的类成员时，不会像Python解释器那样抛出`AttributeError`，而是作为`None`处理。

   示例如下：

   ```python
   from mindspore import nn

   class Net(nn.Cell):
       def __init__(self):
           super(Net, self).__init__()

       def construct(self, x):
           return x + self.y

   net = Net()
   net(1)
   ```

   上面所定义的网络里，`construct`里使用了并未定义的类成员`self.y`，此时会将`self.y`作为`None`处理。

   结果报错如下：

   ```Text
   RuntimeError: mindspore/ccsrc/frontend/operator/composite/multitype_funcgraph.cc:161 GenerateFromTypes] The 'add' operation does not support the type [Int64, kMetaTypeNone]
   ```
