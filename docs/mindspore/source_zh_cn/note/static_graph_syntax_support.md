# 静态图语法支持

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/static_graph_syntax_support.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

在Graph模式下，Python代码并不是由Python解释器去执行，而是将代码编译成静态计算图，然后执行静态计算图。

使用Graph模式有两种方式：一是调用`@jit`装饰器修饰函数或者类的成员方法，所修饰的函数或方法将会被编译成静态计算图；二是设置`ms.set_context(mode=ms.GRAPH_MODE)`，使用`Cell`类并且在`construct`函数中编写执行代码，此时`construct`函数的代码将会被编译成静态计算图。

`jit`使用规则详见[jit API文档](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.jit.html#mindspore.jit)。

`Cell`定义详见[Cell API文档](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html)。

由于语法解析的限制，当前在编译构图时，支持的数据类型、语法以及相关操作并没有完全与Python语法保持一致，部分使用受限。

本文主要介绍，在编译静态图时，支持的数据类型、语法以及相关操作，这些规则仅适用于Graph模式。

## 数据类型

### Python内置数据类型

当前支持的`Python`内置数据类型包括：`Number`、`String`、`List`、`Tuple`和`Dictionary`。

#### Number

支持`int`、`float`、`bool`，不支持complex（复数）。

支持在网络里定义`Number`，即支持语法：`y = 1`、`y = 1.2`、 `y = True`。

当数据为常量时，编译时期可以获取到数值，因此在网络中可以支持强转`Number`的语法：`y = int(x)`、`y = float(x)`、`y = bool(x)`。

#### String

支持在网络里构造`String`，即支持语法`y = "abcd"`。

可以通过str()的方式进行将常量转换成字符串。

例如：

```python
import mindspore as ms

@ms.jit()
def test_str_format():
    x = "{} is zero".format(0)
    return x

x = test_str_format()
print(x)
```

结果如下：

```text
0 is zero
```

#### List

`List` 以及 `Tuple` 是Python中最基本的序列内置类型，`List` 与 `Tuple` 最核心的区别是 `List` 是可以改变的对象，而 `Tuple` 是不可以更改的。 这意味着 `Tuple` 一旦被创建，就不可以在对象地址不变的情况下被更改。而 `List` 则可以通过一系列inplace操作，在不改变对象地址的情况下， 对对象进行修改。 例如：

```python
a = [1, 2, 3, 4]
a_id = id(a)
a.append(5)
a_after_id = id(a)
assert a_id == a_after_id
```

上述示例代码中，通过 `append` 这个inplace语法更改 `List` 对象时候， 其对象的地址并没有被修改。 而 `Tuple` 是不支持这种inplace操作的。 在 `JIT_SYNTAX_LEVEL` 设置为 `COMPATIBLE` 以及 `LAX` 的情况下，静态图模式可以支持部分 `List` 对象的inplace操作。

MindSpore图模式语法扩展了对 `List` 的支持，方便用户使用 `List` 进行网络构建。

- 图模式支持图内创建List

  支持在图模式内创建 `List` 对象，且 `List` 内对象的元素可以包含任意图模式支持的类型，也支持多层嵌套。 例如：

  ```python
  import numpy as np
  import mindspore as ms

  @ms.jit
  def generate_list():
    a = [1, 2, 3, 4]
    b = ["1", "2", "a"]
    c = [ms.Tensor([1]), ms.Tensor([2])]
    d = [np.array([1, 2, 3]), np.array(["1, 2, 3"])]
    d = [a, b, c, d, (4, 5)]
    return d
  ```

  上述示例代码中，所有的 `List` 对象都可以被正常的创建。

- 图模式支持返回List

  在MindSpore2.0版本之前，当图模式返回 `List` 对象时，`List` 会被转换为 `Tuple`。MindSpore2.0版本已经可以支持返回 `List` 对象。例如：

  ```python
  import mindspore as ms

  @ms.jit
  def list_func():
      a = [1, 2, 3, 4]
      return d

  output = list_func()  # output: [1, 2, 3, 4]
  ```

  与图模式内创建 `List` 相同，图模式返回 `List` 对象可以包括任意图模式支持的类型，也支持多层嵌套。

- 图模式支持从全局变量中获取List对象

  图模式可以从全局变量中获取 `List` 对象。例如：

  ```python
  import mindspore as ms

  global_list = [1, 2, 3, 4]

  @ms.jit
  def list_func():
      global_list.reverse()
      return global_list

  output = list_func()  # output: [4, 3, 2, 1]
  assert id(global_list) == id(output)
  ```

  在上述示例中，静态图获取到 `List` 对象，并在原有对象上进行了图模式支持的inplace操作 `list.reverse()`, 并将原有对象返回。可以看到图模式返回的对象与原有的全局变量对象id相同， 即两者为同一对象。 若 `JIT_SYNTAX_LEVEL` 设置为 `STRICT` 选项，则返回的 `List` 对象与全局对象为两个不同的对象。

- 图模式支持以List作为输入

  图模式支持 `List` 作为静态图的输入， 作为输入的 `List` 对象的元素必须为图模式支持的输入类型, 也支持多层嵌套。

  ```python
  import mindspore as ms

  list_input = [1, 2, 3, 4]

  @ms.jit
  def list_func(x):
      return x

  output = list_func()  # output: [1, 2, 3, 4]
  ```

  `List` 作为静态图输入存在两点注意事项：

  1) `List` 作为静态图输入的时候，无论其内部的元素是什么类型， 一律被视为常量。

  2) `List` 作为静态图输入的时候，会对该 `List` 对象进行一次复制，并使用该复制对象进行后续的计算， 因此无法对原输入对象进行inplace操作。 例如：

  ```python
  import mindspore as ms

  list_input = [1, 2, 3, 4]

  @ms.jit
  def list_func(x):
      x.reverse()
      return x

  output = list_func()  # output: [4, 3, 2, 1]  list_input: [1, 2, 3, 4]
  assert id(output) != id(list_input)
  ```

  如上述用例所示，`List` 对象作为图模式输入时无法在原有对象上进行inplace操作。图模式返回的对象与输入的对象id不同，为不同对象。

- 图模式支持List的内置方法

    在 `JIT_SYNTAX_LEVEL` 设置为 `COMPATIBLE` 以及 `LAX` 的情况下，图模式部分 `List` 内置函数支持inplace。在 `JIT_SYNTAX_LEVEL` 为 `STRICT` 的情况下，所有方法均不支持inplace操作。  
    图模式支持的 `List` 内置方法如下表所示：

    | 方法名       | 是否支持inplace （JIT_SYNTAX_LEVEL=COMPATIBLE/LAX）    |  
    | ----------  | ------------      |
    | 索引取值      | 非inplace操作      |
    | 索引赋值      | 不支持             |
    | append      | 不支持             |
    | clear       | 不支持             |
    | extend      | 支持               |
    | pop         | 支持               |
    | reverse     | 支持               |
    | insert      | 支持               |

    `List` 内置方法的详细介绍如下：

    - List索引取值

        基础语法：```element = list_object[index]```。

        基础语义：将 `List` 对象中位于第 `index` 位的元素提取出来（ `index` 从0开始）。支持多层索引取值。

        索引值`index` 支持类型包括 `int`， `Tensor` 和`slice`。其中，`int` 以及 `Tensor` 类型的输入可以支持常量以及变量，`slice`内部数据必须为编译时能够确定的常量。

        示例如下：

        ```python
        import mindspore as ms
        @ms.jit()
        def list_getitem_func():
            x = [[1, 2], 3, 4]
            a = x[0]
            b = x[0][ms.Tensor([1])]
            c = x[1:3:1]
        return a, b, c

        a, b, c = list_getitem_func()
        print('a:{}'.format(a))
        print('b:{}'.format(b))
        print('c:{}'.format(c))
        ```

        结果如下：

        ```text
        a:[1, 2]
        b:2
        c:[3, 4]
        ```

    - List索引赋值

        基础语法：```list_object[index] = target_element```。

        基础语义：将 `List` 对象中位于第 `index` 位的元素赋值为 `target_element`（ `index` 从0开始）。支持多层索引赋值。

        索引值 `index` 支持类型包括 `int`， `Tensor` 和`slice`。 其中， `int` 以及 `Tensor` 类型的输入可以支持常量以及变量，`slice`内部数据必须为编译时能够确定的常量。

        索引赋值对象 `target_element` 支持所有图模式支持的类型。

        目前，`List` 索引赋值不支持inplace, 索引赋值后将会生成一个新的对象。该操作后续将会支持inplace。

        示例如下：

        ```python
        import mindspore as ms
        import numpy as np

        @ms.jit()
        def test_setitem_func():
            x = [[0, 1], 2, 3, 4]
            x[1] = 10
            x[2] = "ok"
            x[3] = (1, 2, 3)
            x[0][1] = 88
            return x

        output = test_index()
        print('output:{}'.format(output))
        ```

        结果如下：

        ```text
        output:[[0, 88], 10, "ok", (1, 2, 3)]
        ```

    - List.append

        基础语法：```list_object.append(target_element)```。

        基础语义：向 `List` 对象 `list_object` 的最后追加元素 `target_element`。

        目前，`List.append` 不支持inplace, 索引赋值后将会生成一个新的对象。该操作后续将会支持inplace。

        示例如下：

        ```python
        import mindspore as ms

        @ms.jit()
        def test_list():
            x = [1, 2, 3]
            x.append(4)
            return x

        x = test_list()
        print('x:{}'.format(x))
        ```

        结果如下：

        ```text
        x:[1, 2, 3, 4]
        ```

    - List.clear

        基础语法：```list_object.clear()```。

        基础语义：向 `List` 对象 `list_object` 的内的对象清空。

        目前，`List.clear` 不支持inplace, 索引赋值后将会生成一个新的对象。该操作后续将会支持inplace。

        示例如下：

        ```python
        import mindspore as ms

        @ms.jit()
        def test_list_clear():
            x = [1, 3, 4]
            x.clear()
            return x

        x = test_list_clear()
        print('x:{}'.format(x))
        ```

        结果如下：

        ```text
        x:[]
        ```

    - List.extend

        基础语法：```list_object.extend(target)```。

        基础语义：向 `List` 对象 `list_object` 的最后依次插入 `target` 内的所有元素。

        `target` 支持的类型为 `Tuple`， `List` 以及 `Tensor`。其中，如果 `target` 类型为 `Tensor` 的情况下，会先将该 `Tensor` 转换为 `List`，在进行插入操作。

        在 `JIT_SYNTAX_LEVEL` 设置为 `COMPATIBLE` 以及 `LAX` 的情况下，`List.extend` 支持inplace操作，函数运行后不生成新的对象。

        示例如下：

        ```python
        import mindspore as ms

        @ms.jit()
        def test_list_extend():
            x1 = [1, 2, 3]
            x1.extends((4, "a"))
            x2 = [1, 2, 3]
            x2.extends(ms.Tensor([4, 5]))
            return x1, x2

        output1, output2 = test_list_extend()
        print('output1:{}'.format(output1))
        print('output2:{}'.format(output2))
        ```

        结果如下：

        ```text
        output1:[1, 2, 3, 4, "a"]
        output2:[1, 2, 3, Tensor(shape=[1], dtype=Int64, value= [4]), Tensor(shape=[1], dtype=Int64, value= [5])]
        ```

    - List.pop

        基础语法：```pop_element = list_object.pop(index=-1)```。

        基础语义：将 `List` 对象 `list_object` 的的第 `index` 个元素从 `list_object` 中删除，并返回该元素。

        `index` 要求必须为常量 `int`, 当 `list_object` 的长度为 `list_obj_size` 时，`index` 的取之范围为： `[-list_obj_size， list_obj_size-1]`。 `index` 为负数代表从后往前的位数。当没有输入 `index` 时， 默认值为-1， 即删除最后一个元素。

        在 `JIT_SYNTAX_LEVEL` 设置为 `COMPATIBLE` 以及 `LAX` 的情况下， `List.pop` 支持inplace操作，函数运行后不生成新的对象。

        ```python
        import mindspore as ms

        @ms.jit()
        def test_list_pop():
            x = [1, 2, 3]
            b = x.pop()
            return b, x

        pop_element, res_list = test_list_extend()
        print('pop_element:{}'.format(pop_element))
        print('res_list:{}'.format(res_list))
        ```

        结果如下：

        ```text
        pop_element:3
        res_list:[1, 2]
        ```

    - List.reverse

        基础语法：```list_object.reverse()```。

        基础语义：将 `List` 对象 `list_object` 的元素顺序倒转。

        在 `JIT_SYNTAX_LEVEL` 设置为 `COMPATIBLE` 以及 `LAX` 的情况下， `List.reverse` 支持inplace操作，函数运行后不生成新的对象。

        示例如下：

        ```python
        import mindspore as ms

        @ms.jit()
        def test_list_reverse():
            x = [1, 2, 3]
            x.reverse()
            return x

        output = test_list_extend()
        print('output:{}'.format(output))
        ```

        结果如下：

        ```text
        output1:[3, 2, 1]
        ```

    - List.insert

        基础语法：```list_object.insert(index, target_obj)```。

        基础语义：将 `target_obj` 插入到 `list_object` 的第 `index` 位。

        `index` 要求必须为常量 `int`。 如果 `list_object` 的长度为 `list_obj_size` 。当 `index < -list_obj_size` 时，插入到 `List` 的第一位。当 `index >= -list_obj_size` 时，插入到 `List` 的最后。 `index` 为负数代表从后往前的位数。

        在 `JIT_SYNTAX_LEVEL` 设置为 `COMPATIBLE` 以及 `LAX` 的情况下， `List.insert` 支持inplace操作，函数运行后不生成新的对象。

        示例如下：

        ```python
        import mindspore as ms

        @ms.jit()
        def test_list_reverse():
            x = [1, 2, 3]
            x.reverse()
            return x

        output = test_list_extend()
        print('output:{}'.format(output))
        ```

        结果如下：

        ```text
        output1:[3, 2, 1]
        ```

#### Tuple

支持在网络里构造`Tuple`，即支持语法`y = (1, 2, 3)`。

关于Tuple取值的引用类型问题与List相同，请见List的相关介绍。

- 支持索引取值

  索引值支持`int`、`slice`、`Tensor`，也支持多层索引取值，即支持语法`data = tuple_x[index0][index1]...`。

  索引值为`Tensor`有如下限制：

    - `tuple`里存放的都是`Cell`，每个`Cell`要在tuple定义之前完成定义，每个`Cell`的入参个数、入参类型和入参`shape`要求一致，每个`Cell`的输出个数、输出类型和输出`shape`也要求一致。

    - 索引`Tensor`是一个`dtype`为`int32`的标量`Tensor`，取值范围在`[-tuple_len, tuple_len)`，`Ascend`后端不支持负数索引。

    - 该语法不支持`if`、`while`、`for`控制流条件为变量的运行分支，仅支持控制流条件为常量。

    - 支持`GPU`和`Ascend`后端。

  `int`、`slice`索引示例如下：

  ```python
  import mindspore as ms
  import numpy as np

  t = ms.Tensor(np.array([1, 2, 3]))

  @ms.jit()
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
  import mindspore as ms
  from mindspore import nn, set_context

  set_context(mode=ms.GRAPH_MODE)

  class Net(nn.Cell):
      def __init__(self):
          super(Net, self).__init__()
          self.relu = nn.ReLU()
          self.softmax = nn.Softmax()
          self.layers = (self.relu, self.softmax)

      def construct(self, x, index):
          ret = self.layers[index](x)
          return ret

  x = ms.Tensor([-1.0], ms.float32)

  net = Net()
  ret = net(x, 0)
  print('ret:{}'.format(ret))
  ```

  结果如下：

  ```text
  ret:[0.]
  ```

#### Dictionary

支持在网络里构造`Dictionary`，即支持语法`y = {"a": 1, "b": 2}`。

当前`key`类型支持`String`、`Number`、常量`Tensor`以及只包含这些类型对象的`Tuple`，`value`类型支持`Number`、`Tuple`、`Tensor`、`List`、`Dictionary`。

- 支持接口

  `keys`：取出`dict`里所有的`key`值，组成`Tuple`返回。

  `values`：取出`dict`里所有的`value`值，组成`Tuple`返回。

  `items`：取出`dict`里每一对`key`和`value`组成的`Tuple`，最终组成`List`返回。

  `get`：`dict.get(key[, value])`返回指定`key`对应的`value`值，如果指定`key`不存在，返回默认值`None`或者设置的默认值`value`。

  `clear`：删除`dict`里所有的元素。

  `has_key`：`dict.has_key(key)`判断`dict`里是否存在指定`key`。

  `update`：`dict1.update(dict2)`把`dict2`中的元素更新到`dict1`中。

  `fromkeys`：`dict.fromkeys(seq([, value]))`用于创建新的`Dictionary`，以序列`seq`中的元素做`Dictionary`的`key`，`value`为所有`key`对应的初始值。

  示例如下：

  ```python
  import mindspore as ms
  import numpy as np

  x = {"a": ms.Tensor(np.array([1, 2, 3])), "b": ms.Tensor(np.array([4, 5, 6])), "c": ms.Tensor(np.array([7, 8, 9]))}

  @ms.jit()
  def test_dict():
      x_keys = x.keys()
      x_values = x.values()
      x_items = x.items()
      value_a = x.get("a")
      check_key = x.has_key("a")
      y = {"a": ms.Tensor(np.array([0, 0, 0]))}
      x.update(y)
      new_dict = x.fromkeys("abcd", 123)
      return x_keys, x_values, x_items, value_a, check_key, x, new_dict

  x_keys, x_values, x_items, value_a, check_key, new_x, new_dict = test_dict()
  print('x_keys:{}'.format(x_keys))
  print('x_values:{}'.format(x_values))
  print('x_items:{}'.format(x_items))
  print('value_a:{}'.format(value_a))
  print('check_key:{}'.format(check_key))
  print('new_x:{}'.format(new_x))
  print('new_dict:{}'.format(new_dict))
  ```

  结果如下：

  ```text
  x_keys:('a', 'b', 'c')
  x_values:(Tensor(shape=[3], dtype=Int64, value= [1, 2, 3]), Tensor(shape=[3], dtype=Int64, value= [4, 5, 6]), Tensor(shape=[3], dtype=Int64, value= [7, 8, 9]))
  x_items:[('a', Tensor(shape=[3], dtype=Int64, value= [1, 2, 3])), ('b', Tensor(shape=[3], dtype=Int64, value= [4, 5, 6])), ('c', Tensor(shape=[3], dtype=Int64, value= [7, 8, 9]))]
  value_a:[1 2 3]
  check_key:True
  new_x:{'a': Tensor(shape=[3], dtype=Int64, value= [0, 0, 0]), 'b': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6]), 'c': Tensor(shape=[3], dtype=Int64, value= [7, 8, 9])}
  new_dict:{'a': 123, 'b': 123, 'c': 123, 'd': 123}
  ```

- 支持索引取值和赋值

  示例如下：

  ```python
  import mindspore as ms
  import numpy as np

  x = {"a": ms.Tensor(np.array([1, 2, 3])), "b": ms.Tensor(np.array([4, 5, 6])), "c": ms.Tensor(np.array([7, 8, 9]))}

  @ms.jit()
  def test_dict():
      y = x["b"]
      x["a"] = (2, 3, 4)
      return x, y

  out1, out2 = test_dict()
  print('out1:{}'.format(out1))
  print('out2:{}'.format(out2))
  ```

  结果如下：

  ```text
  out1:{'a': (2, 3, 4), 'b': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6]), 'c': Tensor(shape=[3], dtype=Int64, value= [7, 8, 9])}
  out2:[4 5 6]
  ```

- 支持计算图返回`Dictionary`

  示例如下：

  ```python
  import mindspore as ms

  @ms.jit()
  def test_dict():
      x = {'a': 'a', 'b': 'b'}
      y = x.get('a')
      z = dict(y=y)
      return z

  out = test_dict()
  print("out:", out)
  ```

  结果如下：

  ```text
  out:{'y': 'a'}
  ```

### MindSpore自定义数据类型

当前MindSpore自定义数据类型包括：`Tensor`、`Primitive`、`Cell`和`Parameter`。

#### Tensor

目前已支持在网络里构造Tensor。

Tensor的属性与接口详见[Tensor API文档](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Tensor.html#mindspore-tensor)。

#### Primitive

当前支持在construct里构造`Primitive`及其子类的实例。

但在调用时，参数只能通过位置参数方式传入，不支持通过键值对方式传入。

示例如下：

```python
import mindspore as ms
from mindspore import nn, ops, Tensor, set_context
import numpy as np

set_context(mode=ms.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        reduce_sum = ops.ReduceSum(True) #支持在construct里构造`Primitive`及其子类的实例
        ret = reduce_sum(x, axis=2)
        return ret

x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
net = Net()
ret = net(x)
print('ret.shape:{}'.format(ret.shape))
```

上面所定义的网络里，reduce_sum(x, axis=2)的参数不支持通过键值对方式传入，只能通过位置参数方式传入，即reduce_sum(x, 2)。

结果报错如下：

```text
TypeError: Only supported positional parameter type for python primitive, but got keyword parameter type.
```

当前不支持在网络调用`Primitive`及其子类相关属性和接口。

当前已定义的`Primitive`详见[Primitive API文档](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Primitive.html#mindspore.ops.Primitive)。

#### Cell

当前支持在网络里构造`Cell`及其子类的实例，即支持语法`cell = Cell(args...)`。

但在调用时，参数只能通过位置参数方式传入，不支持通过键值对方式传入，即不支持在语法`cell = Cell(arg_name=value)`。

当前不支持在网络调用`Cell`及其子类相关属性和接口，除非是在`Cell`自己的`construct`中通过`self`调用。

`Cell`定义详见[Cell API文档](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html)。

#### Parameter

`Parameter`是变量张量，代表在训练网络时，需要被更新的参数。

`Parameter`的定义和使用详见[Parameter API文档](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Parameter.html#mindspore.Parameter)。

## 原型

原型代表编程语言中最紧密绑定的操作。

### 属性引用与属性修改

属性引用是后面带有一个句点加一个名称的原型。以下两种情况的属性引用是允许进行修改的：

- 被修改的属性属于本 `cell` 对象， 即必须为 `self.xxx`。 且该属性在Cell的 `__init__` 函数中完成初始化。 示例如下：

  ```python
  import mindspore as ms
  from mindspore import nn, set_context
  set_context(mode=ms.GRAPH_MODE)

  class Net(nn.Cell):
      def __init__(self):
          super().__init__()
          self.weight = ms.Parameter(ms.Tensor([1]), name="w")
          self.m = 2

      def construct(self, x, y):
          self.weight = x  # 满足条件可以修改
          self.m = 3  # 满足条件可以修改
          # self.a = 2 属性a未在__init__内初始化， 无法进行修改。
          return x

  net = Net()
  ret = net(1, 2)
  print('net.weight:{}'.format(net.weight))
  print('net.m:{}'.format(net.m))
  ```

  结果如下:

  ```text
  net.weight:Parameter (name=w, shape=(1,), dtype=Int64, requires_grad=True)
  net.x:3
  ```

- 被修改属性的对象为全局对象， 示例如下：

  ```python
  import mindspore as ms
  from mindspore import nn, set_context

  set_context(mode=ms.GRAPH_MODE)

  class AssignTarget:
      def __init__(self):
          self.x = 1

  data_obj = AssignTarget()

  @ms.jit
  def test_assign():
      data_obj.x = 10

  test_assign()
  print('data_obj.x:{}'.format(data_obj.x))
  ```

  结果如下:

  ```text
  data_obj.x:10
  ```

### 索引取值

对序列`Tuple`、`List`、`Dictionary`、`Tensor`的索引取值操作(Python称为抽取)。

`Tuple`的索引取值请参考本文的[Tuple](#tuple)章节。

`List`的索引取值请参考本文的[List](#list)章节。

`Dictionary`的索引取值请参考本文的[Dictionary](#dictionary)章节。

`Tensor`的索引取详见[Tensor 索引取值文档](https://www.mindspore.cn/docs/zh-CN/master/note/index_support.html#索引取值)。

### 调用

所谓调用就是附带可能为空的一系列参数来执行一个可调用对象(例如：`Cell`、`Primitive`)。

示例如下：

```python
import mindspore as ms
from mindspore import nn, ops, set_context
import numpy as np

set_context(mode=ms.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = ops.MatMul()

    def construct(self, x, y):
        out = self.matmul(x, y)  # Primitive调用
        return out

x = ms.Tensor(np.ones(shape=[1, 3]), ms.float32)
y = ms.Tensor(np.ones(shape=[3, 4]), ms.float32)
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

### 单目算术运算符

| 单目算术运算符 | 支持类型                                        |
| :------------- | :---------------------------------------------- |
| `+`            | `Number`、`Tensor`，取正值。                    |
| `-`            | `Number`、`Tensor`、`COOTensor`、`CSRTensor`，取负值。 |
| `~`            | `Tensor`， 且其数据类型为`Bool`。成员逐个取反。 |

说明：

- 在Python中`~`操作符对输入的整数按位取反; MindSpore对`~`的功能重新定义为对`Tensor(Bool)`的逻辑取反。

### 二元算术运算符

| 二元算术运算符 | 支持类型                                                     |
| :------------- | :----------------------------------------------------------- |
| `+`            | `Number` + `Number`、`String` + `String`、`Number` + `Tensor`、`Tensor` + `Number`、`Tuple` + `Tensor`、`Tensor` + `Tuple`、`List` + `Tensor`、`Tensor`+`List`、`List`+`List`、`Tensor` + `Tensor`、`Tuple` + `Tuple`、`COOTensor` + `Tensor`、`Tensor` + `COOTensor`、`COOTensor` + `COOTensor`、`CSRTensor` + `CSRTensor`。 |
| `-`            | `Number` - `Number`、`Tensor` - `Tensor`、`Number` - `Tensor`、`Tensor` - `Number`、`Tuple` - `Tensor`、`Tensor` - `Tuple`、`List` - `Tensor`、`Tensor` - `List`、`COOTensor` - `Tensor`、`Tensor` - `COOTensor`、`COOTensor` - `COOTensor`、`CSRTensor` - `CSRTensor`。 |
| `*`            | `Number` \* `Number`、`Tensor` \* `Tensor`、`Number` \* `Tensor`、`Tensor` \* `Number`、`List` \* `Number`、`Number` \* `List`、`Tuple` \* `Number`、`Number` \* `Tuple`、`Tuple` \* `Tensor`、`Tensor` \* `Tuple`、 `List` \* `Tensor`、`Tensor` \* `List`、`COOTensor` \* `Tensor`、`Tensor` \* `COOTensor`、`CSRTensor` \* `Tensor`、`Tensor` \* `CSRTensor`。 |
| `/`            | `Number` / `Number`、`Tensor` / `Tensor`、`Number` / `Tensor`、`Tensor` / `Number`、`Tuple` / `Tensor`、`Tensor` / `Tuple`、`List` / `Tensor`、`Tensor` / `List`、`COOTensor` / `Tensor`、`CSRTensor` / `Tensor`。 |
| `%`            | `Number` % `Number`、`Tensor` % `Tensor`、`Number` % `Tensor`、`Tensor` % `Number`、`Tuple` % `Tensor`、`Tensor` % `Tuple`、`List` % `Tensor`、`Tensor` % `List`。 |
| `**`           | `Number` \*\* `Number`、`Tensor` \*\* `Tensor`、`Number` \*\* `Tensor`、`Tensor` \*\* `Number`、`Tuple` \*\* `Tensor`、`Tensor` \*\* `Tuple`、 `List` \*\* `Tensor`、`Tensor` \*\* `List`。 |
| `//`           | `Number` // `Number`、`Tensor` // `Tensor`、`Number` // `Tensor`、`Tensor` // `Number`、`Tuple` // `Tensor`、`Tensor` // `Tuple`、`List` // `Tensor`、`Tensor` // `List`。 |
| `&`     | `Number` & `Number`、`Tensor` & `Tensor`、`Number` & `Tensor`、`Tensor` & `Number`。                                                                                                                                                                  |
| `∣`      | `Number` &#124; `Number`、`Tensor` &#124; `Tensor`、`Number` &#124; `Tensor`、`Tensor` &#124; `Number`。                                                                                                                                                             |
| `^`     | `Number` ^ `Number`、`Tensor` ^ `Tensor`、`Number` ^ `Tensor`、`Tensor` ^ `Number`。                                                                                                                                                                  |
| `<<`    | `Number` << `Number`。                                                                                                                                                                                                                             |
| `>>`    | `Number` >> `Number`。                                                                                                                                                                                                                             |

限制：

- 当左右操作数都为`Number`类型时，不支持`Float64` 和 `Int32`间的运算。`+`、`-`、`*`、`/`、`%`、`**`、`//` 支持左右操作数的值同时为`Bool`。
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
| `&=`     | `Number` &= `Number`、`Tensor` &= `Tensor`、`Number` &= `Tensor`、`Tensor` &= `Number`。                                                                                                                                                                              |
| `∣=`      | `Number` &#124;= `Number`、`Tensor` &#124;= `Tensor`、`Number` &#124;= `Tensor`、`Tensor` &#124;= `Number`。                                                                                                                                                         |
| `^=`     | `Number` ^= `Number`、`Tensor` ^= `Tensor`、`Number` ^= `Tensor`、`Tensor` ^= `Number`。                                                                                                                                                                              |
| `<<=`    | `Number` <<= `Number`。                                                                                                                                                                                                                                         |
| `>>=`    | `Number` >>= `Number`。                                                                                                                                                                                                                                         |

限制：

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

- `and`、`or`的左操作数必须要能被转换成布尔值。例如：左操作数不能为存在多个元素的Tensor。当`and`、`or`的左操作数是变量Tensor时，右操作数必须也是同类型Tensor且Tensor成员个数只能有一个。在其余情况下，右操作数无要求。

- `and`、`or`的左右操作数存在图模式无法支持的对象（例如：第三方对象以及由图模式不原生支持的语法产生的对象）时，左右操作数需要均为常量。

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

- 如果`cond`不为常量，在不同分支中同一符号被赋予的变量或者常量的数据类型应一致，如果是被赋予变量或者常量数据类型是`Tensor`，则要求`Tensor`的type和shape也应一致。shape一致性约束详见[ShapeJoin规则](https://www.mindspore.cn/tutorials/experts/zh-CN/master/network/control_flow.html#shapejoin规则)。

示例1：

```python
import mindspore as ms

x = ms.Tensor([1, 4], ms.int32)
y = ms.Tensor([0, 3], ms.int32)
m = 1
n = 2

@ms.jit()
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
def test_cond(x, y):
    out = 3
    if (x > y).any():
        out = m
    else:
        out = n
    return out

ret = test_cond(x, y)
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
def test_cond(x, y):
    out = 2
    if (x > y).any():
        out = m
    return out

ret = test_cond(x, y)
print('ret:{}'.format(ret))
```

`if`分支中`out`被赋值的变量或者常量`m`与`out`初始赋值的变量或者常量`init`的数据类型必须一致。

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
import mindspore as ms
import numpy as np

z = ms.Tensor(np.ones((2, 3)))

@ms.jit()
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

- 如果`cond`不为常量，在循环体内外同一符号被赋值的变量或者常量的数据类型应一致，如果是被赋予数据类型`Tensor`，则要求`Tensor`的type和shape也应一致。shape一致性约束详见[ShapeJoin规则](https://www.mindspore.cn/tutorials/experts/zh-CN/master/network/control_flow.html#shapejoin规则)。

- 不支持`while...else...`语句

- 如果`cond`不为常量， 循环体内部不能更新循环体外的`Number`、`List`、`Tuple`类型数据， 不能更改`Tensor`类型数据的shape。

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

用于定义函数。

使用方式：

`def function_name(args): statements...`

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

用于生成函数。

使用方式：`lambda x, y: x + y`

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

### 列表生成式和生成器表达式

支持列表生成式（List Comprehension）和生成器表达式（Generator Expression）。

#### 列表生成式

用于生成列表。

使用方式：参考Python语法说明。

示例如下：

```python
import mindspore as ms

@ms.jit()
def test(x, y):
    l = [x * x for x in range(1, 11) if x % 2 == 0]
    return l

ret = test(1, 5)
print('ret:{}'.format(ret))
```

结果如下：

```text
ret:[4, 16, 36, 64, 100]
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

用于生成列表。

使用方式：同列表生成式。

示例如下：

```python
import mindspore as ms

@ms.jit()
def test(x, y):
    l = (x * x for x in range(1, 11) if x % 2 == 0)
    return l

ret = test(1, 5)
print('ret:{}'.format(ret))
```

结果如下：

```text
ret:[4, 16, 36, 64, 100]
```

使用限制同列表生成式。

### with语句

在图模式下，有限制地支持with语句。with语句要求对象必须有两个魔术方法：`__enter__()`和`__exit__()`。

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

### raise语句

功能：根据提供的错误类型和报错语句抛出异常。

调用：`raise Exception(error message)`

入参：

- `Exception` -- 异常类型。

- `error message` -- 异常信息。

返回值：None。

示例如下：

```python
import mindspore as ms

@ms.jit()
def test(tensor_to_raise):
    if tensor_to_raise > 0:
        raise ValueError(f"input should not be {tensor_to_raise}")

tensor_to_raise = ms.Tensor(1)
ret = test(tensor_to_raise)
```

结果如下：

```text
ValueError: input should not be 1.
```

## 函数

### Python内置函数

当前支持的Python内置函数包括：`int`、`float`、`bool`、`str`、`list`、`tuple`、`getattr`、`hasattr`、`len`、`isinstance`、`all`、`any`、`round`、`max`、`min`、`sum`、`abs`、`partial`、`map`、`range`、`enumerate`、`super`、`pow`、`filter`。图模式下内置函数的使用方法与对应的Python内置函数类似。

#### int

功能：返回一个基于数字或字符串构造的整数对象。

调用：`int(x=0, base=10)`

入参：

- `x` -- 需要被转换为整数的对象，支持类型为`int`、`float`、`bool`、`str`、`Tensor`以及第三方对象（例如`numpy.ndarray`）。

- `base` -- 待转换进制， 只有在`x`为常量`str`的时候， 才可以设置该输入。

返回值：转换后的整数值。

代码用例如下：

```python
import mindspore as ms

@ms.jit
def func():
   a = int(3)
   b = int(3.6)
   c = int('12', 16)
   d = int('0xa', 16)
   e = int('10', 8)
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
a: 3
b: 3
c: 18
d: 10
e: 8
```

#### float

功能：返回一个基于数字或字符串构造的浮点数对象。

调用：`float(x=0)`

入参：`x` -- 需要被转换为浮点数的对象，支持类型为`int`、`float`、`bool`、`str`、`Tensor`以及第三方对象（例如`numpy.ndarray`）。

返回值：转换后的浮点数值。

代码用例如下：

```python
import mindspore as ms

@ms.jit
def func():
   a = float(1)
   b = float(112)
   c = float(-123.6)
   d = float('123')
   return a, b, c, d

a, b, c, d = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
```

输出结果：

```text
a: 1.0
b: 112.0
c: -123.6
d: 123.0
```

#### bool

功能：返回一个基于输入构造的布尔值的对象。

调用：`bool(x=false)`

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
   e = bool(ms.Tensor([10]))
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

#### str

功能：返回一个基于输入构造的字符串的对象。

调用：`str(x='')`

入参：`x` -- 需要被转换为字符串的对象，支持类型为`int`、`float`、`bool`、`str`、`list`、 `tuple`、 `dict`、常量`Tensor`以及第三方对象（例如`numpy.ndarray`）。其中，`list`、 `tuple`以及`dict`中不能含有非常量值。

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

#### tuple

功能：返回一个基于输入构造的元组。

调用：`tuple(x=())`

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
   return a, b, c ,d

a, b, c ,d = func()
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

#### list

功能：返回一个基于输入构造的列表。

调用：`list(x=())`

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

#### getattr

功能：获取对象的属性。

调用：`getattr(x, attr, default)`

入参：

- `x` -- 需要被获取属性的对象，可以为任意的图模式支持类型，不支持第三方库类型。

- `attr` -- 需要获取的属性， 需要为`str`。

- `default` -- 可选参数。若`x`没有`attr`, 则返回`default`, 可以为任意的图模式支持类型，不支持第三方库类型。若未输入`default`，且`x`没有属性`attr`，则会抛出AttributeError。

返回值：目标属性或者`default`。

代码用例如下：

```python
import mindspore as ms

@ms.jit_class
class MSClass1:
  def __init__(self):
    self.num0 = 0

ms_obj = MSClass1()

@ms.jit
def func():
   a = getattr(ms_obj, 'num0')
   b = getattr(ms_obj, 'num1', 2)
   return a, b

a, b = func()
print("a: ", a)
print("b: ", b)
```

输出结果:

```text
a: 0
b: 2
```

在静态图模式下对象的属性可能会和动态图模式下有区别，建议使用`default`输入，或者在使用`getattr`前先使用`hasattr`进行校验。

#### hasattr

功能：判断对象是否具有该属性。

调用：`hasattr(x, attr)`

入参：

- `x` -- 需要被判断是否具有某属性的对象，可以为任意的图模式支持类型，也可以为第三方库类型。

- `attr` -- 属性名， 需要为`str`。

返回值：布尔值， 表示是否具有该属性。

代码用例如下：

```python
import mindspore as ms

@ms.jit_class
class MSClass1:
  def __init__(self):
    self.num0 = 0

ms_obj = MSClass1()

@ms.jit
def func():
   a = hasattr(ms_obj, 'num0')
   b = hasattr(ms_obj, 'num1')
   return a, b

a, b = func()
print("a: ", a)
print("b: ", b)
```

输出结果:

```text
a: True
b: False
```

#### len

功能：求序列的长度。

调用：`len(sequence)`

入参：`sequence` -- `Tuple`、`List`、`Dictionary`、`Tensor`以及第三方对象（例如numpy.ndarray）。

返回值：序列的长度，类型为`int`。当入参是`Tensor`时，返回的是`Tensor`第零维的长度。

示例如下：

```python
import mindspore as ms
import numpy as np

z = ms.Tensor(np.ones((6, 4, 5)))

@ms.jit()
def test():
    x = (2, 3, 4)
    y = [2, 3, 4]
    d = {"a": 2, "b": 3}
    n = np.array([1, 2, 3, 4])
    x_len = len(x)
    y_len = len(y)
    d_len = len(d)
    z_len = len(z)
    n_len = len(n)
    return x_len, y_len, d_len, z_len, n_len

x_len, y_len, d_len, z_len, n_len = test()
print('x_len:{}'.format(x_len))
print('y_len:{}'.format(y_len))
print('d_len:{}'.format(d_len))
print('z_len:{}'.format(z_len))
print('n_len:{}'.format(n_len))
```

结果如下：

```text
x_len:3
y_len:3
d_len:2
z_len:6
z_len:4
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
import mindspore as ms
import numpy as np

z = ms.Tensor(np.ones((6, 4, 5)))

@ms.jit()
def test():
    x = (2, 3, 4)
    y = [2, 3, 4]
    x_is_tuple = isinstance(x, tuple)
    y_is_list = isinstance(y, list)
    z_is_tensor = isinstance(z, ms.Tensor)
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

#### all

功能：判断输入中的元素是否均为真值。

调用：`all(x)`

入参：`x` -- 可迭代对象，支持类型包括`tuple`、`list`、`dict`、`Tensor`以及第三方对象（例如`numpy.ndarray`）。

返回值：布尔值， 表示输入中的元素是否均为真值。

代码用例如下：

```python
import mindspore as ms

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
   return a, b, c, d, e, f, g, h

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
a: True
b: False
c: False
d: True
e: False
f: False
g: True
h: True
```

#### any

功能：判断输入中的元素是存在为真值。

调用：`any(x)`

入参：`x` -- 可迭代对象，支持类型包括`tuple`、`list`、`dict`、`Tensor`以及第三方对象（例如`numpy.ndarray`）。

返回值：布尔值，表示输入中的元素是否存在真值。

代码用例如下：

```python
import mindspore as ms

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
   return a, b, c, d, e, f, g, h

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
a: True
b: True
c: False
d: True
e: True
f: False
g: False
h: False
```

#### round

功能：返回输入的四舍五入。

调用：`round(x, digit=0)`

入参：

- `x` -- 需要四舍五入的值，有效类型为 `int`、`float`、`bool`、`Tensor` 以及定义了魔术方法 `__round__()` 第三方对象。

- `digit` -- 表示进行四舍五入的小数点位数，默认值为0，支持 `int` 类型以及 `None`。 若 `x` 为 `Tensor` 类型， 则不支持输入 `digit`。

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

#### max

功能：返回最大值。

调用：`max(*data)`

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

#### min

功能：返回最小值。

调用：`min(*data)`

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

#### sum

功能：对输入序列进行求和计算。

调用：`sum(x, n=0)`

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

#### abs

功能：返回绝对值，使用方法与python的`abs()`一致。

调用：`abs(x)`

入参： - `x` -- 有效类型为`int`、`float`、`bool`、`complex`、`Tensor`以及第三方对象（例如`numpy.ndarray`）。

返回值：绝对值。

代码用例如下：

```python
import mindspore as ms

@ms.jit
def func():
   a = abs(-45)
   b = abs(100.12)
   return a, b

a, b = func()
print("a: ", a)
print("b: {:.2f}".format(b))
```

输出结果：

```text
a: 45
b: 100.12
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
import mindspore as ms

def add(x, y):
    return x + y

@ms.jit()
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

入参：`sequence` -- 一个或多个序列(`Tuple`或`List`)。

返回值：返回一个`Tuple`。

示例如下：

```python
import mindspore as ms

@ms.jit()
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

#### pow

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

#### print

功能：用于打印。

调用：`print(arg, ...)`

入参：`arg` -- 要打印的信息(`int` 、`float`、`bool`、`String`或`Tensor`)。
当打印的数据是`int`，`float`或者`bool`时，会将其包成一个`0-D`的tensor打印出来。

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

#### filter

功能：根据提供的函数对一个序列的元素做判断，每个元素依次作为参数传入函数中，将返回结果不为0或False的元素组成新的序列。

调用：`filter(func, sequence)`

入参：

- `func` -- 函数。

- `sequence` -- 序列（`Tuple`或`List`）。

返回值：返回一个`Tuple`。

示例如下：

```python
import mindspore as ms

def is_odd(x):
    if x % 2:
        return True
    return False

@ms.jit()
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
import mindspore as ms
from mindspore import nn, ops, set_context
import numpy as np

set_context(mode=ms.GRAPH_MODE)

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
        self.forward_net = net

    def construct(self, x, y, z):
        return ms.grad(self.forward_net, grad_position=(0, 1, 2))(x, y, z)

flag = "ok"
input_x = ms.Tensor(np.ones((2, 3)).astype(np.float32))
input_y = 2
input_z = ms.Tensor(np.ones((2, 3)).astype(np.float32) * 2)

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

- 带[@jit](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.jit.html)装饰器的普通Python函数。

- 继承自[nn.Cell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html)的Cell子类。

### 网络构造组件

| 类别                 | 内容                                                                                                                                                                                                         |
| :------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Cell`实例           | [mindspore/nn/*](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.nn.html)、自定义[Cell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html)。 |
| `Cell`实例的成员函数 | Cell的construct中可以调用其他类成员函数。                                                                                                                                                                    |
| `jit_class`实例      | 使用[@jit_class](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.jit_class.html)装饰的类。                                                                                                                                                                                     |
| `Primitive`算子      | [mindspore/ops/operations/*](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.ops.primitive.html)                                                                                              |
| `Composite`算子      | [mindspore/ops/composite/*](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.ops.primitive.html)                                                                                               |
| `constexpr`生成算子  | 使用[@constexpr](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.constexpr.html)生成的值计算算子。                                                                          |
| 函数                 | 自定义Python函数、前文中列举的系统函数。                                                                                                                                                                     |

### 网络使用约束

1. 当`construct`函数里，使用未定义的类成员时，将抛出`AttributeError`异常。

   示例如下：

   ```python
   import mindspore as ms
   from mindspore import nn, set_context

   set_context(mode=ms.GRAPH_MODE)

   class Net(nn.Cell):
       def __init__(self):
           super(Net, self).__init__()

       def construct(self, x):
           return x + self.y

   net = Net()
   net(1)
   ```

   结果报错如下：

   ```Text
   AttributeError: External object has no attribute y
   ```

2. `nn.Cell`不支持`classmethod`修饰的类方法。
