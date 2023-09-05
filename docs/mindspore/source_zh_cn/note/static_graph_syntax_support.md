# 静态图语法支持

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/static_graph_syntax_support.md)

## 概述

在Graph模式下，Python代码并不是由Python解释器去执行，而是将代码编译成静态计算图，然后执行静态计算图。

在静态图模式下，MindSpore通过源码转换的方式，将Python的源码转换成中间表达IR（Intermediate Representation），并在此基础上对IR图进行优化，最终在硬件设备上执行优化后的图。MindSpore使用基于图表示的函数式IR，称为MindIR，详情可参考[中间表示MindIR](https://www.mindspore.cn/docs/zh-CN/master/design/all_scenarios.html#中间表示mindir)。

MindSpore的静态图执行过程实际包含两步，对应静态图的Define和Run阶段，但在实际使用中，在实例化的Cell对象被调用时用户并不会分别感知到这两阶段，MindSpore将两阶段均封装在Cell的`__call__`方法中，因此实际调用过程为：

`model(inputs) = model.compile(inputs) + model.construct(inputs)`，其中`model`为实例化Cell对象。

使用Graph模式有两种方式：一是调用`@jit`装饰器修饰函数或者类的成员方法，所修饰的函数或方法将会被编译成静态计算图。`jit`使用规则详见[jit API文档](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.jit.html#mindspore.jit)。二是设置`ms.set_context(mode=ms.GRAPH_MODE)`，使用`Cell`类并且在`construct`函数中编写执行代码，此时`construct`函数的代码将会被编译成静态计算图。`Cell`定义详见[Cell API文档](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html)。

由于语法解析的限制，当前在编译构图时，支持的数据类型、语法以及相关操作并没有完全与Python语法保持一致，部分使用受限。借鉴传统JIT编译的思路，从图模式的角度考虑动静图的统一，扩展图模式的语法能力，使得静态图提供接近动态图的语法使用体验，从而实现动静统一。为了便于用户选择是否扩展静态图语法，提供了JIT语法支持级别选项`jit_syntax_level`，其值必须在[STRICT，LAX]范围内，选择`STRICT`则认为使用基础语法，不扩展静态图语法。默认值为`LAX`，更多请参考本文的[扩展语法（LAX级别）](#扩展语法lax级别)章节。全部级别都支持所有后端。

- STRICT: 仅支持基础语法，且执行性能最佳。可用于MindIR导入导出。
- LAX: 支持更多复杂语法，最大程度地兼容Python所有语法。由于存在可能无法导出的语法，不能用于MindIR导入导出。

本文主要介绍，在编译静态图时，支持的数据类型、语法以及相关操作，这些规则仅适用于Graph模式。

## 基础语法（STRICT级别）

### 静态图内的常量与变量

在静态图中，常量与变量是理解静态图语法的一个重要概念，很多语法在常量输入和变量输入情况下支持的方法与程度是不同的。因此，在介绍静态图具体支持的语法之前，本小节先会对静态图中常量与变量的概念进行说明。

在静态图模式下，一段程序的运行会被分为编译期以及执行期。 在编译期，程序会被编译成一张中间表示图，并且程序不会真正的执行，而是通过抽象推导的方式对中间表示进行静态解析。这使得在编译期时，我们无法保证能获取到所有中间表示中节点的值。 常量和变量也就是通过能否能在编译器获取到其真实值来区分的。

- 常量： 编译期内可以获取到值的量。
- 变量： 编译期内无法获取到值的量。

在有些情况下，一个量是常量还是变量是很难判断的，此时我们可以通过`ops.isconstant`来判断其是否为常量。例如：

```python
from mindspore import Tensor, jit, ops

a = Tensor([1])

@jit
def foo(a):
    b = Tensor([2])
    m = ops.isconstant(a)
    n = ops.isconstant(b)
    return m, n
```

上述代码中，`a`为变量，因此`m`为`False`。`b`为常量，因此`n`为`True`。

#### 常量产生场景

- 作为图模式输入的标量，列表以及元组均为常量（在不使用mutable接口的情况下）。例如：

  ```python
  from mindspore import Tensor, jit

  a = 1
  b = [Tensor([1]), Tensor([2])]
  c = ["a", "b", "c"]

  @jit
  def foo(a, b, c):
      return a, b, c
  ```

  上述代码中，输入`a`，`b`，`c`均为常量。

- 图模式内生成的标量或者Tensor为常量。例如：

  ```python
  from mindspore import jit, Tensor

  @jit
  def foo():
      a = 1
      b = "2"
      c = Tensor([1, 2, 3])
      return a, b, c
  ```

  上述代码中， `a`，`b`，`c`均为常量。

- 常量运算得到的结果为常量。例如：

  ```python
  from mindspore import jit, Tensor

  @jit
  def foo():
      a = Tensor([1, 2, 3])
      b = Tensor([1, 1, 1])
      c = a + b
      return c
  ```

  上述代码中，`a`、`b`均为图模式内产生的Tensor为常量，因此其计算得到的结果也是常量。但如果其中之一为变量时，其返回值也会为变量。

#### 变量产生场景

- 所有mutable接口的返回值均为变量(无论是在图外使用mutable还是在图内使用)。例如：

  ```python
  from mindspore import Tensor, jit
  from mindspore.common import mutable

  a = mutable([Tensor([1]), Tensor([2])])

  @jit
  def foo(a):
      b = mutable(Tensor([3]))
      c = mutable((Tensor([1]), Tensor([2])))
      return a, b, c
  ```

  上述代码中，`a`是在图外调用mutable接口的，`b`和`c`是在图内调用mutable接口生成的，`a`、`b`、`c`均为变量。

- 作为静态图的输入的Tensor都是变量。例如：

  ```python
  from mindspore import Tensor, jit

  a = Tensor([1])
  b = (Tensor([1]), Tensor([2]))

  @jit
  def foo(a, b):
      return a, b
  ```

  上述代码中，`a`是作为图模式输入的Tensor，因此其为变量。但`b`是作为图模式输入的元组，非Tensor类型，即使其内部的元素均为Tensor，`b`也是常量。

- 通过变量计算得到的是变量。

  如果一个量是算子的输出，那么其多数情况下为常量。例如：

  ```python
  from mindspore import Tensor, jit, ops

  a = Tensor([1])
  b = Tensor([2])

  @jit
  def foo(a, b):
      c = a + b
      return c
  ```

  在这种情况下，`c`是`a`和`b`计算来的结果，且用来计算的输入`a`、`b`均为变量，因此`c`也是变量。

### 数据类型

#### Python内置数据类型

当前支持的`Python`内置数据类型包括：`Number`、`String`、`List`、`Tuple`和`Dictionary`。

##### Number

支持`int`（整型）、`float`（浮点型）、`bool`（布尔类型），不支持`complex`（复数）。

支持在网络里定义`Number`，即支持语法：`y = 1`、`y = 1.2`、`y = True`。

当数据为常量时，编译时期可以获取到数值，在网络中可以支持强转`Number`的语法：`y = int(x)`、`y = float(x)`、`y = bool(x)`。
当数据为变量时，即需要在运行时期才可以获取到数值，也支持使用int()，float()，bool()等内置函数[Python内置函数](#python内置函数)进行数据类型的转换。例如：

```python
from mindspore import Tensor, jit

@jit
def foo(x):
  out1 = int(11.1)
  out2 = int(Tensor([10]))
  out3 = int(x.asnumpy())
  return out1, out2, out3

res = foo(Tensor(2))
print("res[0]:", res[0])
print("res[1]:", res[1])
print("res[2]:", res[2])
```

结果如下：

```text
res[0]: 11
res[1]: 10
res[2]: 2
```

支持返回Number类型。例如：

```python
import mindspore as ms

@ms.jit
def test_return_scalar(x, y):
    return x + y

res = test_return_scalar(ms.mutable(1), ms.mutable(2))
print(res)
```

```text
3
```

##### String

支持在网络里构造`String`，即支持使用引号（`'`或`"`）来创建字符串，如`x = 'abcd'`或`y = "efgh"`。可以通过`str()`的方式进行将常量转换成字符串。支持对字符串连接，截取，以及使用成员运算符（`in`或`not in`）判断字符串是否包含指定的字符。支持格式化字符串的输出，将一个值插入到一个有字符串格式符`%s`的字符串中。支持在常量场景下使用格式化字符串函数`str.format()`。

例如：

```python
from mindspore import jit

@jit
def foo():
  var1 = 'Hello!'
  var2 = "MindSpore"
  var3 = str(123)
  var4 = "{} is {}".format("string", var3)
  return var1[0], var2[4:9], var1 + var2, var2 * 2, "H" in var1, "My name is %s!" % var2, var4

res = foo()
print("res:", res)
```

结果如下：

```text
res: ('H', 'Spore', 'Hello!MindSpore', 'MindSporeMindSpore', True, 'My name is MindSpore!', 'string is 123')
```

##### List

在`JIT_SYNTAX_LEVEL`设置为`LAX`的情况下，静态图模式可以支持部分`List`对象的inplace操作，具体介绍详见[支持列表就地修改操作](#支持列表就地修改操作)章节。

`List`的基础使用场景如下：

- 图模式支持图内创建`List`。

  支持在图模式内创建`List`对象，且`List`内对象的元素可以包含任意图模式支持的类型，也支持多层嵌套。例如：

  ```python
  import numpy as np
  import mindspore as ms

  @ms.jit
  def generate_list():
    a = [1, 2, 3, 4]
    b = ["1", "2", "a"]
    c = [ms.Tensor([1]), ms.Tensor([2])]
    d = [a, b, c, (4, 5)]
    return d
  ```

  上述示例代码中，所有的`List`对象都可以被正常的创建。

- 图模式支持返回`List`。

  在MindSpore2.0版本之前，当图模式返回`List` 对象时，`List`会被转换为`Tuple`。MindSpore2.0版本已经可以支持返回`List`对象。例如：

  ```python
  import mindspore as ms

  @ms.jit
  def list_func():
      a = [1, 2, 3, 4]
      return a

  output = list_func()  # output: [1, 2, 3, 4]
  ```

  与图模式内创建`List` 相同，图模式返回`List`对象可以包括任意图模式支持的类型，也支持多层嵌套。

- 图模式支持从全局变量中获取`List`对象。

  ```python
  import mindspore as ms

  global_list = [1, 2, 3, 4]

  @ms.jit
  def list_func():
      global_list.reverse()
      return global_list

  output = list_func()  # output: [4, 3, 2, 1]
  ```

  需要注意的是，在基础场景下图模式返回的列表与全局变量的列表不是同一个对象，当`JIT_SYNTAX_LEVEL`设置为`LAX`时，返回的对象与全局对象为统一对象。

- 图模式支持以`List`作为输入。

  图模式支持`List`作为静态图的输入，作为输入的`List`对象的元素必须为图模式支持的输入类型，也支持多层嵌套。

  ```python
  import mindspore as ms

  list_input = [1, 2, 3, 4]

  @ms.jit
  def list_func(x):
      return x

  output = list_func()  # output: [1, 2, 3, 4]
  ```

  需要注意的是，`List`作为静态图输入时，无论其内部的元素是什么类型，一律被视为常量。

- 图模式支持List的内置方法。

    `List` 内置方法的详细介绍如下：

    - List索引取值

        基础语法：```element = list_object[index]```。

        基础语义：将`List`对象中位于第`index`位的元素提取出来（`index`从0开始）。支持多层索引取值。

        索引值`index`支持类型包括`int`，`Tensor`和`slice`。其中，`int`以及`Tensor`类型的输入可以支持常量以及变量，`slice`内部数据必须为编译时能够确定的常量。

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

        基础语义：将`List`对象中位于第`index`位的元素赋值为 `target_element`（`index`从0开始）。支持多层索引赋值。

        索引值`index`支持类型包括`int`，`Tensor`和`slice`。其中，`int` 以及`Tensor`类型的输入可以支持常量以及变量，`slice`内部数据必须为编译时能够确定的常量。

        索引赋值对象`target_element`支持所有图模式支持的数据类型。

        目前，`List`索引赋值不支持inplace操作, 索引赋值后将会生成一个新的对象。该操作后续将会支持inplace操作。

        示例如下：

        ```python
        import mindspore as ms

        @ms.jit()
        def test_setitem_func():
            x = [[0, 1], 2, 3, 4]
            x[1] = 10
            x[2] = "ok"
            x[3] = (1, 2, 3)
            x[0][1] = 88
            return x

        output = test_setitem_func()
        print('output:{}'.format(output))
        ```

        结果如下：

        ```text
        output:[[0, 88], 10, 'ok', (1, 2, 3)]
        ```

    - List.append

        基础语法：```list_object.append(target_element)```。

        基础语义：向`List`对象`list_object`的最后追加元素`target_element`。

        目前，`List.append`不支持inplace操作, 索引赋值后将会生成一个新的对象。该操作后续将会支持inplace操作。

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

        基础语义：清空`List`对象 `list_object`中包含的元素。

        目前，`List.clear`不支持inplace, 索引赋值后将会生成一个新的对象。该操作后续将会支持inplace。

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

        基础语义：向`List`对象`list_object`的最后依次插入`target`内的所有元素。

        `target`支持的类型为`Tuple`，`List`以及`Tensor`。其中，如果`target`类型为`Tensor`的情况下，会先将该`Tensor`转换为`List`，再进行插入操作。

        示例如下：

        ```python
        import mindspore as ms

        @ms.jit()
        def test_list_extend():
            x1 = [1, 2, 3]
            x1.extend((4, "a"))
            x2 = [1, 2, 3]
            x2.extend(ms.Tensor([4, 5]))
            return x1, x2

        output1, output2 = test_list_extend()
        print('output1:{}'.format(output1))
        print('output2:{}'.format(output2))
        ```

        结果如下：

        ```text
        output1:[1, 2, 3, 4, 'a']
        output2:[1, 2, 3, Tensor(shape=[1], dtype=Int64, value= [4]), Tensor(shape=[1], dtype=Int64, value= [5])]
        ```

    - List.pop

        基础语法：```pop_element = list_object.pop(index=-1)```。

        基础语义：将`List`对象`list_object` 的第`index`个元素从`list_object`中删除，并返回该元素。

        `index` 要求必须为常量`int`, 当`list_object`的长度为`list_obj_size`时，`index`的取值范围为：`[-list_obj_size，list_obj_size-1]`。`index`为负数，代表从后往前的位数。当没有输入`index`时，默认值为-1，即删除最后一个元素。

        ```python
        import mindspore as ms

        @ms.jit()
        def test_list_pop():
            x = [1, 2, 3]
            b = x.pop()
            return b, x

        pop_element, res_list = test_list_pop()
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

        基础语义：将`List`对象`list_object`的元素顺序倒转。

        示例如下：

        ```python
        import mindspore as ms

        @ms.jit()
        def test_list_reverse():
            x = [1, 2, 3]
            x.reverse()
            return x

        output = test_list_reverse()
        print('output:{}'.format(output))
        ```

        结果如下：

        ```text
        output1:[3, 2, 1]
        ```

    - List.insert

        基础语法：```list_object.insert(index, target_obj)```。

        基础语义：将`target_obj`插入到`list_object`的第`index`位。

        `index`要求必须为常量`int`。如果`list_object`的长度为`list_obj_size`。当`index < -list_obj_size`时，插入到`List`的第一位。当`index >= -list_obj_size`时，插入到`List`的最后。`index`为负数代表从后往前的位数。

        示例如下：

        ```python
        import mindspore as ms

        @ms.jit()
        def test_list_insert():
            x = [1, 2, 3]
            x.insert(3, 4)
            return x

        output = test_list_insert()
        print('output:{}'.format(output))
        ```

        结果如下：

        ```text
        output:[1, 2, 3, 4]
        ```

##### Tuple

支持在网络里构造元组`Tuple`，使用小括号包含元素，即支持语法`y = (1, 2, 3)`。元组`Tuple`的元素不能修改，但支持索引访问元组`Tuple`中的元素，支持对元组进行连接组合。

- 支持索引取值。

  支持使用方括号加下标索引的形式来访问元组`Tuple`中的元素，索引值支持`int`、`slice`、`Tensor`，也支持多层索引取值，即支持语法`data = tuple_x[index0][index1]...`。

  索引值为`Tensor`有如下限制：

    - `Tuple`里存放的都是`Cell`，每个`Cell`要在`Tuple`定义之前完成定义，每个`Cell`的入参个数、入参类型和入参`shape`要求一致，每个`Cell`的输出个数、输出类型和输出`shape`也要求一致。

    - 索引`Tensor`是一个`dtype`为`int32`的标量`Tensor`，取值范围在`[-tuple_len, tuple_len)`，`Ascend`后端不支持负数索引。

    - 支持`CPU`、`GPU`和`Ascend`后端。

  `int`、`slice`索引示例如下：

  ```python
  import numpy as np
  import mindspore as ms

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

- 支持连接组合。

  与字符串`String`类似，元组支持使用`+`和`*`进行组合，得到一个新的元组`Tuple`，例如：

  ```python
  import mindspore as ms

  @ms.jit()
  def test_index():
      x = (1, 2, 3)
      y = (4, 5, 6)
      return x + y, x * 2

  out1, out2 = test_index()
  print('out1:{}'.format(out1))
  print('out2:{}'.format(out2))
  ```

  结果如下：

  ```text
  out1:(1, 2, 3, 4, 5, 6)
  out2:(1, 2, 3, 1, 2, 3)
  ```

##### Dictionary

支持在网络里构造字典`Dictionary`，每个键值`key:value`用冒号`:`分割，每个键值对之间用逗号`,`分割，整个字典使用大括号`{}`包含键值对，即支持语法`y = {"a": 1, "b": 2}`。

键`key`是唯一的，如果字典中存在多个相同的`key`，则重复的`key`以最后一个作为最终结果；而值`value`可以不是唯一的。键`key`需要保证是不可变的。当前键`key`支持`String`、`Number`、常量`Tensor`以及只包含这些类型对象的`Tuple`；值`value`支持`Number`、`Tuple`、`Tensor`、`List`、`Dictionary`和`None`。

- 支持接口。

  `keys`：取出`dict`里所有的`key`值，组成`Tuple`返回。

  `values`：取出`dict`里所有的`value`值，组成`Tuple`返回。

  `items`：取出`dict`里每一对`key`和`value`组成的`Tuple`，最终组成`List`返回。

  `get`：`dict.get(key[, value])`返回指定`key`对应的`value`值，如果指定`key`不存在，返回默认值`None`或者设置的默认值`value`。

  `clear`：删除`dict`里所有的元素。

  `has_key`：`dict.has_key(key)`判断`dict`里是否存在指定`key`。

  `update`：`dict1.update(dict2)`把`dict2`中的元素更新到`dict1`中。

  `fromkeys`：`dict.fromkeys(seq([, value]))`用于创建新的`Dictionary`，以序列`seq`中的元素做`Dictionary`的`key`，`value`为所有`key`对应的初始值。

  示例如下，其中返回值中的`x`和`new_dict`是一个`Dictionary`，在图模式JIT语法支持级别选项为LAX下扩展支持，更多Dictionary的高阶使用请参考本文的[支持Dictionary的高阶用法](#支持dictionary的高阶用法)章节。

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

#### MindSpore自定义数据类型

当前MindSpore自定义数据类型包括：`Tensor`、`Primitive`、`Cell`和`Parameter`。

##### Tensor

Tensor的属性与接口详见[Tensor API文档](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Tensor.html#mindspore-tensor)。

支持在静态图模式下创建和使用Tensor。

```python
import mindspore.nn as nn
import mindspore as ms

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()

    def construct(self):
        return ms.Tensor(1, dtype=ms.int32)

ms.set_context(mode=ms.GRAPH_MODE)

net = Net()
print(net())
```

```text
1
```

上述例子，使用了`Tensor`类接口创建`Tensor`，有些情况下可能会需要创建运行时的`Tensor`，即在编译时期获取不到值的`Tensor`数据，即需要在图模式JIT语法支持级别选项为LAX下扩展支持。此时既可以采用上述类`ms.Tensor`接口来创建`Tensor`，也可以采用 [tensor函数接口](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.tensor.html#mindspore.tensor)来创建`Tensor`，代码用例如下。

其与`Tensor`类接口的区别在于，其内部增加了[Annotation Type](#annotation-type)的标记，在设定了`dtype`的情况下，在类型推导阶段能够指定其输出的`Tensor dtype`，从而避免`AnyType`的产生。在需要动态地创建运行时的`Tensor`时，我们推荐使用这种方式来进行`Tensor`创建，并希望用户能够传入期待的dtype类型。以此避免`AnyType`的产生。

```python
import mindspore as ms
import mindspore.nn as nn

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()

    @ms.jit
    def construct(self, x):
        return ms.tensor(x.asnumpy(), dtype=ms.float32)

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
x = ms.Tensor(1, dtype=ms.int32)
print(net(x))
```

```text
1.0
```

##### Primitive

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

##### Cell

当前支持在网络里构造`Cell`及其子类的实例，即支持语法`cell = Cell(args...)`。

但在调用时，参数只能通过位置参数方式传入，不支持通过键值对方式传入，即不支持在语法`cell = Cell(arg_name=value)`。

当前不支持在网络调用`Cell`及其子类相关属性和接口，除非是在`Cell`自己的`construct`中通过`self`调用。

`Cell`定义详见[Cell API文档](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html)。

##### Parameter

`Parameter`是变量张量，代表在训练网络时，需要被更新的参数。

`Parameter`的定义和使用详见[Parameter API文档](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Parameter.html#mindspore.Parameter)。

### 运算符

算术运算符和赋值运算符支持`Number`和`Tensor`运算，也支持不同`dtype`的`Tensor`运算。详见[运算符](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax/operators.html)。

### 原型

原型代表编程语言中最紧密绑定的操作。

#### 属性引用与修改

属性引用是后面带有一个句点加一个名称的原型。

在MindSpore的Cell 实例中使用属性引用作为左值需满足如下要求：

- 被修改的属性属于本`cell`对象，即必须为`self.xxx`。
- 该属性在Cell的`__init__`函数中完成初始化且其为Parameter类型。

在JIT语法支持级别选项为`LAX`时，可以支持更多情况的属性修改，具体详见[支持属性设置与修改](#支持属性设置与修改)。

示例如下：

```python
import mindspore as ms
from mindspore import nn, set_context

set_context(mode=ms.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight = ms.Parameter(ms.Tensor(3, ms.float32), name="w")
        self.m = 2

    def construct(self, x, y):
        self.weight = x  # 满足条件可以修改
        # self.m = 3     # self.m 非Parameter类型禁止修改
        # y.weight = x   # y不是self，禁止修改
        return x

net = Net()
ret = net(1, 2)
print('ret:{}'.format(ret))
```

结果如下:

```text
ret:1
```

#### 索引取值

对序列`Tuple`、`List`、`Dictionary`、`Tensor`的索引取值操作(Python称为抽取)。

`Tuple`的索引取值请参考本文的[Tuple](#tuple)章节。

`List`的索引取值请参考本文的[List](#list)章节。

`Dictionary`的索引取值请参考本文的[Dictionary](#dictionary)章节。

`Tensor`的索引取详见[Tensor 索引取值文档](https://www.mindspore.cn/docs/zh-CN/master/note/index_support.html#索引取值)。

#### 调用

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

### 语句

当前静态图模式支持部分Python语句，包括raise语句、assert语句、pass语句、return语句、break语句、continue语句、if语句、for语句、while语句、with语句、列表生成式、生成器表达式、函数定义语句等，详见[Python语句](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax/statements.html)。

### Python内置函数

当前静态图模式支持部分Python内置函数，其使用方法与对应的Python内置函数类似，详见[Python内置函数](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax/python_builtin_functions.html)。

### 网络定义

#### 网络入参

整网（最外层网络）入参仅支持`bool`、`int`、`float`、`Tensor`、`None`、`mstype.number(mstype.bool_、mstype.int、mstype.float、mstype.uint)`，以及只包含这些类型对象的`list`或者`tuple`，和`value`值是这些类型的`Dictionary`。

如果网络里要使用其他类型，可在初始化网络的时候，传入该类型对象，作为网络属性保存起来，然后在`construct`里使用。内层调用的网络入参无此限制。

示例如下。定义的Net网络里，在初始化时传入一个`string`类型的flag参数，作为网络属性`self.flag`，然后在`construct`里使用`self.flag`这个属性。

```python
import mindspore as ms
from mindspore import nn

ms.set_context(mode=ms.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self, flag):
        super(Net, self).__init__()
        self.flag = flag

    def construct(self, x):
        if self.flag == "ok":
            return x + 1
        return x - 1

flag = "ok"
net = Net(flag)
ret = net(ms.Tensor([5]))
print('ret:{}'.format(ret))
```

结果如下：

```text
ret:[6]
```

在对整网入参求梯度的时候，会忽略非`Tensor`的入参，只计算`Tensor`入参的梯度。

示例如下。整网入参`(x, y, z)`中，`x`和`z`是`Tensor`，`y`是非`Tensor`。因此，`grad_net`在对整网入参`(x, y, z)`求梯度的时候，会自动忽略`y`的梯度，只计算`x`和`z`的梯度，返回`(grad_x, grad_z)`。

```python
import numpy as np
import mindspore as ms
from mindspore import nn

ms.set_context(mode=ms.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()

    def construct(self, x, y, z):
        return x + y + z

class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.forward_net = net

    def construct(self, x, y, z):
        return ms.grad(self.forward_net, grad_position=(0, 1, 2))(x, y, z)

input_x = ms.Tensor([1])
input_y = 2
input_z = ms.Tensor([3])

net = Net()
grad_net = GradNet(net)
ret = grad_net(input_x, input_y, input_z)
print('ret:{}'.format(ret))
```

结果如下：

```text
ret:(Tensor(shape=[1], dtype=Int64, value= [1]), Tensor(shape=[1], dtype=Int64, value= [1]))
```

## 基础语法的语法约束

图模式下的执行图是从源码转换而来，并不是所有的Python语法都能支持。下面介绍在基础语法下存在的一些语法约束。更多网络编译问题可见[网络编译](https://www.mindspore.cn/docs/zh-CN/master/faq/network_compilation.html)。

1. 当`construct`函数里，使用未定义的类成员时，将抛出`AttributeError`异常。示例如下：

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

   ```text
   AttributeError: External object has no attribute y
   ```

2. `nn.Cell`不支持`classmethod`修饰的类方法。示例如下：

   ```python
   import mindspore as ms

   ms.set_context(ms.GRAPH_MODE)

    class Net(ms.nn.Cell):
    @classmethod
    def func(cls, x, y):
        return x + y

    def construct(self, x, y):
        return self.func(x, y)

    net = Net()
    out = net(ms.Tensor(1), ms.Tensor(2))
    print(out)
    ```

    结果报错如下：

    ```Text
    TypeError: too many positional arguments
    ```

3. 在图模式下，有些Python语法难以转换成图模式下的[中间表示MindIR](https://www.mindspore.cn/docs/zh-CN/master/design/all_scenarios.html#中间表示mindir)。对标Python的关键字，存在部分关键字在图模式下是不支持的：AsyncFunctionDef、ClassDef、Delete、AnnAssign、AsyncFor、AsyncWith、Match、Try、Import、ImportFrom、Nonlocal、NamedExpr、Set、SetComp、DictComp、Await、Yield、YieldFrom、Starred。如果在图模式下使用相关的语法，将会有相应的报错信息提醒用户。

    如果使用Try语句，示例如下：

    ```python
    import mindspore as ms

    @ms.jit
    def test_try_except(x, y):
        global_out = 1
        try:
            global_out = x / y
        except ZeroDivisionError:
            print("division by zero, y is zero.")
        return global_out

    test_try_except_out = test_try_except(1, 0)
    print("out:", test_try_except_out)
    ```

    结果报错如下：

    ```Text
    RuntimeError: Unsupported statement 'Try'.
    ```

4. 对标Python内置数据类型，除去当前图模式下支持的[Python内置数据类型](#python内置数据类型)，复数`complex`和集合`set`类型是不支持的。列表`list`和字典`dictionary`的一些高阶用法在基础语法场景下是不支持的，需要在JIT语法支持级别选项`jit_syntax_level`为`LAX`时才支持，更多请参考本文的[扩展语法（LAX级别）](#扩展语法lax级别)章节。

5. 对标Python的内置函数，在基础语法场景下，除去当前图模式下支持的[Python内置函数](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax/python_builtin_functions.html)，仍存在部分内置函数在图模式下是不支持的，例如：basestring、bin、bytearray、callable、chr、cmp、compile、 delattr、dir、divmod、eval、execfile、file、frozenset、hash、hex、id、input、issubclass、iter、locals、long、memoryview、next、object、oct、open、ord、property、raw_input、reduce、reload、repr、reverse、set、slice、sorted、unichr、unicode、vars、xrange、\_\_import\_\_。

6. Python提供了很多第三方库，通常需要通过import语句调用。在图模式下JIT语法支持级别为STRICT时，不能直接使用第三方库。如果需要在图模式下使用第三方库的数据类型或者调用第三方库的方法，需要在JIT语法支持级别选项`jit_syntax_level`为`LAX`时才支持，更多请参考本文的[扩展语法（LAX级别）](#扩展语法lax级别)中的[调用第三方库](#调用第三方库)章节。

7. 在图模式下JIT语法支持级别为STRICT时，不能直接使用自定义类的对象，属性和方法。如果需要在图模式下使用自定义类的信息，更多请参考本文的[扩展语法（LAX级别）](#扩展语法lax级别)中的[支持自定义类的使用](#支持自定义类的使用)章节。

   例如：

   ```python
   import mindspore as ms

   ms.set_context(mode=ms.GRAPH_MODE)

   class GetattrClass():
       def __init__(self):
           self.attr1 = 99
           self.attr2 = 1

       def method1(self, x):
           return x + self.attr2

   class GetattrClassNet(ms.nn.Cell):
       def __init__(self):
           super(GetattrClassNet, self).__init__()
           self.cls = GetattrClass()

       def construct(self):
           return self.cls.method1(self.cls.attr1)

   net = GetattrClassNet()
   out = net()
   assert out == 100
   ```

   将会有相关报错：

   ```Text
   TypeError: Do not support to convert <class '__main__.GetattrClass'> object into graph node.
   ```

## 扩展语法（LAX级别）

下面主要介绍当前扩展支持的静态图语法。

### 调用第三方库

- 第三方库

  1. Python内置模块和Python标准库。例如`os`、`sys`、`math`、`time`等模块。

  2. 第三方代码库。路径在Python安装目录的`site-packages`目录下，需要先安装后导入，例如`NumPy`、`SciPy`等。需要注意的是，`mindyolo`、`mindflow`等MindSpore套件不被视作第三方库。

  3. 通过环境变量`MS_JIT_IGNORE_MODULES`指定的模块。与之相对的有环境变量`MS_JIT_MODULES`，具体使用方法请参考[环境变量](https://www.mindspore.cn/docs/zh-CN/master/note/env_var_list.html)。

- 支持第三方库的数据类型，允许调用和返回第三方库的对象。

  示例如下：

  ```python
  import numpy as np
  import mindspore as ms

  @ms.jit
  def func():
      a = np.array([1, 2, 3])
      b = np.array([4, 5, 6])
      out = a + b
      return out

  print(func())
  ```

  结果如下：

  ```text
  [5 7 9]
  ```

- 支持调用第三方库的方法。

  示例如下：

  ```python
  from scipy import linalg
  import mindspore as ms

  @ms.jit
  def func():
      x = [[1, 2], [3, 4]]
      return linalg.qr(x)

  out = func()
  print(out[0].shape)
  ```

  结果如下：

  ```text
  (2, 2)
  ```

- 支持使用NumPy第三方库数据类型创建Tensor对象。

  示例如下：

  ```python
  import numpy as np
  import mindspore as ms

  @ms.jit
  def func():
      x = np.array([1, 2, 3])
      out = ms.Tensor(x) + 1
      return out

  print(func())
  ```

  结果如下：

  ```text
  [2, 3, 4]
  ```

- 暂不支持对第三方库数据类型的下标索引赋值。

  示例如下：

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

  报错信息如下:

  ```text
  RuntimeError: For operation 'setitem', current input arguments types are <External, Number, Number>. The 1-th argument type 'External' is not supported now.
  ```

### 支持自定义类的使用

支持在图模式下使用用户自定义的类，可以对类进行实例化，使用对象的属性及方法。

例如下面的例子，其中`GetattrClass`是用户自定义的类，没有使用`@ms_class`修饰，也没有继承`nn.Cell`。

```python
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE)

class GetattrClass():
    def __init__(self):
        self.attr1 = 99
        self.attr2 = 1

    def method1(self, x):
        return x + self.attr2

class GetattrClassNet(ms.nn.Cell):
    def __init__(self):
        super(GetattrClassNet, self).__init__()
        self.cls = GetattrClass()

    def construct(self):
        return self.cls.method1(self.cls.attr1)

net = GetattrClassNet()
out = net()
assert out == 100
```

### 基础运算符支持更多数据类型

在静态图语法重载了以下运算符: ['+', '-', '*', '/', '//', '%', '**', '<<', '>>', '&', '|', '^', 'not', '==', '!=', '<', '>', '<=', '>=', 'in', 'not in', 'y=x[0]']。图模式重载的运算符详见[运算符](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax/operators.html)。列表中的运算符在输入图模式中不支持的输入类型时将使用扩展静态图语法支持，并使输出结果与动态图模式下的输出结果一致。

代码用例如下。

```python
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
ms.set_context(mode=ms.GRAPH_MODE)

class InnerClass(nn.Cell):
    def construct(self, x, y):
        return x.asnumpy() + y.asnumpy()

net = InnerClass()
ret = net(Tensor([4, 5]), Tensor([1, 2]))
print(ret)
```

```text
[5 7]
```

上述例子中，`.asnumpy()`输出的数据类型:  `numpy.ndarray`为运算符`+`在图模式中不支持的输入类型。因此`x.asnumpy() + y.asnumpy()`将使用扩展语法支持。

在另一个用例中：

```python
class InnerClass(nn.Cell):
    def construct(self):
        return (None, 1) in ((None, 1), 1, 2, 3)

net = InnerClass()
print(net())
```

```text
True
```

`tuple` in `tuple`在原本的图模式中是不支持的运算，现已使用扩展静态图语法支持。

### 基础类型

扩展对Python原生数据类型`List`、`Dictionary`、`None`的支持。

#### 支持列表就地修改操作

列表`List`以及元组`Tuple`是Python中最基本的序列内置类型，`List`与`Tuple`最核心的区别是`List`是可以改变的对象，而`Tuple`是不可以更改的。这意味着`Tuple`一旦被创建，就不可以在对象地址不变的情况下更改。而`List`则可以通过一系列inplace操作，在不改变对象地址的情况下，对对象进行修改。例如：

 ```python
 a = [1, 2, 3, 4]
 a_id = id(a)
 a.append(5)
 a_after_id = id(a)
 assert a_id == a_after_id
 ```

上述示例代码中，通过`append`这个inplace语法更改`List`对象的时候，其对象的地址并没有被修改。而`Tuple`是不支持这种inplace操作的。在`JIT_SYNTAX_LEVEL`设置为`LAX`的情况下，静态图模式可以支持部分`List`对象的inplace操作。

具体使用场景如下：

- 支持从全局变量中获取原`List`对象。

  在下面示例中，静态图获取到`List`对象，并在原有对象上进行了图模式支持的inplace操作`list.reverse()`, 并将原有对象返回。可以看到图模式返回的对象与原有的全局变量对象id相同，即两者为同一对象。若`JIT_SYNTAX_LEVEL`设置为`STRICT`选项，则返回的`List`对象与全局对象为两个不同的对象。

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

- 不支持对输入`List`对象进行inplace操作。

  `List`作为静态图输入时，会对该`List`对象进行一次复制，并使用该复制对象进行后续的计算，因此无法对原输入对象进行inplace操作。例如：

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

  如上述用例所示，`List`对象作为图模式输入时无法在原有对象上进行inplace操作。图模式返回的对象与输入的对象id不同，为不同对象。

- 支持部分`List`内置函数的就地修改操作。

  在`JIT_SYNTAX_LEVEL`设置为`LAX`的情况下，图模式部分`List`内置函数支持inplace。在 `JIT_SYNTAX_LEVEL`为 `STRICT` 的情况下，所有方法均不支持inplace操作。

  目前，图模式支持的`List`就地修改内置方法有`extend`、`pop`、`reverse`以及`insert`。内置方法`append`、`clear`以及索引赋值暂不支持就地修改，后续版本将会支持。

  示例如下：

  ```python
  import mindspore as ms

  list_input = [1, 2, 3, 4]

  @ms.jit
  def list_func():
      list_input.reverse()
      return list_input

  output = list_func()  # output: [4, 3, 2, 1]  list_input: [4, 3, 2, 1]
  assert id(output) == id(list_input)
  ```

#### 支持Dictionary的高阶用法

- 支持顶图返回Dictionary。

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

- 支持Dictionary索引取值和赋值。

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

#### 支持使用None

`None`是Python中的一个特殊值，表示空，可以赋值给任何变量。对于没有返回值语句的函数认为返回`None`。同时也支持`None`作为顶图或者子图的入参或者返回值。支持`None`作为切片的下标，作为`List`、`Tuple`、`Dictionary`的输入。

示例如下：

```python
import mindspore as ms

@ms.jit
def test_return_none():
    return 1, "a", None

res = test_return_none()
print(res)
```

输出结果：

```text
(1, 'a', None)
```

对于没有返回值的函数，默认返回`None`对象。

```python
import mindspore as ms

@ms.jit
def foo():
    x = 3
    print("x:", x)

res = foo()
assert res is None
```

如下面例子，`None`作为顶图的默认入参。

```python
import mindspore as ms

@ms.jit
def foo(x, y=None):
    if y is not None:
        print("y:", y)
    else:
        print("y is None")
    print("x:", x)
    return y

x = [1, 2]
res = foo(x)
assert res is None
```

### 内置函数支持更多数据类型

扩展内置函数的支持范围。Python内置函数完善支持更多输入类型，例如第三方库数据类型。

例如下面的例子，`x.asnumpy()`和`np.ndarray`均是扩展支持的类型。更多内置函数的支持情况可见[Python内置函数](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax/python_builtin_functions.html)章节。

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE)

class Net(nn.Cell):
    def construct(self, x):
        return isinstance(x.asnumpy(), np.ndarray)

x = Tensor(np.array([-1, 2, 4]))
net = Net()
out = net(x)
assert out
```

### 支持控制流

为了提高Python标准语法支持度，实现动静统一，扩展支持更多数据类型在控制流语句的使用。控制流语句是指`if`、`for`、`while`等流程控制语句。理论上，通过扩展支持的语法，在控制流场景中也支持。代码用例如下：

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

结果如下：

```text
res: 2
```

### 支持属性设置与修改

具体使用场景如下：

- 对自定义类对象以及第三方类型的属性进行设置与修改。

  图模式下支持对自定义类对象的属性进行设置与修改，例如：

  ```python
  import mindspore import jit

  class AssignClass():
      def __init__(self):
          self.x = 1

  obj = AssignClass()

  @jit
  def foo():
      obj.x = 100
      return

  foo()
  print(f"obj.x is: {obj.x}")
  ```

  运行结果为：

  ```text
  obj.x is: 100
  ```

  图模式下支持对第三方库对象的属性进行设置与修改，例如：

  ```python
  import mindspore import jit
  import numpy as np

  @jit
  def foo():
      a = np.array([1, 2, 3, 4])
      a.shape = (2, 2)
      return a.shape

  shape = foo()
  print(f"shape is {shape}")
  ```

  运行结果为：

  ```text
  shape is (2, 2)
  ```

- 对Cell的self对象进行修改，例如：

  ```python
  import mindspore as ms
  from mindspore import nn, set_context
  set_context(mode=ms.GRAPH_MODE)

  class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.m = 2

    def construct(self):
        self.m = 3
        return

  net = Net()
  net()
  print(f"net.m is {net.m}")
  ```

  运行结果为：

  ```text
  net.m is 3
  ```

  注意，self对象只支持属性修改，而不支持属性的设置，即只支持对`__init__`函数内设置的属性进行修改。若`__init__`内没有定义某个属性，则图模式内不允许设置此属性。例如：

  ```python
  import mindspore as ms
  from mindspore import nn, set_context
  set_context(mode=ms.GRAPH_MODE)

  class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.m = 2

    def construct(self):
        self.m2 = 3 # self.m2在__init__内没有被设置，因此无法在图模式内进行设置
        return

  net = Net()
  net()
  ```

- 对静态图内的Cell对象以及jit_class对象进行设置与修改。

  支持对图模式Cell对象进行属性修改，例如：

  ```python
  import mindspore as ms
  from mindspore import nn, set_context
  set_context(mode=ms.GRAPH_MODE)

  class InnerNet(nn.Cell):
      def __init__(self):
          super(InnerNet, self).__init__()
          self.x = 10

  class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.inner = InnerNet()

    def construct(self):
        self.inner.x = 100
        return

  net = Net()
  net()
  print(f"net.inner.x is {net.inner.x}")
  ```

  运行结果为：

  ```text
  net.inner.x is 100
  ```

  支持对图模式jit_class对象进行属性修改，例如：

  ```python
  import mindspore as ms
  from mindspore import nn, set_context, jit_class
  set_context(mode=ms.GRAPH_MODE)

  @jit_class
  class InnerClass():
      def __init__(self):
          self.x = 10

  class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.inner = InnerClass()

    def construct(self):
        self.inner.x = 100
        return

  net = Net()
  net()
  print(f"net.inner.x is {net.inner.x}")
  ```

  运行结果为：

  ```text
  net.inner.x is 100
  ```

  注意，若在图模式内对Cell/jit_class对象进行属性修改前也获取了相同属性，该获取到的属性会被解析为常量。在多次运行相同网络时可能会造成问题，例如：

  ```python
  import mindspore as ms
  from mindspore import nn, set_context
  set_context(mode=ms.GRAPH_MODE)

  class InnerNet(nn.Cell):
      def __init__(self):
          self.x = 1

  class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.inner = InnerNet()

    def construct(self):
        a = self.inner.x
        self.inner.x = a + 1
        return

  net = Net()
  value0 = net.inner.x
  net()
  value1 = net.inner.x
  net()
  value2 = net.inner.x
  print(f"value0 is {value0}")
  print(f"value1 is {value1}")
  print(f"value2 is {value2}")
  ```

  运行结果为：

  ```text
  value0 is 1
  value1 is 2
  value2 is 2
  ```

  但是在动态图模式下，`value2`的值应该为3。但因为语句`a = self.inner.x`中的`self.inner.x`被固化为常量2，导致两次运行时`self.inner.x`被设置的值均为2。此问题将在后续版本解决。

### 支持求导

扩展支持的静态图语法，同样支持其在求导中使用，例如：

```python
import mindspore as ms
from mindspore import ops

@ms.jit
def dict_net(a):
    x = {'a': a, 'b': 2}
    return a, (x, (1, 2))

out = ops.grad(dict_net)(ms.Tensor([1]))
assert out == 2
```

### Annotation Type

对于运行时的扩展支持的语法，会产生一些无法被类型推导出的节点，比如动态创建Tensor等。这种类型称为`Any`类型。因为该类型无法在编译时推导出正确的类型，所以这种`Any`将会以一种默认最大精度`Float64`进行运算，防止其精度丢失。为了能更好的优化相关性能，需要减少`Any`类型数据的产生。当用户可以明确知道当前通过扩展支持的语句会产生具体类型的时候，我们推荐使用`Annotation @jit.typing:`的方式进行指定对应Python语句类型，从而确定解释节点的类型避免`Any`类型的生成。

例如，[Tensor](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor)类和[tensor](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.tensor.html#mindspore.tensor)接口的区别就在于在`tensor`接口内部运用了Annotation Type机制。当`tensor`函数的`dtype`确定时，函数内部会利用`Annotation`指定输出类型从而避免`Any`类型的产生。`Annotation Type`的使用只需要在对应Python语句上面或者后面加上注释 `# @jit.typing: () -> tensor_type[float32]` 即可，其中 `->` 后面的 `tensor_type[float32]` 指示了被注释的语句输出类型。

代码用例如下。

```python
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, Tensor

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.abs = ops.Abs()

    @ms.jit
    def construct(self, x, y):
        y1 = ms.tensor(x.asnumpy() + y.asnumpy(), dtype=ms.float32)
        y2 = ms.Tensor(x.asnumpy() + y.asnumpy(), dtype=ms.float32) # @jit.typing: () -> tensor_type[float32]
        y3 = Tensor(x.asnumpy() + y.asnumpy())
        y4 = Tensor(x.asnumpy() + y.asnumpy(), dtype=ms.float32)
        return self.abs(y1), self.abs(y2), self.abs(y3), self.abs(y4)

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
x = ms.Tensor(-1, dtype=ms.int32)
y = ms.Tensor(-1, dtype=ms.float32)
y1, y2, y3, y4 = net(x, y)

print(f"y1 value is {y1}, dtype is {y1.dtype}")
print(f"y2 value is {y2}, dtype is {y2.dtype}")
print(f"y3 value is {y3}, dtype is {y3.dtype}")
print(f"y4 value is {y4}, dtype is {y4.dtype}")
```

```text
y1 value is 2.0, dtype is Float32
y2 value is 2.0, dtype is Float32
y3 value is 2.0, dtype is Float64
y4 value is 2.0, dtype is Float64
```

上述例子，可以看到创建了`Tensor`的相关区别。对于`y3`、`y4`，因为`Tensor`类没有增加`Annotation`指示，`y3`、`y4`没有办法推出正确的类型，导致只能按照最高精度`Float64`进行运算。
对于`y2`，由于创建`Tensor`时，通过`Annotation`指定了对应类型，使得其类型可以按照指定类型进行运算。
对于`y1`，由于使用了`tensor`函数接口创建`Tensor`，传入的`dtype`参数作为`Annotation`的指定类型，所以也避免了`Any`类型的产生。

## 扩展语法的语法约束

在使用静态图扩展支持语法时，请注意以下几点：

1. 对标动态图的支持能力，即：须在动态图语法范围内，包括但不限于数据类型等。

2. 在扩展静态图语法时，支持了更多的语法，但执行性能可能会受影响，不是最佳。

3. 在扩展静态图语法时，支持了更多的语法，由于使用Python的能力，不能使用MindIR导入导出的能力。

4. 暂不支持跨Python文件重复定义同名的全局变量，且这些全局变量在网络中会被用到。
