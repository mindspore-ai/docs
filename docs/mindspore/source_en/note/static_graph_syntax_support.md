# Static Graph Syntax Support

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/static_graph_syntax_support.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

In graph mode, Python code is not executed by the Python interpreter. Instead, the code is compiled into a static computation graph, and then the static computation graph is executed.

Currently, the function, and class (including Cell, Cell subclass, ms_class class, and common user defined class) methods modified by the `@ms_function` decorator can be built.
For a function, build the function definition. For the network, build the `construct` method and other methods or functions called by the `construct` method.

For details about how to use `ms_function`, click [ms_function API document](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.ms_function.html#mindspore.ms_function).

For details about the definition of `Cell`, click [Cell API document](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html).

Due to syntax parsing restrictions, the supported data types, syntax, and related operations during graph building are not completely consistent with the Python syntax. As a result, some usage is restricted.

The following describes the data types, syntax, and related operations supported during static graph building. These rules apply only to graph mode.

> All the following examples run on the network in graph mode. For brevity, the network definition is not described.
>

## Data Types

### Built-in Python Data Types

Currently, the following built-in `Python` data types are supported: `Number`, `String`, `List`, `Tuple`, and `Dictionary`.

#### Number

Supports `int`, `float`, and `bool`, but does not support complex numbers.

`Number` can be defined on the network. That is, the syntax `y = 1`, `y = 1.2`, and `y = True` are supported.

When the data is constant, the value of the data can be achieved at compile time, the forcible conversion to `Number` is supported in the network. That is, the syntax `y = int(x)`, `y = float(x)`, and `y = bool(x)` are supported.

#### String

`String` can be constructed on the network. That is, the syntax `y = "abcd"` is supported.

Use str() to change the constant value to string, str.format() can use to format the string, but not supported to input a kwargs type arguments and the argument of format function cannot be a variable.

For example:

```python
from mindspore import ms_function

@ms_function()
def test_str_format():
    x = "{} is zero".format(0)
    return x

x = test_str_format
print(x)
```

The result is as follows:

```text
0 is zero
```

#### List

`List` can be constructed on the network, that is, the syntax `y = [1, 2, 3]` is supported.

`List` to be output in the computation graph will be converted into `Tuple`.

When using List index to get the element，the reference type between MindSpore and Python interpreter may be different. Due to MindSpore using ListGetItem to implement getting value of the list, and the operator ListGetItem will return a copy of the variable, that make the reference type may not same with Python interpreter.

For example:

Python：

```python
>>>a = [[1,2,3],4,5]
>>>b = a[0]
>>>b[0] = 123123
>>>a
[123123, 2, 3], 4, 5]
```

MindSpore:

```python
from mindspore import ms_function

@ms_function
def test_list():
    a = [[1,2,3],4,5]
    b = a[0]
    b[0] = 123123
    return a

a = test_list()
print('a:{}'.format(a))
```

The result is as follows:

```text
x: ((1, 2, 3), 4, 5)
```

- Supported APIs

  `append`: adds an element to `list`.

  For example:

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

  The result is as follows:

  ```text
  x: (1, 2, 3, 4)
  ```

  `insert`: inserts the specified element at the specified position in the `list`.

  For example:

  ```python
  from mindspore import ms_function

  @ms_function()
  def test_list_insert():
      x = [1, 3, 4]
      x.insert(0, 2)
      return x

  x = test_list_insert()
  print('x:{}'.format(x))
  ```

  The result is as follows:

  ```text
  x: (2, 1, 3, 4)
  ```

  `pop`: removes the element at the specified position in `list`, removing the last one by default.

  For example:

  ```python
  from mindspore import ms_function

  @ms_function()
  def test_list_pop():
      x = [1, 3, 4]
      y = x.pop()
      return x, y

  x, y = test_list_pop()
  print('x:{}'.format(x))
  print('y:', y)
  ```

  The result is as follows:

  ```text
  x: (1, 3)
  y: 4
  ```

  `clear`: clears the elements in `list`.

  For example:

  ```python
  from mindspore import ms_function

  @ms_function()
  def test_list_clear():
      x = [1, 3, 4]
      x.clear()
      return x

  x = test_list_clear()
  ```

  The result is as follows:

  ```text
  x: ()
  ```

  `extend`: appends multiple elements of another sequence to the end of `list`.

  For example:

  ```python
  from mindspore import ms_function

  @ms_function()
  def test_list_extend():
      x = [1, 2, 3, 4]
      y = [5, 6, 7]
      x.extend(y)
      return x

  x = test_list_extend()
  print('x:{}'.format(x))
  ```

  The result is as follows:

  ```text
  x: (1, 2, 3, 4, 5, 6, 7)
  ```

  `reverse`: reverses the elements of `list`.

  For example:

  ```python
  from mindspore import ms_function

  @ms_function()
  def test_list_reverse():
      x = [1, 2, 3, 4]
      x.reverse()
      return x

  x = test_list_reverse()
  print('x:{}'.format(x))
  ```

  The result is as follows:

  ```text
  x: (4, 3, 2, 1)
  ```

  `count`: counts the number of occurrences of an element in `list`. The current count method only supports constant scenarios.

  For example:

  ```python
  from mindspore import ms_function

  @ms_function()
  def test_list_count():
      x = [1, 2, 3, 4]
      num = x.count(2)
      return num

  num = test_list_count()
  print('num:', num)
  ```

  The result is as follows:

  ```text
  num: 1
  ```

  If there is a Tensor variable in the usage scenario of count, a related exception will be thrown.

  For example:

  ```python
  from mindspore import ms_function, Tensor

  @ms_function()
  def test_list_count(input_x):
      x = [1, 2, 3, 4]
      num = x.count(input_x)
      return num

  input_x = Tensor(2)
  num = test_list_count()
  print('num:', num)
  ```

  The result is as follows:

  ```text
  The list count not support variable scene now. The count data is Tensor type.
  ```

- Supported index values and value assignment

  Single-level and multi-level index values and value assignment are supported.

  The index value supports only `int` and `slice`.

  The element of `slice` data should be constant that can be deduced in the state of compiling graph.

  The assigned value can be `Number`, `String`, `Tuple`, `List`, or `Tensor`.

  When the value of the current slice is Tensor, the Tensor needs to be converted to a List, which is currently implemented through JIT Fallback. Therefore, variable scenarios cannot be supported temporarily.

  For example:

  ```python
  import numpy as np
  from mindspore import ms_function

  t = ms.Tensor(np.array([1, 2, 3]))

  @ms_function()
  def test_index():
      x = [[1, 2], 2, 3, 4]
      m = x[0][1]
      z = x[1::2]
      x[1] = t
      x[2] = "ok"
      x[3] = (1, 2, 3)
      x[0][1] = 88
      n = x[-3]
      return m, z, x, n

  m, z, x, n = test_index()
  print('m:{}'.format(m))
  print('z:{}'.format(z))
  print('x:{}'.format(x))
  print('n:{}'.format(n))
  ```

  The result is as follows:

  ```text
  m:2
  z:[2, 4]
  x:[[1, 88], Tensor(shape=[3], dtype=Int64, value= [1, 2, 3]), 'ok', (1, 2, 3)]
  n:[1 2 3]
  ```

#### Tuple

`Tuple` can be constructed on the network, that is, the syntax `y = (1, 2, 3)` is supported.

Forcible conversion to `Tuple` is not supported on the network. That is, the syntax `y = tuple(x)` is not supported.

The reference type of tuple is same as List, please  refer to List.

- Supported index values

  The index value can be `int`, `slice`, `Tensor`, and multi-level index value. That is, the syntax `data = tuple_x[index0][index1]...` is supported.

  Restrictions on the index value `Tensor` are as follows:

    - `Tuple` stores `Cell`. Each `Cell` must be defined before a tuple is defined. The number of input parameters, input parameter type, and input parameter `shape` of each `Cell` must be the same. The number of outputs of each `Cell` must be the same. The output type must be the same as the output shape.

    - The index `Tensor` is a scalar `Tensor` whose `dtype` is `int32`. The value range is `[-tuple_len, tuple_len)`, negative index is not supported in `Ascend` backend.

    - This syntax does not support the running branches whose control flow conditions `if`, `while`, and `for` are variables. The control flow conditions can be constants only.

    - `GPU` and `Ascend` backend is supported.

  An example of the `int` and `slice` indexes is as follows:

  ```python
  import numpy as np
  from mindspore import ms_function

  t = ms.Tensor(np.array([1, 2, 3]))

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

  The result is as follows:

  ```text
  y:3
  z:[1 2 3]
  m:((2, 3, 4), 3, 4)
  n:(2, 3, 4)
  ```

  An example of the `Tensor` index is as follows:

  ```python
  import mindspore as ms
  from mindspore import nn

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

  The result is as follows:

  ```text
  ret:[0.]
  ```

#### Dictionary

`Dictionary` can be constructed on the network. That is, the syntax `y = {"a": 1, "b": 2}` is supported. Currently, only `String` can be used as the `key` value.

`Dictionary` to be output in the computational graph will extract all `value` values to form the `Tuple` output.

- Supported APIs

  `keys`: extracts all `key` values from `dict` to form `Tuple` and return it.

  `values`: extracts all `value` values from `dict` to form `Tuple` and return it.

  `items`: extracts `Tuple` composed of each pair of `value` values and `key` values in `dict` to form `Tuple` and return it.

  For example:

  ```python
  import mindspore as ms
  import numpy as np
  from mindspore import ms_function

  x = {"a": ms.Tensor(np.array([1, 2, 3])), "b": ms.Tensor(np.array([4, 5, 6])), "c": ms.Tensor(np.array([7, 8, 9]))}

  @ms_function()
  def test_dict():
      y = x.keys()
      z = x.values()
      q = x.items()
      return y, z, q

  y, z, q = test_dict()
  print('y:{}'.format(y))
  print('z:{}'.format(z))
  ```

  The result is as follows:

  ```text
  y:('a', 'b', 'c')
  z:(Tensor(shape=[3], dtype=Int64, value= [1, 2, 3]), Tensor(shape=[3], dtype=Int64, value= [4, 5, 6]), Tensor(shape=[3], dtype=Int64, value= [7, 8, 9]))
  q:[('a', Tensor(shape=[3], dtype=Int64, value= [1, 2, 3])), ('b', Tensor(shape=[3], dtype=Int64, value= [4, 5, 6])), ('c', Tensor(shape=[3], dtype=Int64, value= [7, 8, 9]))]
  ```

- Supported index values and value assignment

  The index value supports only `String`. The assigned value can be `Number`, `Tuple`, or `Tensor`.

  For example:

  ```python
  import mindspore as ms
  import numpy as np
  from mindspore import ms_function

  x = {"a": ms.Tensor(np.array([1, 2, 3])), "b": ms.Tensor(np.array([4, 5, 6])), "c": ms.Tensor(np.array([7, 8, 9]))}

  @ms_function()
  def test_dict():
      y = x["b"]
      x["a"] = (2, 3, 4)
      return x, y

  x, y = test_dict()
  print('x:{}'.format(x))
  print('y:{}'.format(y))
  ```

  The result is as follows:

  ```text
  x:{'a': (2, 3, 4), 'b': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6]), 'c': Tensor(shape=[3], dtype=Int64, value= [7, 8, 9])}
  y:[4 5 6]
  ```

### MindSpore User-defined Data Types

Currently, MindSpore supports the following user-defined data types: `Tensor`, `Primitive`, and `Cell`.

#### Tensor

Currently, tensors can be constructed in the network.

For details of `Tensor`, click [Tensor API document](https://mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Tensor.html#mindspore-tensor).

#### Primitive

Currently, `Primitive` and its subclass instances can be constructed on the network. That is, the `reduce_sum = ReduceSum(True)` syntax is supported.

However, during construction, the parameter can be specified only in position parameter mode, and cannot be specified in the key-value pair mode. That is, the syntax `reduce_sum = ReduceSum(keep_dims=True)` is not supported.

Currently, the attributes and APIs related to `Primitive` and its subclasses cannot be called on the network.

For details about the defined `Primitive`, click [Primitive API document](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Primitive.html#mindspore.ops.Primitive).

#### Cell

Currently, `Cell` and its subclass instances can be constructed on the network. That is, the syntax `cell = Cell(args...)` is supported.

However, during construction, the parameter can be specified only in position parameter mode, and cannot be specified in the key-value pair mode. That is, the syntax `cell = Cell(arg_name=value)` is not supported.

Currently, the attributes and APIs related to `Cell` and its subclasses cannot be called on the network unless they are called through `self` in `construct` of `Cell`.

For details about the definition of `Cell`, click [Cell API document](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html).

#### Parameter

`Parameter` is a variable tensor, indicating the parameters that need to be updated during network training.

For details about the definition of `Parameter`, click [Parameter API document](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Parameter.html#mindspore.Parameter).

## Primaries

Primaries represent the most tightly bound operations of the language Which contains `Attribute references`, `Subscriptions`, `Calls`.

### Attribute References

An attribute reference is a primary followed by a period and a name.

In `Cell` instance of MindSpore, using attribute reference as left operands must meet the restrictions below:

- The attribute must belong to self, such as self.xxx. It is not supported to change attribute of other instance.

- The attribute type must be `Parameter` and be initialized in `__init__` function.

For example:

```python
import mindspore as ms
from mindspore import nn
import numpy as np
from mindspore.ops import constexpr

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight = ms.Parameter(ms.Tensor(3, ms.float32), name="w")
        self.m = 2

    def construct(self, x, y):
        self.weight = x     # restictions matched,  success
        # self.m = 3               # self.m not Parameter type, failure
        # y.weight = x          # not attribute of self, failure
        return x

net = Net()
ret = net(1, 2)
print('ret:{}'.format(ret))
```

The result is as follows:

```text
ret:1
```

### Index Value

Index value of  a sequence `Tuple`, `List`, `Dictionary`, `Tensor` which called subscription in Python.

Index value of `Tuple` refers to chapter [Tuple](#tuple) of this page.

Index value of `List` refers to chapter [List](#list) of this page.

Index value of `Dictionary` refers to chapter [Dictionary](#dictionary) of this page.

Index value of `Tensor` refers to [Tensor index value document](https://www.mindspore.cn/docs/en/master/note/index_support.html#index-values).

### Calls

A call calls a callable object (e.g., `Cell` or `Primitive`) with a possibly empty series of arguments.

For example:

```python
import mindspore as ms
from mindspore import nn, ops
import numpy as np

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = ops.MatMul()

    def construct(self, x, y):
        out = self.matmul(x, y)  # A call of Primitive
        return out

x = ms.Tensor(np.ones(shape=[1, 3]), ms.float32)
y = ms.Tensor(np.ones(shape=[3, 4]), ms.float32)
net = Net()
ret = net(x, y)
print('ret:{}'.format(ret))
```

The result is as follows:

```text
ret:[[3. 3. 3. 3.]]
```

## Operators

Arithmetic operators and assignment operators support the `Number` and `Tensor` operations, as well as the `Tensor` operations of different `dtype`.

This is because these operators are converted to operators with the same name for computation, and they support implicit type conversion.

For details about the rules, click <https://www.mindspore.cn/docs/en/master/note/operator_list_implicit.html#conversion-rules>.

### Unary Arithmetic Operators

| Unary Arithmetic Operator | Supported Type                 |
| :------------------------ | :----------------------------- |
| `+`                       | `Number`, `Tensor`             |
| `-`                       | `Number`, `Tensor`             |
| `~`                       | `Tensor` with `Bool` data type |

notes:

- In native python the `~` operator get the bitwise inversion of its integer argument; in Mindspore the `~` redefined to get logic not for `Tensor(Bool)`.

### Binary Arithmetic Operators

| Binary Arithmetic Operator | Supported Type|
| :----------- |:--------|
| `+` |`Number` + `Number`, `String` + `String`, `Number` + `Tensor`, `Tensor` + `Number`, `Tuple` + `Tensor`, `Tensor` + `Tuple`, `List` + `Tensor`, `Tensor`+`List`, `List`+`List`, `Tensor` + `Tensor`, `Tuple` + `Tuple`.|
| `-` |`Number` - `Number`, `Tensor` - `Tensor`, `Number` -`Tensor`, `Tensor` - `Number`, `Tuple` -`Tensor`, `Tensor` -`Tuple`, `List` -`Tensor`, `Tensor` -`List`.|
| `*` |`Number` \* `Number`, `Tensor` \* `Tensor`, `Number` \* `Tensor`, `Tensor` \* `Number`, `List` \* `Number`, `Number` \* `List`, `Tuple` \* `Number`, `Number` \* `Tuple`, `Tuple` \* `Tensor`, `Tensor` \* `Tuple`,  `List` \*`Tensor`, `Tensor` \* `List`.|
| `/` |`Number` / `Number`, `Tensor` / `Tensor`, `Number` / `Tensor`, `Tensor` / `Number`, `Tuple` / `Tensor`, `Tensor` / `Tuple`,  `List` / `Tensor`, `Tensor` / `List`.|
| `%` |`Number` % `Number`, `Tensor` % `Tensor`, `Number` % `Tensor`, `Tensor` % `Number`, `Tuple` % `Tensor`, `Tensor` % `Tuple`, `List` % `Tensor`, `Tensor` % `List`.|
| `**` |`Number` \*\* `Number`, `Tensor` \*\* `Tensor`, `Number` \*\* `Tensor`, `Tensor` \*\* `Number`, `Tuple` \*\* `Tensor`, `Tensor` \*\* `Tuple`,  `List` \*\* `Tensor`, `Tensor` \*\* `List`.|
| `//` |`Number` // `Number`, `Tensor` // `Tensor`, `Number` // `Tensor`, `Tensor` // `Number`, `Tuple` // `Tensor`, `Tensor` // `Tuple`,  `List` // `Tensor`, `Tensor` // `List`.|
| `&`     | `Number` & `Number`、`Tensor` & `Tensor`、`Number` & `Tensor`、`Tensor` & `Number`.                                                                                                                                                                  |
| `∣`      | `Number` &#124; `Number`、`Tensor` &#124; `Tensor`、`Number` &#124; `Tensor`、`Tensor` &#124; `Number`.                                                                                                                                                             |
| `^`     | `Number` ^ `Number`、`Tensor` ^ `Tensor`、`Number` ^ `Tensor`、`Tensor` ^ `Number`.                                                                                                                                                                  |
| `<<`    | `Number` << `Number`.                                                                                                                                                                                                                             |
| `>>`    | `Number` >> `Number`.                                                                                                                                                                                                                             |

Restrictions:

- If all operands are `Number` type, value of Number can't be `Bool`.
- If all operands are `Number` type, operations between  `Float64` and `Int32` are not supported.
- If either operand is `Tensor` type, left and right operands can't both be `Bool` value.
- The result of `List  * Number`  is concatenate  duplicate List Number times, data type of the `List` must be `Number`, `String`, `None` or `List`/`Tuple` that contains these types.  This rule applies to `Number * List`,  `Tuple * Number`, `Number * Tuple` too.

### Assignment Operators

| Assignment Operator | Supported Type、 |
| :----------- |:--------|
| `=`          |All Built-in Python Types that MindSpore supported and MindSpore User-defined Data Types.|
| `+=` |`Number` += `Number`, `String` += `String`, `Number` += `Tensor`, `Tensor` += `Number`, `Tuple` += `Tensor`, `Tensor` += `Tuple`, `List` += `Tensor`, `Tensor` += `List`, `List` += `List`, `Tensor` += `Tensor`, `Tuple` += `Tuple`.|
| `-=` |`Number` -= `Number`, `Tensor` -= `Tensor`, `Number` -= `Tensor`, `Tensor` -= `Number`, `Tuple` -= `Tensor`, `Tensor` -= `Tuple`, `List` -= `Tensor`, `Tensor` -= `List`.|
| `*=` |`Number` \*= `Number`, `Tensor` \*= `Tensor`, `Number` \*= `Tensor`, `Tensor` \*= `Number`, `List` \*= `Number`, `Number` \*= `List`, `Tuple` \*= `Number`, `Number` \*= `Tuple`, `Tuple` \*= `Tensor`, `Tensor` \*= `Tuple`,  `List` \*= `Tensor`, `Tensor` \*= `List`.|
| `/=` |`Number` /= `Number`, `Tensor` /= `Tensor`, `Number` /= `Tensor`, `Tensor` /= `Number`, `Tuple` /= `Tensor`, `Tensor` /= `Tuple`, `List` /= `Tensor`, `Tensor` /= `List`.|
| `%=` |`Number` %= `Number`, `Tensor` %= `Tensor`, `Number` %= `Tensor`, `Tensor` %= `Number`, `Tuple` %= `Tensor`, `Tensor` %= `Tuple`,  `List` %= `Tensor`、`Tensor` %= `List`.|
| `**=` |`Number` \*\*= `Number`, `Tensor` \*\*= `Tensor`, `Number` \*\*= `Tensor`, `Tensor` \*\*= `Number`, `Tuple` \*\*= `Tensor`, `Tensor` \*\*= `Tuple`,  `List` \*\*= `Tensor`, `Tensor` \*\*= `List`.|
| `//=` |`Number` //= `Number`, `Tensor` //= `Tensor`, `Number` //= `Tensor`, `Tensor` //= `Number`, `Tuple` //= `Tensor`, `Tensor` //= `Tuple`, `List` //= `Tensor`, `Tensor` //= `List`.|
| `&=`     | `Number` &= `Number`、`Tensor` &= `Tensor`、`Number` &= `Tensor`、`Tensor` &= `Number`.                                                                                                                                                                              |
| `∣=`      | `Number` &#124;= `Number`、`Tensor` &#124;= `Tensor`、`Number` &#124;= `Tensor`、`Tensor` &#124;= `Number`.                                                                                                                                                         |
| `^=`     | `Number` ^= `Number`、`Tensor` ^= `Tensor`、`Number` ^= `Tensor`、`Tensor` ^= `Number`.                                                                                                                                                                              |
| `<<=`    | `Number` <<= `Number`.                                                                                                                                                                                                                                         |
| `>>=`    | `Number` >>= `Number`.                                                                                                                                                                                                                                         |

Notes:

- For `=` the scenarios below are not allowed:

  Only instance of `Cell` and `Primitve` can be created in function construct, the statement like `xx = Tensor(...)` is forbidden.

  Only `Parameter` attribute of self can be assigned, for more detail refer to [Attribute Reference](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html#attribute-references).

- If all operands of  `AugAssign` are `Number` type, value of Number can't be `Bool`.

- If all operands of  `AugAssign` are `Number` type, operations between  `Float64` and `Int32` are not supported.

- If either operand of  `AugAssign` is `Tensor` type, left and right operands can't both be `Bool` value.

- The result of `List *= Number` is concatenate duplicate List Number times, data type of the `List` must be `Number`, `String`, `None` or `List`/`Tuple` that contains these types. This rule applies to `Number * List`, `Tuple * Number`, `Number * Tuple` too.

### Logical Operators

| Logical Operator | Supported Type|
| :----------- |:--------|
| `and` |`String`,  `Number`,  `Tuple`, `List` , `Dict`, `None`, `Scalar`, `Tensor`.|
| `or` |`String`,  `Number`,  `Tuple`, `List` , `Dict`, `None`, `Scalar`, `Tensor`.|
| `not` |`Number`, `tuple`, `List` and `Tensor`  with only one element.|

Restrictions:

- The left operand of operator `and`, `or` must be able to be converted to boolean value. For example, left operand can not be Tensor with multiple elements. If the left operand of `and`, `or` is variable `Tensor`, the right operand must also be `Tensor` with single element. Otherwise, there is no requirement for right operand.

- If the left or right operand of `and`, `or` is object that the graph mode does not support (such as third-party object and object created by syntax that is not native-supported in the graph mode), both operands need to be constant.

### Compare Operators

| Compare Operator | Supported Type|
| :----------- |:--------|
| `in` |`Number` in `tuple`, `String` in `tuple`, `Tensor` in `Tuple`, `Number` in `List`, `String` in `List`, `Tensor` in `List`, and `String` in `Dictionary`.|
| `not in` | Same as `in`. |
| `is` | The value can only be `None`, `True`, or `False`. |
| `is not` | The value can only be `None`, `True`, or `False`. |
| < | `Number` < `Number`, `Number` < `Tensor`, `Tensor` < `Tensor`, `Tensor` < `Number`. |
| <= | `Number` <= `Number`, `Number` <= `Tensor`, `Tensor` <= `Tensor`, `Tensor` <= `Number`. |
| > | `Number` > `Number`, `Number` > `Tensor`, `Tensor` > `Tensor`, `Tensor` > `Number`. |
| >= | `Number` >= `Number`, `Number` >= `Tensor`, `Tensor` >= `Tensor`, `Tensor` >= `Number`. |
| != | `Number` != `Number` , `Number` != `Tensor`, `Tensor` != `Tensor`, `Tensor` != `Number`, `mstype` != `mstype`, `String` != `String`, `Tuple !` = `Tuple`, `List` != `List`. |
| == | `Number` == `Number`, `Number` == `Tensor`, `Tensor` == `Tensor`, `Tensor` == `Number`, `mstype` == `mstype`, `String` == `String`, `Tuple` == `Tuple`, `List` == `List`. |

Restrictions:

- For operators `<`, `<=`, `>`, `>=`, `!=`, if all operators are `Number` type, value of Number can't be `Bool`.
- For operators `<`, `<=`, `>`, `>=`, `!=`, `==`, if all operands are `Number` type, operations between  `Float64` and `Int32` are not supported.
- For operators `<`, `<=`, `>`, `>=`, `!=`, `==`, if either operand is `Tensor` type, left and right operands can't both be `Bool` value.
- For operator `==`, if all operands are `Number` type,  support both `Number` have `Bool` value, not support only one `Number` has `Bool` value.
- For operators `!=`, `==`, all supported types but `mstype` can compare with `None`.
- The chain comparison like: `a>b>c` is not supported.

## Compound Statements

### Conditional Control Statements

#### if Statements

Usage:

- `if (cond): statements...`

- `x = y if (cond) else z`

Parameter: `cond` -- Variables of `Bool` type and constants of `Bool`, `List`, `Tuple`, `Dict` and `String` types are supported.

Restrictions:

- If `cond` is not a constant, the variable or constant assigned to a same sign in different branches should have same data type.If the data type of assigned variables or constants is `Tensor`, the variables and constants should have same shape and element type.

- The number of `if` cannot exceed 100.

Example 1:

```python
import mindspore as ms
from mindspore import ms_function

x = ms.Tensor([1, 2], ms.int32)
y = ms.Tensor([0, 3], ms.int32)
m = 'xx'
n = 'yy'

@ms_function()
def test_cond(x, y):
    if (x > y).any():
        return m
    else:
        return n

ret = test_cond(x, y)
print('ret:{}'.format(ret))if (x > y).any():
  return m
else:
  return n
```

The data type of `m` returned by the `if` branch and `n` returned by the `else` branch must be same.

The result is as follows:

```text
ret:xx
```

Example 2:

```python
import mindspore as ms
from mindspore import ms_function

x = ms.Tensor([1, 2], ms.int32)
y = ms.Tensor([0, 3], ms.int32)
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
```

The variable or constant `m` assigned to `out` in `if` branch and the variable or constant `n` assigned to out in `false` branch must have same data type.

The result is as follows:

```text
ret:xx
```

Example 3:

```python
import mindspore as ms
from mindspore import ms_function

x = ms.Tensor([1, 2], ms.int32)
y = ms.Tensor([0, 3], ms.int32)
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

The variable or constant `m` assigned to `out` in `if` branch and the variable or constant `init` initially assigned to `out` must have same data type.

The result is as follows:

```text
ret:xx
```

### Loop Statements

#### for Statements

Usage:

- `for i in sequence  statements...`

- `for i in sequence  statements... if (cond) break`

- `for i in sequence  statements... if (cond) continue`

Parameter: `sequence` -- Iterative sequences (`Tuple`, `List`, `range` and so on).

Restrictions:

- The total number of graph operations is a multiple of number of iterations of the `for` loop. Excessive number of iterations of the `for` loop may cause the graph to occupy more memory than usage limit.

- The `for...else...` statement is not supported.

Example:

```python
import numpy as np
from mindspore import ms_function

z = ms.Tensor(np.ones((2, 3)))

@ms_function()
def test_cond():
    x = (1, 2, 3)
    for i in x:
        z += i
    return z

ret = test_cond()
print('ret:{}'.format(ret))
```

The result is as follows:

```text
ret:[[7. 7. 7.]
 [7. 7. 7.]]
```

#### while Statements

Usage:

- `while (cond)  statements...`

- `while (cond)  statements... if (cond1) break`

- `while (cond)  statements... if (cond1) continue`

Parameter: `cond` -- Variables of `Bool` type and constants of `Bool`, `List`, `Tuple`, `Dict` and `String` types are supported.

Restrictions:

- If `cond` is not a constant, the variable or constant assigned to a same sign inside body of `while` and outside body of `while` should have same data type.If the data type of assigned variables or constants is `Tensor`, the variables and constants should have same shape and element type.

- The `while...else...` statement is not supported.

- If `cond` is not a constant, in while body, the data with type of `Number`, `List`, `Tuple` are not allowed to update and the shape  of `Tensor` data are not allowed to change.

- The number of `while` cannot exceed 100.

Example 1:

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

The data type of `m` returned inside `while` and data type of `n` returned outside `while` must have same data type.

The result is as follows:

```text
ret:1
```

Example 2:

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

The variable `op1` assigned to `out` inside `while` and the variable or constant `init` initially assigned to `out` must have same data type.

The result is as follows:

```text
ret:15
```

### Function Definition Statements

#### def Keyword

Defines functions.

Usage:

`def function_name(args): statements...`

For example:

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

The result is as follows:

```text
ret: 6
```

Restrictions:

- The defined function must has `return` statement.
- `Construct` function of the outermost network is not support  kwargs, like:`def construct(**kwargs):`.
- Mixed use of variable argument and non-variable argument is not supported, like:`def function(x, y, *args)` and `def function(x = 1, y = 1, **kwargs)`.

#### lambda Expression

Generates functions.

Usage: `lambda x, y: x + y`

For example:

```python
from mindspore import ms_function

@ms_function()
def test(x, y):
    number_add = lambda x, y: x + y
    return number_add(x, y)

ret = test(1, 5)
print('ret:{}'.format(ret))
```

The result is as follows:

```text
ret: 6
```

### List Comprehension and Generator Expression

Support List Comprehension and Generator Expression.

#### List Comprehension

Generates a list. Own to the implicit converting during compiling, the result of expression is a tuple.

Usage: refer to Python official syntax description.

For example:

```python
from mindspore import ms_function

@ms_function()
def test(x, y):
    l = [x * x for x in range(1, 11) if x % 2 == 0]
    return l

ret = test(1, 5)
print('ret:{}'.format(ret))
```

The result is as follows:

```text
ret:(4, 16, 36, 64, 100)
```

Restrictions:

Use multiple nested iterations comprehension in the generator.

For example (Use two nested iterations):

```python
l = [y for x in ((1, 2), (3, 4), (5, 6)) for y in x]
```

The result would be:

```text
TypeError:  The `generators` supports one `comprehension` in ListComp/GeneratorExp, but got 2 comprehensions.
```

#### Generator Expression

Generates a list. The same as List Comprehension. The expression would generate a list immediately, not like the behavior running in Python.

Usage: Referencing List Comprehension.

For example:

```python
from mindspore import ms_function

@ms_function()
def test(x, y):
    l = (x * x for x in range(1, 11) if x % 2 == 0)
    return l

ret = test(1, 5)
print('ret:{}'.format(ret))
```

The result is as follows:

```text
ret:(4, 16, 36, 64, 100)
```

Restrictions: The same as List Comprehension.

## Functions

### Python Built-in Functions

Currently, the following built-in Python functions are supported: `int`, `float`, `bool`, `str`, `list`, `tuple`, `getattr`, `hasattr`, `len`, `isinstance`, `all`, `round`, `any`, `max`, `min`, `sum`, `abs`, `partial`, `map`, `range`, `enumerate`, `super`, `pow`, and `filter`. The usage of built-in function is similar to the usage of  corresponding Python built-in function.

#### int

Return the integer value based on the input number or string.

Calling: `int(x=0, base=10)`

Input parameter:

- `x` -- the object need to be converted to integer, the valid type of x includes `int`, `float`, `bool`, `str`, constant `Tensor` and third-party object (such as `numpy.ndarray`).

- `base` -- the base to convert. `base` is only allowed when `x` is `str`.

Return value: the converted integer.

For example:

```python
from mindspore import ms_function

@ms_function
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

The result is as follows:

```text
a: 3
b: 3
c: 18
d: 10
e: 8
```

#### float

Return the floating-point number based on the input number or string.

Calling: `float(x=0)`

Input parameter: `x` -- the object need to be converted to floating number, the valid type of x includes `int`, `float`, `bool`, `str`, constant `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: the converted floating-point number.

For example:

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
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
```

The result is as follows:

```text
a: 1.0
b: 112.0
c: -123.6
d: 123.0
```

#### bool

Return the boolean value based on the input.

Calling: `bool(x=false)`

Input parameter: `x` -- the object need to be converted to boolean value, the valid type of x includes `int`, `float`, `bool`, `str`, `list`, `tuple`, `dict`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: if `x` is not `Tensor`, returns the converted boolean scalar. Otherwise, returns boolean `Tensor`.

For example:

```python
from mindspore import ms_function, Tensor

@ms_function
def func():
   a = bool()
   b = bool(0)
   c = bool("abc")
   d = bool([1, 2, 3, 4])
   e = bool(Tensor([10]))
   return a, b, c, d, e

a, b, c, d, e = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
```

The result is as follows:

```text
a: False
b: False
c: True
d: True
e: [True]    # e is boolean Tensor
```

#### str

Return the string value based on the input.

Calling: `str(x='')`

Input parameter: `x` -- the object need to be converted to string value, the valid type of x includes `int`, `float`, `bool`, `str`, `list`, `tuple`, `dict`, `Tensor` and third-party object (such as `numpy.ndarray`). `list`, `tuple` and `dict` can not contain non-constant element.

Return value: string converted from `x`.

For example:

```python
import numpy as np
from mindspore import ms_function, Tensor

@ms_function
def func():
   a = str()
   b = str(0)
   c = str([1, 2, 3, 4])
   d = str(Tensor([10]))
   e = str(np.array([1, 2, 3, 4]))
   return a, b, c, d, e

a, b, c, d, e = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
```

The result is as follows:

```text
a:                                             # a is empty string
b: 0
c: [1, 2, 3, 4]
d: Tensor(shape=[1], dtype=Int64, value=[10])
e: [1 2 3 4]
```

#### tuple

Return a tuple based on the input object.

Calling: `tuple(x=())`

Input parameter: `x` -- the object that need to be converted to tuple, the valid type of x includes `Tuple`, `List`, `Dictionary`, `Tensor` or third-party object (such as numpy.ndarray).

Return value: tuple with elements of `x`, `x` is cut based on zero dimension.

For example:

```python
import numpy as np
import mindspore as ms
from mindspore import ms_function

@ms_function
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

The result is as follows:

```text
a: (1, 2, 3)
b: (1, 2, 3)
c: ('a', 'b', 'c')
d: (Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 2), Tensor(shape=[], dtype=Int64, value= 3))
```

#### list

Return a list based on the input object.

Calling: `list(x=())`

Input parameter: `x` -- the object that need to be converted to list, the valid type of x includes `Tuple`, `List`, `Dictionary`, `Tensor` or third-party object (such as numpy.ndarray).

Return value: list with elements of `x`, `x` is cut based on zero dimension.

For example:

```python
import mindspore as ms
from mindspore import ms_function

@ms_function
def func():
   a = list((1, 2, 3))
   b = list(np.array([1, 2, 3]))
   c = list({'a':1, 'b':2, 'c':3})
   d = list(ms.Tensor([1, 2, 3]))
   return a, b, c, d
a_t, b_t, c_t, d_t = func()
print("a_t: ", a)
print("b_t: ", b)
print("c_t: ", c)
print("d_t: ", d)
```

The result is as follows:

```text
a_t: (1, 2, 3)
b_t: (1, 2, 3)
c_t: ('a', 'b', 'c')
d_t: (Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 2), Tensor(shape=[], dtype=Int64, value= 3))
```

In graph mode, if the return values contain list, the list will be converted to tuple automatically. So the `a_t`, `b_t`, `c_t` and `d_t` in the above example are all tuple. However, `a`, `b`, `c` and `d` are still list.

#### getattr

Get the attribute of python object.

Calling: `getattr(x, attr, default)`

Input parameter:

- `x` -- The object to get attribute, `x` can be all types that graph mode supports. `x` can not be third-party object.

- `attr` -- The name of the attribute, the type of `attr` should be `str`.

- `default` -- Optional input. If `x` do not have `attr`, `getattr` will return `default`. `default` can be all types that graph mode supports but can not be third-party object. If `default` is not set and `x` does not have attribute `attr`, AttributeError will be raised.

Return value: Target attribute or `default`.

For example:

```python
import mindspore as ms
from mindspore import ms_function, ms_class

@ms_class
class MSClass1:
  def __init__(self):
    self.num0 = 0

ms_obj = MSClass1()

@ms_function
def func():
   a = getattr(ms_obj, 'num0')
   b = getattr(ms_obj, 'num1', 2)
   return a, b

a, b = func()
print("a: ", a)
print("b: ", b)
```

The result is as follows:

```text
a: 0
b: 2
```

The attribute of object in graph mode may be different from that in pynative mode. It is suggested to use `default` input in `getattr` or call `hasattr` before using `getattr` to avoid AttributeError.

#### hasattr

Judge whether an object has an attribute.

Calling: `hasattr(x, attr)`

Input parameter:

- `x` -- The object to get attribute, `x` can be all types that graph mode supports and also can be third-party object.

- `attr` -- The name of the attribute, the type of `attr` should be `str`.

Return value: boolean value indicates whether `x` has `attr`.

For example:

```python
import mindspore as ms
from mindspore import ms_function, ms_class

@ms_class
class MSClass1:
  def __init__(self):
    self.num0 = 0

ms_obj = MSClass1()

@ms_function
def func():
   a = hasattr(ms_obj, 'num0')
   b = hasattr(ms_obj, 'num1')
   return a, b

a, b = func()
print("a: ", a)
print("b: ", b)
```

The result is as follows:

```text
a: True
b: False
```

#### len

Return the length of a sequence.

Calling: `len(sequence)`

Input parameter: `sequence` -- `Tuple`, `List`, `Dictionary`, `Tensor` or third-party object (such as numpy.ndarray).

Return value: length of the sequence, which is of the `int` type. If the input parameter is `Tensor`, the length of dimension 0 is returned.

For example:

```python
import mindspore as ms
import numpy as np
from mindspore import ms_function

z = ms.Tensor(np.ones((6, 4, 5)))

@ms_function()
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

The result is as follows:

```text
x_len:3
y_len:3
d_len:2
z_len:6
n_len:4
```

#### isinstance

Determines whether an object is an instance of a class. Different from operator `Isinstance`, the second input parameter of `Isinstance` is the type defined in the `dtype` module of MindSpore.

Calling: `isinstance(obj, type)`

Input parameters:

- `obj` -- Any instance of any supported type.

- `type` -- A type in the `MindSpore dtype` module.

Return value: If `obj` is an instance of `type`, return `True`. Otherwise, return `False`.

For example:

```python
import mindspore as ms
import numpy as np
from mindspore import ms_function

z = ms.Tensor(np.ones((6, 4, 5)))

@ms_function()
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

The result is as follows:

```text
x_is_tuple:True
y_is_list:True
z_is_tensor:True
```

#### all

Judge whether all of the elements in the input is true.

Calling: `all(x)`

Input parameter: - `x` -- Iterable object, the valid types include `list`, `tuple`, `dict`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: boolean, indicates whether all of the elements in the input is true.

For example:

```python
from mindspore import ms_function

@ms_function
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

The result is as follows:

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

Judge whether any of the elements in the input is true.

Calling: `any(x)`

Input parameter: - `x` -- Iterable object, the valid types include `list`, `tuple`, `dict`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: boolean, indicates whether any of the elements in the input is true.

For example:

```python
from mindspore import ms_function

@ms_function
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

The result is as follows:

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

Return the rounding value of input.

Calling: `round(x, digit=0)`

Input parameter:

- `x` -- the object to rounded, the valid types include `int`, `float`, `bool`, `Tensor` and third-party object that defines magic function `__round__()`.

- `digit` -- the number of decimal places to round, the default value is 0. `digit` can be `int` object or `None`. If `x` is `Tensor`, then `round()` does not support input `digit`.

Return value: the value after rounding.

For example:

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
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: {:.2f}".format(e))
print("f: {:.2f}".format(f))
print("g: {:.2f}".format(g))
print("h: {:.2f}".format(h))
```

The result is as follows:

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

Return the maximum of inputs.

Calling: `max(*data)`

Input parameter: - `*data` -- If `*data` is single input, `max` will compare all elements within `data` and `data` must be iterable object. If there are multiple inputs, then `max()` will compare each of them. The valid types of `data` include `int`, `float`, `bool`, `list`, `tuple`, `dict`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: boolean, the maximum of the inputs.

For example:

```python
import numpy as np
import mindspore as ms
from mindspore import ms_function

@ms_function
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

The result is as follows:

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

Return the minimum of inputs.

Calling: `min(*data)`

Input parameter: - `*data` -- If `*data` is single input, then `min()` will compare all elements within `data` and `data` must be iterable object. If there are multiple inputs, then `min()` will compare each of them. The valid types of `data` include `int`, `float`, `bool`, `list`, `tuple`, `dict`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: boolean, the minimum of the inputs.

For example:

```python
import numpy as np
import mindspore as ms
from mindspore import ms_function

@ms_function
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

The result is as follows:

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

Return the sum of input sequence.

Calling: `sum(x, n=0)`

Input parameter:

- `x` -- iterable with numbers, the valid types include `list`, `tuple`, `Tensor` and third-party object (such as `numpy.ndarray`).

- `n` -- the number that will be added to the sum of `x`, which is assumed to be 0 if not given.

Return value: the value obtained by summing `x` and adding it to `n`.

For example:

```python
import numpy as np
import mindspore as ms
from mindspore import ms_function, Tensor

@ms_function
def func():
  a = sum([0, 1, 2])
  b = sum((0, 1, 2), 10)
  c = sum(np.array([1, 2, 3]))
  d = sum(Tensor([1, 2, 3]), 10)
  e = sum(Tensor([[1, 2], [3, 4]]))
  f = sum([1, Tensor([[1, 2], [3, 4]]), Tensor([[1, 2], [3, 4]])], Tensor([[1, 1], [1, 1]]))
  return a, b, ms.Tensor(c), d, e, f

a, b, c, d, e, f = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
print("f: ", f)
```

The result is as follows:

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

Return the absolute value of the input. The usage of `abs()` is the same as python built-in function `abs()`.

Calling: `abs(x)`

Input parameter: - `x` -- The valid types of `x` include `int`, `float`, `bool`, `complex`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: the absolute value of the input.

For example:

```python
from mindspore import ms_function

@ms_function
def func():
   a = abs(-45)
   b = abs(100.12)
   return a, b

a, b = func()
print("a: ", a)
print("b: {:.2f}".format(b))
```

The result is as follows:

```text
a: 45
b: 100.12
```

#### partial

A partial function used to fix the input parameter of the function.

Calling: `partial(func, arg, ...)`

Input parameters:

- `func` --Function.

- `arg` -- One or more parameters to be fixed. Position parameters and key-value pairs can be specified.

Return value: functions with certain input parameter values fixed

For example:

```python
from mindspore import ops
from mindspore import ms_function

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

The result is as follows:

```text
m:5
n:7
```

#### map

Maps one or more sequences based on the provided functions and generates a new sequence based on the mapping result.
If the number of elements in multiple sequences is inconsistent, the length of the new sequence is the same as that of the shortest sequence.

Calling: `map(func, sequence, ...)`

Input parameters:

- `func` -- Function.

- `sequence` -- One or more sequences (`Tuple` or `List`).

Return value: A `Tuple`

For example:

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

The result is as follows:

```text
ret: (5, 7, 9)
```

#### zip

Packs elements in the corresponding positions in multiple sequences into tuples, and then uses these tuples to form a new sequence.
If the number of elements in each sequence is inconsistent, the length of the new sequence is the same as that of the shortest sequence.

Calling: `zip(sequence, ...)`

Input parameter: `sequence` -- One or more sequences (`Tuple` or `List`)`.

Return value: A `Tuple`

For example:

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

The result is as follows:

```text
ret:((1, 4), (2, 5), (3, 6))
```

#### range

Creates a `Tuple` based on the start value, end value, and step.

Calling:

- `range(start, stop, step)`

- `range(start, stop)`

- `range(stop)`

Input parameters:

- `start` -- start value of the count. The type is `int`. The default value is 0.

- `stop` -- end value of the count (exclusive). The type is `int`.

- `step` -- Step. The type is `int`. The default value is 1.

Return value: A `Tuple`

For example:

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

The result is as follows:

```text
x:(0, 2, 4)
y:(0, 1, 2, 3, 4)
z:(0, 1, 2)
```

#### enumerate

Generates an index sequence of a sequence. The index sequence contains data and the corresponding subscript.

Calling:

- `enumerate(sequence, start)`

- `enumerate(sequence)`

Input parameters:

- `sequence` -- A sequence (`Tuple`, `List`, or `Tensor`).

- `start` -- Start position of the subscript. The type is `int`. The default value is 0.

Return value: A `Tuple`

For example:

```python
import mindspore as ms
import numpy as np
from mindspore import ms_function

y = ms.Tensor(np.array([[1, 2], [3, 4], [5, 6]]))

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

The result is as follows:

```text
m:((3, 100), (4, 200), (5, 300), (6, 400))
n:((0, Tensor(shape=[2], dtype=Int64, value= [1, 2])), (1, Tensor(shape=[2], dtype=Int64, value= [3, 4])), (2, Tensor(shape=[2], dtype=Int64, value= [5, 6])))
```

#### super

Calls a method of the parent class (super class). Generally, the method of the parent class is called after `super`.

Calling:

- `super().xxx()`

- `super(type, self).xxx()`

Input parameters:

- `type` -- Class.

- `self` -- Object.

Return value: method of the parent class.

For example:

```python
from mindspore import nn, context

context.set_context(mode=context.GRAPH_MODE)

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

Return the power.

Calling: `pow(x, y)`

Input parameters:

- `x` -- Base number, `Number`, or `Tensor`.

- `y` -- Power exponent, `Number`, or `Tensor`.

Return value: `y` power of `x`, `Number`, or `Tensor`

For example:

```python
import mindspore as ms
import numpy as np
from mindspore import ms_function

x = ms.Tensor(np.array([1, 2, 3]))
y = ms.Tensor(np.array([1, 2, 3]))

@ms_function()
def test(x, y):
    return pow(x, y)

ret = test(x, y)

print('ret:{}'.format(ret))
```

The result is as follows:

```text
ret:[ 1  4 27]
```

#### print

Prints logs.

Calling: `print(arg, ...)`

Input parameter: `arg` -- Information to be printed (`int`, `float`, `bool`, `String` or `Tensor`).
When the `arg` is `int`, `float`, or `bool`, it will be printed out as a `0-D` tensor.

Return value: none

For example:

```python
import mindspore as ms
import numpy as np
from mindspore import ms_function

x = ms.Tensor(np.array([1, 2, 3]), ms.int32)
y = ms.Tensor(3, ms.int32)

@ms_function()
def test(x, y):
    print(x)
    print(y)
    return x, y

ret = test(x, y)
```

The result is as follows:

```text
Tensor(shape=[3], dtype=Int32, value= [1 2 3])
3
```

#### filter

According to the provided function to judge the elements of a sequence. Each element is passed into the function as a parameter in turn, and the elements whose return result is not 0 or False form a new sequence.

Calling: `filter(func, sequence)`

Input parameters:

- `func` -- Function.
- `sequence` -- A sequence (`Tuple` or `List`).

Return value: A `Tuple`.

For example:

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

The result is as follows:

```text
ret:(1, 3, 5)
```

### Function Parameters

- Default parameter value: The data types `int`, `float`, `bool`, `None`, `str`, `tuple`, `list`, and `dict` are supported, whereas `Tensor` is not supported.

- Variable parameters: Inference and training of networks with variable parameters are supported.

- Key-value pair parameter: Functions with key-value pair parameters cannot be used for backward propagation on computational graphs.

- Variable key-value pair parameter: Functions with variable key-value pairs cannot be used for backward propagation on computational graphs.

## Network Definition

### Network Input parameters

The input parameters of the outermost network only can be `lool`, `int`, `float`, `Tensor`, `None`, `mstype.number(mstype.bool, mstype.int, mstype.float, mstype.uint)`, `List` or `Tuple` that contains these types, and `Dictionary` whose values are these types.

While calculating gradient for outermost network, only `Tensor` input could be calculated, input of other type will be ignored. For example, input parameter `(x, y,  z)` of outermost network, `x` and `z` are `Tensor` type, `y` is other type. While calculating gradient for the network, only gradients of `x` and `z` are calculated, and `(grad_x, grad_y)` is returned.
If you want to use other types of input for the network, please transfer them to the network while initializing network in the `__init__` function, and save them as network attributes, then use  in the `construct`.

The input parameters of inner network do not have this restriction.

For example:

```python
import mindspore as ms
from mindspore import nn, ops
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
input_x = ms.Tensor(np.ones((2, 3)).astype(np.float32))
input_y = 2
input_z = ms.Tensor(np.ones((2, 3)).astype(np.float32) * 2)

net = Net(flag)
grad_net = GradNet(net)
ret = grad_net(input_x, input_y, input_z)

print('ret:{}'.format(ret))
```

The result is as follows:

```text
ret:(Tensor(shape=[2, 3], dtype=Float32, value=
[[ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00],
 [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00]]), Tensor(shape=[2, 3], dtype=Float32, value=
[[ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00],
 [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00]]))
```

In the `Net` defined above,  `string` flag is transferred during initialization and saved as attribute `self.flag`, then used in the `construct`.

The input parameter `x` and `z` are `Tensor`, `y` is `int`. While `grad_net` calculates gradient of the input parameters `(x, y, z)` for the outermost network, gradient of `y` is automatically ignored, only the gradient of `x` and `z` is calculated, `ret = (grad_x, grad_z)`.

### Instance Types on the Entire Network

- Common Python function with the [@ms_function](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.ms_function.html) decorator.

- Cell subclass inherited from [nn.Cell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html).

### Network Construction Components

| Category                             | Content                                                      |
| :----------------------------------- | :----------------------------------------------------------- |
| `Cell` instance                      | [mindspore/nn/*](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html) and user-defined [Cell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html). |
| Member function of a `Cell` instance | Member functions of other classes in the construct function of Cell can be called. |
| `Primitive` operator                 | Class decorated with [@ms_class](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.ms_class.html). |
| `Composite` operator                 | [mindspore/ops/operations/*](https://www.mindspore.cn/docs/en/master/api_python/mindspore.ops.html) |
| `constexpr` generation operator      | [mindspore/ops/composite/*](https://www.mindspore.cn/docs/en/master/api_python/mindspore.ops.html) |
| `constexpr`生成算子                  | Value computation operator generated by [@constexpr](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.constexpr.html). |
| Function                             | User-defined Python functions and system functions listed in the preceding content. |

### Network Constraints

1. You are not allowed to modify non-`Parameter` data members of the network.

   For example:

   ```python
   import mindspore as ms
   from mindspore import nn
   import numpy as np

   class Net(nn.Cell):
       def __init__(self):
           super(Net, self).__init__()
           self.x = 2
           self.par = ms.Parameter(ms.Tensor(np.ones((2, 3, 4))), name="par")

       def construct(self, x, y):
           self.par[0] = y
           self.x = x
           return x + y

   net = Net()
   net(1, 2)
   ```

   In the preceding defined network, `self.x` is not a `Parameter` and cannot be modified. `self.par` is a `Parameter` and can be modified.

   The result would be:

   ```Text
   TypeError: 'self.x' should be initialized as a 'Parameter' type in the '__init__' function
   ```

2. When an undefined class member is used in the `construct` function, `AttributeError` is not thrown like the Python interpreter. Instead, it is processed as `None`.

   For example:

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

   In the preceding defined network, `construct` uses the undefined class member `self.y`. In this case, `self.y` is processed as `None`.

   The result would be:

   ```text
   RuntimeError: mindspore/ccsrc/frontend/operator/composite/multitype_funcgraph.cc:161 GenerateFromTypes] The 'add' operation does not support the type [Int64, kMetaTypeNone]
   ```
