# Static Graph Syntax Support

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/static_graph_syntax_support.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

In graph mode, Python code is not executed by the Python interpreter. Instead, the code is compiled into a static computation graph, and then the static computation graph is executed.

Currently, only the function, Cell, and subclass instances modified by the `@ms_function` decorator can be built.
For a function, build the function definition. For the network, build the `construct` method and other methods or functions called by the `construct` method.

For details about how to use `ms_function`, click <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.ms_function.html#mindspore.ms_function>.

For details about the definition of `Cell`, click <https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html>.

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

Forcible conversion to `Number` is not supported on the network. That is, the syntax `y = int(x)`, `y = float(x)`, and `y = bool(x)` are not supported.

#### String

`String` can be constructed on the network. That is, the syntax `y = "abcd"` is supported.

Can use str() to change the constant value to string, str.format() can use to format the string, but not supported to input a kwarts type arguments and the argument of format function cannot be a variable.

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
    x = [[1,2,3],4,5]
    b = x[0]
    b[0] = 123123
    return x

x = test_list()
print('x:{}'.format(x))
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

- Supported index values and value assignment

  Single-level and multi-level index values and value assignment are supported.

  The index value supports only `int` and `slice`.

  The element of `slice` data should be constant that can be deduced in the state of compiling graph.

  The assigned value can be `Number`, `String`, `Tuple`, `List`, or `Tensor`.

  When the value of the current slice is Tensor, the Tensor needs to be converted to a List, which is currently implemented through JIT Fallback. Therefore, variable scenarios cannot be supported temporarily.

  For example:

  ```python
  from mindspore import ms_function, Tensor
  import numpy as np

  t = Tensor(np.array([1, 2, 3]))

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

  The result is as follows:

  ```text
  y:3
  z:[1 2 3]
  m:((2, 3, 4), 3, 4)
  n:(2, 3, 4)
  ```

  An example of the `Tensor` index is as follows:

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

  `items`: extracts `Tuple` composed of each pair of `value` values and `key` values in `dict` to form `Tuple` and return.

  For example:

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

  The result is as follows:

  ```text
  x:{'a': (2, 3, 4), 'b': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6]), 'c': Tensor(shape=[3], dtype=Int64, value= [7, 8, 9])}
  y:[4 5 6]
  ```

### MindSpore User-defined Data Types

Currently, MindSpore supports the following user-defined data types: `Tensor`, `Primitive`, and `Cell`.

#### Tensor

Currently, tensors cannot be constructed on the network. That is, the syntax `x = Tensor(args...)` is not supported.

You can use the `@constexpr` decorator to modify the function and generate the `Tensor` in the function.

For details about how to use `@constexpr`, click <https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.constexpr.html>.

The constant `Tensor` used on the network can be used as a network attribute and defined in `init`, that is, `self.x = Tensor(args...)`. Then the constant can be used in `construct`.

In the following example, `Tensor` of `shape = (3, 4), dtype = int64` is generated by `@constexpr`.

```python
from mindspore import Tensor
from mindspore.ops import constexpr

@constexpr
def generate_tensor():
    return Tensor(np.ones((3, 4)))

x = generate_tensor()
print('x:{}'.format(x))
```

The result is as follows:

```text
x:[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
```

The following describes the attributes, APIs supported by the `Tensor`.

- Supported attributes

  `shape`: obtains the shape of `Tensor` and returns a `Tuple`.

  `dtype`: obtains the data type of `Tensor` and returns a data type defined by `MindSpore`.

- Supported APIs

  `all`: reduces `Tensor` through the `all` operation. Only `Tensor` of the `Bool` type is supported.

  `any`: reduces `Tensor` through the `any` operation. Only `Tensor` of the `Bool` type is supported.

`view`: reshapes `Tensor` into input `shape`.

  `expand_as`: expands `Tensor` to the same `shape` as another `Tensor` based on the broadcast rule.

  For example:

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

  The result is as follows:

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

Currently, `Primitive` and its subclass instances can be constructed on the network. That is, the `reduce_sum = ReduceSum(True)` syntax is supported.

However, during construction, the parameter can be specified only in position parameter mode, and cannot be specified in the key-value pair mode. That is, the syntax `reduce_sum = ReduceSum(keep_dims=True)` is not supported.

Currently, the attributes and APIs related to `Primitive` and its subclasses cannot be called on the network.

For details about the definition of `Primitive`, click <https://www.mindspore.cn/tutorials/experts/en/master/operation/op_classification.html>.

For details about the defined `Primitive`, click <https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Primitive.html#mindspore.ops.Primitive>.

#### Cell

Currently, `Cell` and its subclass instances can be constructed on the network. That is, the syntax `cell = Cell(args...)` is supported.

However, during construction, the parameter can be specified only in position parameter mode, and cannot be specified in the key-value pair mode. That is, the syntax `cell = Cell(arg_name=value)` is not supported.

Currently, the attributes and APIs related to `Cell` and its subclasses cannot be called on the network unless they are called through `self` in `construct` of `Cell`.

For details about the definition of `Cell`, click <https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html>.

For details about the defined `Cell`, click <https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell>.

#### Parameter

`Parameter` is a variable tensor, indicating the parameters that need to be updated during network training.

For details about the definition of `Parameter`：<https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Parameter.html#mindspore.Parameter>

## Primaries

Primaries represent the most tightly bound operations of the language Which contains `Attribute references`, `Subscriptions`, `Calls`.

### Attribute References

An attribute reference is a primary followed by a period and a name.

In `Cell` instance of MindSpore, using attribute reference as left operands must meet the restrictions below:

- The attribute must belong to self, such as self.xxx. It is not supported to change attribute of other instance.

- The attribute type must be `Parameter` and be initialized in `__init__` function.

For example:

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

Index value of `Tensor` refers to  <https://www.mindspore.cn/docs/en/master/note/index_support.html#index-values>

### Calls

A call calls a callable object (e.g., `Cell` or `Primitive`) with a possibly empty series of arguments.

For example:

```python
from mindspore import Tensor, nn, dtype, ops
import numpy as np

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = ops.MatMul()

    def construct(self, x, y):
        out = self.matmul(x, y)  # A call of Primitive
        return out

x = Tensor(np.ones(shape=[1, 3]), dtype.float32)
y = Tensor(np.ones(shape=[3, 4]), dtype.float32)
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

- For operator `and`, `or`,  if left operand is a `Tensor`,  right operand should be `Tensor` which has same data type with left operand, and both`Tensor` must have only one element.
- For operator `and`, `or`,  if left operand not `Tensor`, right operand can be any supported type.

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
```

The variable or constant `m` assigned to `out` in `if` branch and the variable or constant `n` assigned to out in `false` branch must have same data type.

The result is as follows:

```text
ret:xx
```

Example 3:

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

Currently, the following built-in Python functions are supported: `len`, `isinstance`, `partial`, `map`, `range`, `enumerate`, `super`, `pow`, and `filter`.

#### len

Returns the length of a sequence.

Calling: `len(sequence)`

Input parameter: `sequence` -- `Tuple`, `List`, `Dictionary`, or `Tensor`.

Return value: length of the sequence, which is of the `int` type. If the input parameter is `Tensor`, the length of dimension 0 is returned.

For example:

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

The result is as follows:

```text
x_len:3
y_len:3
d_len:2
z_len:6
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

The result is as follows:

```text
x_is_tuple:True
y_is_list:True
z_is_tensor:True
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

- `func` --Function.

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

- `type` --Class.

- `self` --Object.

Return value: method of the parent class.

For example:

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

Returns the power.

Calling: `pow(x, y)`

Input parameters:

- `x` -- Base number, `Number`, or `Tensor`.

- `y` -- Power exponent, `Number`, or `Tensor`.

Return value: `y` power of `x`, `Number`, or `Tensor`

For example:

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

The result is as follows:

```text
ret:(Tensor(shape=[2, 3], dtype=Float32, value=
[[ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00],
 [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00]]), Tensor(shape=[2, 3], dtype=Float32, value=
[[ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00],
 [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00]]))
```

In the `Net` defined above,  string `flag` is transferred during initialization and saved as attribute `self.flag`, then used in the construct. The input parameter `x` and `z` are `Tensor`, `y` is `int`. While `grad_net` calculates gradient of the input parameters for the outermost network, gradient of `y` is automatically ignored, only the gradient of x and z is calculated, ret = (grad_x, grad_z).

### Instance Types on the Entire Network

- Common Python function with the [@ms_function](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.ms_function.html) decorator.

- Cell subclass inherited from [nn.Cell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html).

### Network Construction Components

| Category                 | Content
| :-----------             |:--------
| `Cell` instance |[mindspore/nn/*](https://www.mindspore.cn/docs/en/master/api_python/mindspore.nn.html) and user-defined [Cell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html).
| Member function of a `Cell` instance | Member functions of other classes in the construct function of Cell can be called.
| `ms_class` instance | Class decorated with [@ms_class](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.ms_class.html).
| `Primitive` operator |[mindspore/ops/operations/*](https://www.mindspore.cn/docs/en/master/api_python/mindspore.ops.html)
| `Composite` operator |[mindspore/ops/composite/*](https://www.mindspore.cn/docs/en/master/api_python/mindspore.ops.html)
| `constexpr` generation operator | Value computation operator generated by [@constexpr](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.constexpr.html).
| Function                 | User-defined Python functions and system functions listed in the preceding content.

### Network Constraints

1. You are not allowed to modify non-`Parameter` data members of the network.

   For example:

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

   In the preceding defined network, `self.num` is not a `Parameter` and cannot be modified. `self.par` is a `Parameter` and can be modified.

   The result would be:

   ```Text
   TypeError: mindspore/ccsrc/pipeline/jit/parse/parse.cc:1740 HandleAssignClassMember] 'self.x' should be initialized as a 'Parameter' in the '__init__' function before assigning.
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
