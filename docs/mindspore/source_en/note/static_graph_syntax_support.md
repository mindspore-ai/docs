# Static Graph Syntax Support

<a href="https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/static_graph_syntax_support.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png"></a>

## Overview

In graph mode, Python code is not executed by the Python interpreter. Instead, the code is compiled into a static computation graph, and then the static computation graph is executed.

There are two ways to use the graph mode. The first way is to call the `@jit` decorator to modify a function or a class member method, and then the decorated function or method will be compiled into a static computation graph. The second way is to set `ms.set_context(mode=ms.GRAPH_MODE)`, then write the code in the `construct` function of the `Cell` so that the code in the `construct` function will be compiled into a static computation graph.

For details about how to use `jit`, click [jit API document](https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore/mindspore.jit.html#mindspore.jit).

For details about the definition of `Cell`, click [Cell API document](https://www.mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.Cell.html).

Due to syntax parsing restrictions, the supported data types, syntax, and related operations during graph building are not completely consistent with the Python syntax. As a result, some usage is restricted. JIT Fallback scheme considers the unification of static and dynamic graphs from the perspective of graph mode and extends the syntax capabilities of graph patterns. Borrowing the traditional JIT compilation idea, when it is found to be the Python syntax that is not supported in the graph mode, interpretation and execution are performed by Fallback. For more information, please refer to the [JIT Fallback](#jit-fallback) section.

The following describes the data types, syntax, and related operations supported during static graph building. These rules apply only to graph mode.

## Data Types

### Built-in Python Data Types

Currently, the following built-in `Python` data types are supported: `Number`, `String`, `List`, `Tuple`, and `Dictionary`.

#### Number

Supports `int`, `float`, and `bool`, but does not support `complex` numbers.

`Number` can be defined on the network. That is, the syntax `y = 1`, `y = 1.2`, and `y = True` are supported.

When the data is a constant, the value of the data can be achieved at compile time, the forcible conversion to `Number` is supported in the network. The syntax `y = int(x)`, `y = float(x)`, and `y = bool(x)` are supported.
When the data is a variable, i.e., you can get the value only at runtime. It also supports data type conversion using built-in functions [Python Built-in Functions](#python-built-in-functions) such as int(), float() and bool(). For example:

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

The result is as follows:

```text
res[0]: 11
res[0]: 10
res[2]: 2
```

#### String

`String` can be constructed on the network, i.e., support for using quotes (`'` or `"`) to create strings such as `x = 'abcd'` or `y = "efgh"`. Convert constants to strings by means of str(). Support string concatenation, truncation, and the use of membership operators (`in` or `not in`) to determine whether a string contains the specified character. Support for formatting string output by inserting a value into a string with the string format `%s`. Support for using the format string function str.format().

For example:

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

The result is as follows:

```text
res: ('H', 'Spore', 'Hello!MindSpore', 'MindSporeMindSpore', True, 'My name is MindSpore!', 'string is 123')
```

#### List

The `List` and the `Tuple` are the most basic sequence built-in types in Python, and the core difference between `List` and `Tuple` is that `List` is an object that can be changed, while `Tuple` is not, which means that `Tuple`, once created, cannot be changed without the address of the object remaining the same and a `List` can be modified without changing the address of the object through a series of inplace operations. For example:

```python
a = [1, 2, 3, 4]
a_id = id(a)
a.append(5)
a_after_id = id(a)
assert a_id == a_after_id
```

In the above sample code, through the `append`, the inplace syntax, to change the `List` object, the address of the object has not been modified. `Tuple` does not support this inplace operation. In the case of `JIT_SYNTAX_LEVEL` set to `COMPATIBLE` and `LAX`, the static graph mode can support inplace operation of some `List` objects.

MindSpore graph mode syntax extends support for `List` to facilitate network construction using `List`.

- The graph mode supports creating `Lists` in graph.

  Support creating `List` objects within graph mode, and the elements of the `List` objects can contain any of the types supported by the graph mode, as well as multiple levels of nesting. For example:

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

  The above sample code, all `List` objects can be created normally.

- The graph mode supports returning `List`

  Before MindSpore version 2.0, `List` is converted to `Tuple` when the graph mode returns a `List` object. In MindSpore version 2.0, `List` objects can be returned. For example:

  ```python
  import mindspore as ms

  @ms.jit
  def list_func():
      a = [1, 2, 3, 4]
      return d

  output = list_func()  # output: [1, 2, 3, 4]
  ```

  In the same way that a `List` is created within a graph mode, the graph mode returns a `List` object that can include any of the types supported by the graph mode, as well as multiple levels of nesting.

- The graph mode supports obtaining `List` objects from global variables

  In the following example, the static graph obtains the `List` object, performs the replace operation `list.reverse()` supported by the graph mode on the original object, and returns the original object. You can see that the object returned by the graph mode has the same id as the original global variable object, i.e. they are the same object. If `JIT_SYNTAX_LEVEL` is set to the `STRICT` option, the returned `List` object and the global object are two different objects.

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

- Graph mode supports `List` as input

  The graph mode supports `List` as input to static graphs. The elements of the `List` object used as input must be of an input type supported by the graph mode, which also supports multiple levels of nesting.

  ```python
  import mindspore as ms

  list_input = [1, 2, 3, 4]

  @ms.jit
  def list_func(x):
      return x

  output = list_func()  # output: [1, 2, 3, 4]
  ```

  Two precautions exist for `List` as static graph input:

  1. A `List`, as static graph input, is treated as a constant regardless of the type of the elements inside it.

  2. A `List`, as static graph input, copies `List` object for one time, and subsequent calculations are performed using that copied object, so it is not possible to perform an inplace operation on the original input object. For example:

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

  As shown in the above use case, the `List` object cannot perform inplace operation on the original object when it is graph mode input. The object returned by the graph mode is different from the input object id, which is a different object.

- The graph mode supports the built-in methods of List

    With `JIT_SYNTAX_LEVEL` set to `COMPATIBLE` as well as `LAX`, part of the `List` built-in function in graph mode supports inplace. All methods do not support inplace operations when `JIT_SYNTAX_LEVEL` is `STRICT`.
    The `List` built-in methods supported by the graph mode are shown in the following table:

    | Method names       | Whether the inplace operation is supported （JIT_SYNTAX_LEVEL=COMPATIBLE/LAX）    |  
    | ----------  | ------------      |
    | Index values      | Non-inplace operations      |
    | Index assignments      | No             |
    | append      | No             |
    | clear       | No             |
    | extend      | Yes               |
    | pop         | Yes               |
    | reverse     | Yes               |
    | insert      | Yes               |

    The details of the `List` built-in methods are described below:

    - List index values

        Basic syntax: ```element = list_object[index]```.

        Basic semantics: Extract the element at the `index` position in a `List` object (`index` starts at 0). Support multiple levels of index values.

        The index value `index` supports types `int`, `Tensor` and `slice`. Of these, inputs of type `int` and `Tensor` can support constants as well as variables, and the internal data of `slice` must be a constant that can be determined at compile time.

        The example is as follows:

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

        The result is as follows:

        ```text
        a:[1, 2]
        b:2
        c:[3, 4]
        ```

    - List index assignments

        Basic syntax: ```list_object[index] = target_element```.

        Basic semantics: Assigns the element at `index` in a `List` object to `target_element` (`index` starts at 0). Multiple levels of index assignment are supported.

        The index value `index` supports the types `int`, `Tensor` and `slice`. Of these, inputs of type `int` and `Tensor` can support constants as well as variables, and the internal data of `slice` must be a constant that can be determined at compile time.

        The index assignment object `target_element` supports all data types supported by the graph mode.

        Currently, `List` index assignment does not support inplace operation, and a new object will be created after index assignment. This operation will be supported later.

        The example is as follows:

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

        The result is as follows:

        ```text
        output:[[0, 88], 10, "ok", (1, 2, 3)]
        ```

    - List.append

        Basic syntax: ```list_object.append(target_element)```.

        Basic semantics: Appends element `target_element` to the end of `list` object `list_object`.

        Currently, `List.append` does not support the inplace operation, and a new object will be created after the index assignment. This operation will be supported later.

        The example is as follows:

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

        The result is as follows:

        ```text
        x:[1, 2, 3, 4]
        ```

    - List.clear

        Basic syntax: ```list_object.clear()```.

        Basic semantics: Clear the elements contained in the `List` object `list_object`.

        Currently, `List.clear` does not support inplace, and a new object will be created after the index assignment. This operation will support inplace in the future.

        The example is as follows:

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

        The result is as follows:

        ```text
        x:[]
        ```

    - List.extend

        Basic syntax: ```list_object.extend(target)```.

        Basic semantics: Insert all elements within `target` in order to the end of the `List` object `list_object`.

        The types supported for `target` are `Tuple`, `List` and `Tensor`. In this case, if `target` is of type `Tensor`, the `Tensor` will be converted to `List` before insertion.

        With `JIT_SYNTAX_LEVEL` set to `COMPATIBLE` as well as `LAX`, `List.extend` supports the inplace operation, which does not generate a new object after the function is run.

        The example is as follows:

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

        The result is as follows:

        ```text
        output1:[1, 2, 3, 4, "a"]
        output2:[1, 2, 3, Tensor(shape=[1], dtype=Int64, value= [4]), Tensor(shape=[1], dtype=Int64, value= [5])]
        ```

    - List.pop

        Basic syntax: ```pop_element = list_object.pop(index=-1)```.

        Basic semantics: Remove the `index` element of the `List` object `list_object` from `list_object` and return that element.

        `index` is required to be a constant `int`, and when `list_object` is of length `list_obj_size`, `index` is taken in the range `[-list_obj_size, list_obj_size-1]`. A negative number for `index` represents the number of bits from back to front. When there is no `index` input, the default value is -1, which means the last element is removed.

        With `JIT_SYNTAX_LEVEL` set to `COMPATIBLE` as well as `LAX`, `List.pop` supports the inplace operation, which does not generate a new object after the function is run.

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

        The result is as follows:

        ```text
        pop_element:3
        res_list:[1, 2]
        ```

    - List.reverse

        Basic syntax: ```list_object.reverse()```.

        Basic semantics: Reverse the order of the elements of the `List` object `list_object`.

        With `JIT_SYNTAX_LEVEL` set to `COMPATIBLE` as well as `LAX`, `List.reverse` supports the inplace operation, which does not generate a new object after the function is run.

        The example is as follows:

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

        The result is as follows:

        ```text
        output1:[3, 2, 1]
        ```

    - List.insert

        Basic syntax: ```list_object.insert(index, target_obj)```.

        Basic semantics: Insert `target_obj` into the `index` position of `list_object`.

        The `index` requires to be the constant `int`. If `list_object` is of length `list_obj_size`, when `index < -list_obj_size`, insert into the first position of `List`. When `index >= -list_obj_size`, insert to the end of `List`. A negative `index` represents the number of digits from back to front.

        With `JIT_SYNTAX_LEVEL` set to `COMPATIBLE` as well as `LAX`, `List.insert` supports the inplace operation, and the function runs without generating a new object.

        The example is as follows:

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

        The result is as follows:

        ```text
        output1:[3, 2, 1]
        ```

#### Tuple

`Tuple` can be constructed on the network, that is, the syntax `y = (1, 2, 3)` is supported. The elements of the tuple `Tuple` cannot be modified, but indexed access to elements in the tuple `Tuple` is supported, and concatenated combinations of tuples are supported.

- Supported index values

  Support accessing elements in the tuple `Tuple` using square brackets plus subscripted indexes. The index value can be `int`, `slice`, `Tensor`, and multi-level index value. That is, the syntax `data = tuple_x[index0][index1]...` is supported.

  Restrictions on the index value `Tensor` are as follows:

    - `Tuple` stores `Cell`. Each `Cell` must be defined before a tuple is defined. The number of input parameters, input parameter type, and input parameter `shape` of each `Cell` must be the same. The number of outputs of each `Cell` must be the same. The output type must be the same as the output `shape`.

    - The index `Tensor` is a scalar `Tensor` whose `dtype` is `int32`. The value range is `[-tuple_len, tuple_len)`, and negative index is not supported in `Ascend` backend.

    - `CPU`, `GPU` and `Ascend` backend is supported.

  An example of the `int` and `slice` indexes is as follows:

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

  The result is as follows:

  ```text
  ret:[0.]
  ```

- Support connection combinations

  Similar to the string `String`, tuples support combining using `+` and `*` to get a new tuple `Tuple`, for example:

  ```python
  import mindspore as ms
  import numpy as np

  @ms.jit()
  def test_index():
      x = (1, 2, 3)
      y = (4, 5, 6)
      return x + y, x * 2

  out1, out2 = test_index()
  print('out1:{}'.format(out1))
  print('out2:{}'.format(out2))
  ```

  The result is as follows:

  ```text
  out1:(1, 2, 3, 4, 5, 6)
  out2:(1, 2, 3, 1, 2, 3)
  ```

#### Dictionary

`Dictionary` can be constructed on the network. Each key value `key:value` is separated by a colon `:`, and each key value pair is separated by a comma `,`. The entire dictionary contains the key-value pairs using curly braces `{}`. That is, the syntax `y = {"a": 1, "b": 2}` is supported.

The `key` is unique, and if there are multiple identical `keys` in the dictionary, the duplicate `keys` are finalized with the last one and the value `value` can be non-unique. The key `key` needs to be guaranteed to be immutable. Currently, the `key` can be `String`, `Number`, constant `Tensor`, or `Tuple` that contains these types. The `value` can be `Number`, `Tuple`, `Tensor`, `List` or `Dictionary`.

- Supported APIs

  `keys`: extracts all `key` values from `dict` to form `Tuple` and return it.

  `values`: extracts all `value` values from `dict` to form `Tuple` and return it.

  `items`: extracts `Tuple` composed of each pair of `value` values and `key` values in `dict` to form `List` and return it.

  `get`: `dict.get(key[, value])` returns the `value` value corresponding to the specified `key`, if the specified `key` does not exist, the default value `None` or the set default value `value` is returned .

  `clear`: removes all elements in `dict`.

  `has_key`: `dict.has_key(key)` determines whether the specified `key` exists in `dict`.

  `update`: `dict1.update(dict2)` updates the elements in `dict2` to `dict1`.

  `fromkeys`: `dict.fromkeys(seq([, value]))` is used to create a new `Dictionary`, using the elements in the sequence `seq` as the `key` of the `Dictionary`, and the `value` is initial value corresponding to all `key`.

  For example:

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

  The result is as follows:

  ```text
  x_keys:('a', 'b', 'c')
  x_values:(Tensor(shape=[3], dtype=Int64, value= [1, 2, 3]), Tensor(shape=[3], dtype=Int64, value= [4, 5, 6]), Tensor(shape=[3], dtype=Int64, value= [7, 8, 9]))
  x_items:[('a', Tensor(shape=[3], dtype=Int64, value= [1, 2, 3])), ('b', Tensor(shape=[3], dtype=Int64, value= [4, 5, 6])), ('c', Tensor(shape=[3], dtype=Int64, value= [7, 8, 9]))]
  value_a:[1 2 3]
  check_key:True
  new_x:{'a': Tensor(shape=[3], dtype=Int64, value= [0, 0, 0]), 'b': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6]), 'c': Tensor(shape=[3], dtype=Int64, value= [7, 8, 9])}
  new_dict:{'a': 123, 'b': 123, 'c': 123, 'd': 123}
  ```

- Supported index values and value assignment

  For example:

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

  The result is as follows:

  ```text
  out1:{'a': (2, 3, 4), 'b': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6]), 'c': Tensor(shape=[3], dtype=Int64, value= [7, 8, 9])}
  out2:[4 5 6]
  ```

- Supported calculation graph return `Dictionary`

  For example:

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

  The result is as follows:

  ```text
  out:{'y': 'a'}
  ```

### MindSpore User-defined Data Types

Currently, MindSpore supports the following user-defined data types: `Tensor`, `Primitive`, and `Cell`.

#### Tensor

Currently, tensors can be constructed in the network.

For details of `Tensor`, click [Tensor API document](https://mindspore.cn/docs/en/r2.1/api_python/mindspore/mindspore.Tensor.html#mindspore-tensor).

#### Primitive

Currently, `Primitive` and its subclass instances can be constructed in construct.

However, during call, the parameter can be specified only in position parameter mode, and cannot be specified in the key-value pair mode.

For example:

```python
import mindspore as ms
from mindspore import nn, ops, Tensor, set_context
import numpy as np

set_context(mode=ms.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        reduce_sum = ops.ReduceSum(True) #`Primitive` and its subclass instances can be constructed in construct.
        ret = reduce_sum(x, axis=2)
        return ret

x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
net = Net()
ret = net(x)
print('ret.shape:{}'.format(ret.shape))
```

In the network defined above, the parameters of reduce_sum(x, axis=2) cannot be specified in the key-value pair mode. The parameter can be specified only in position parameter mode, that is, reduce_sum(x, 2).

The error is reported as follows:

```text
TypeError: Only supported positional parameter type for python primitive, but got keyword parameter type.
```

Currently, the attributes and APIs related to `Primitive` and its subclasses cannot be called on the network.

For details about the defined `Primitive`, click [Primitive API document](https://www.mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.Primitive.html#mindspore.ops.Primitive).

#### Cell

Currently, `Cell` and its subclass instances can be constructed on the network. That is, the syntax `cell = Cell(args...)` is supported.

However, during call, the parameter can be specified only in position parameter mode, and cannot be specified in the key-value pair mode. That is, the syntax `cell = Cell(arg_name=value)` is not supported.

Currently, the attributes and APIs related to `Cell` and its subclasses cannot be called on the network unless they are called through `self` in `construct` of `Cell`.

For details about the definition of `Cell`, click [Cell API document](https://www.mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.Cell.html).

#### Parameter

`Parameter` is a variable tensor, indicating the parameters that need to be updated during network training.

For details about the definition of `Parameter`, click [Parameter API document](https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore/mindspore.Parameter.html#mindspore.Parameter).

## Primaries

Primaries represent the most tightly bound operations of the language.

### Attribute References and Attribute Modification

An attribute reference is a primary followed by a period and a name. The following two cases of attribute references are allowed to be modified:

- The modified attribute belongs to this `cell` object, i.e. it must be `self.xxx`. The attribute is initialized in the Cell's `__init__` function. For example:

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
          self.weight = x  # modified if conditions are met
          self.m = 3  # modified if conditions are met
          # self.a = 2 Attribute a is not initialized within __init__ and cannot be modified.
          return x

  net = Net()
  ret = net(1, 2)
  print('net.weight:{}'.format(net.weight))
  print('net.m:{}'.format(net.m))
  ```

  The result is as follows:

  ```text
  net.weight:Parameter (name=w, shape=(1,), dtype=Int64, requires_grad=True)
  net.x:3
  ```

- The object whose attributes are modified is a global object, as exemplified by the following:

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

  The result is as follows:

  ```text
  data_obj.x:10
  ```

### Index Value

Index value of  a sequence `Tuple`, `List`, `Dictionary`, `Tensor` which called subscription in Python.

Index value of `Tuple` refers to chapter [Tuple](#tuple) of this page.

Index value of `List` refers to chapter [List](#list) of this page.

Index value of `Dictionary` refers to chapter [Dictionary](#dictionary) of this page.

Index value of `Tensor` refers to [Tensor index value document](https://www.mindspore.cn/docs/en/r2.1/note/index_support.html#index-values).

### Calls

A call calls a callable object (e.g., `Cell` or `Primitive`) with a possibly empty series of arguments.

For example:

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

### Unary Arithmetic Operators

| Unary Arithmetic Operator | Supported Type                               |
| :------------------------ | :------------------------------------------- |
| `+`                       | `Number`, `Tensor`, taking positive values.                           |
| `-`                       | `Number`, `Tensor`, `COOTensor`, `CSRTensor`, taking negative values. |
| `~`                       | `Tensor` with `Bool` data type, members take negation one by one.               |

notes:

- In native python the `~` operator get the bitwise inversion of its integer argument; in Mindspore the `~` redefined to get logic not for `Tensor(Bool)`.

### Binary Arithmetic Operators

| Binary Arithmetic Operator | Supported Type|
| :----------- |:--------|
| `+` |`Number` + `Number`, `String` + `String`, `Number` + `Tensor`, `Tensor` + `Number`, `Tuple` + `Tensor`, `Tensor` + `Tuple`, `List` + `Tensor`, `Tensor`+`List`, `List`+`List`, `Tensor` + `Tensor`, `Tuple` + `Tuple`, `COOTensor` + `Tensor`, `Tensor` + `COOTensor`, `COOTensor` + `COOTensor`, `CSRTensor` + `CSRTensor`.|
| `-` |`Number` - `Number`, `Tensor` - `Tensor`, `Number` -`Tensor`, `Tensor` - `Number`, `Tuple` -`Tensor`, `Tensor` -`Tuple`, `List` -`Tensor`, `Tensor` -`List`, `COOTensor` - `Tensor`, `Tensor` - `COOTensor`, `COOTensor` - `COOTensor`, `CSRTensor` - `CSRTensor`.|
| `*` |`Number` \* `Number`, `Tensor` \* `Tensor`, `Number` \* `Tensor`, `Tensor` \* `Number`, `List` \* `Number`, `Number` \* `List`, `Tuple` \* `Number`, `Number` \* `Tuple`, `Tuple` \* `Tensor`, `Tensor` \* `Tuple`,  `List` \*`Tensor`, `Tensor` \* `List`, `COOTensor` \* `Tensor`, `Tensor` \* `COOTensor`, `CSRTensor` \* `Tensor`, `Tensor` \* `CSRTensor`.|
| `/` |`Number` / `Number`, `Tensor` / `Tensor`, `Number` / `Tensor`, `Tensor` / `Number`, `Tuple` / `Tensor`, `Tensor` / `Tuple`,  `List` / `Tensor`, `Tensor` / `List`, `COOTensor` / `Tensor`, `CSRTensor` / `Tensor`.|
| `%` |`Number` % `Number`, `Tensor` % `Tensor`, `Number` % `Tensor`, `Tensor` % `Number`, `Tuple` % `Tensor`, `Tensor` % `Tuple`, `List` % `Tensor`, `Tensor` % `List`.|
| `**` |`Number` \*\* `Number`, `Tensor` \*\* `Tensor`, `Number` \*\* `Tensor`, `Tensor` \*\* `Number`, `Tuple` \*\* `Tensor`, `Tensor` \*\* `Tuple`,  `List` \*\* `Tensor`, `Tensor` \*\* `List`.|
| `//` |`Number` // `Number`, `Tensor` // `Tensor`, `Number` // `Tensor`, `Tensor` // `Number`, `Tuple` // `Tensor`, `Tensor` // `Tuple`,  `List` // `Tensor`, `Tensor` // `List`.|
| `&`     | `Number` & `Number`、`Tensor` & `Tensor`、`Number` & `Tensor`、`Tensor` & `Number`.                                                                                                                                                                  |
| `∣`      | `Number` &#124; `Number`、`Tensor` &#124; `Tensor`、`Number` &#124; `Tensor`、`Tensor` &#124; `Number`.                                                                                                                                                             |
| `^`     | `Number` ^ `Number`、`Tensor` ^ `Tensor`、`Number` ^ `Tensor`、`Tensor` ^ `Number`.                                                                                                                                                                  |
| `<<`    | `Number` << `Number`.                                                                                                                                                                                                                             |
| `>>`    | `Number` >> `Number`.                                                                                                                                                                                                                             |

Restrictions:

- If all operands are `Number` type, operations between  `Float64` and `Int32` are not supported. Operators including `+`, `-`, `*`, `/`, `%`, `**`, `//` all support left and right operands to be `Bool` value.
- If either operand is `Tensor` type, left and right operands can't both be `Bool` value.
- The `*` operation on `List/Tuple` and `Number` means that `List/Tuple` is copied from `Number` and then concatenated. The data type inside `List` can be any data type supported by the graph mode, and multi-layer nesting is also supported. The data type in `Tuple` must be `Number`, `String`, `None`, and multi-layer nesting is also supported.

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

Constraints:

- If all operands of  `AugAssign` are `Number` type, value of Number can't be `Bool`.

- If all operands of  `AugAssign` are `Number` type, operations between  `Float64` and `Int32` are not supported.

- If either operand of  `AugAssign` is `Tensor` type, left and right operands can't both be `Bool` value.

- The `*=` operation on `List/Tuple` and `Number` means that `List/Tuple` is copied from `Number` and then concatenated, and the elements of the object within `List/Tuple` can contain any of the types supported by the intentional pattern, and multiple levels of nesting are also supported.

### Logical Operators

| Logical Operator | Supported Type|
| :----------- |:--------|
| `and` |`String`,  `Number`,  `Tuple`, `List` , `Dict`, `None`, `Scalar`, `Tensor`.|
| `or` |`String`,  `Number`,  `Tuple`, `List` , `Dict`, `None`, `Scalar`, `Tensor`.|
| `not` |`Number`, `tuple`, `List` and `Tensor`  with only one element.|

Restrictions:

- The left operand of operator `and`, `or` must be able to be converted to boolean value. For example, left operand can not be Tensor with multiple elements. If the left operand of `and`, `or` is variable `Tensor`, the right operand must also be single-element `Tensor` with the same type. Otherwise, there is no requirement for right operand.

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

## Simple Statements

### raise Statements

Support the use of `raise` to trigger an exception. `raise` syntax format: `raise[Exception [, args]]`. The `Exception` in the statement is the type of the exception, and the `args` is the user-supplied argument to the exception, usually a string or other object. The following types of errors are supported: NoExceptionType, UnknownError, ArgumentError, NotSupportError, NotExistsError, DeviceProcessError, AbortedError, IndexError, ValueError, TypeError, KeyError, AttributeError, NameError, AssertionError, BaseException, KeyboardInterrupt, Exception, StopIteration, OverflowError, ZeroDivisionError, EnvironmentError, IOError, OSError, ImportError, MemoryError, UnboundLocalError, RuntimeError, NotImplementedError, IndentationError, RuntimeWarning.

For example:

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

The output result:

```text
ValueError: x should be greater than y.
```

### assert Statements

Supports the use of assert for exception checking, `assert` syntax format: `assert[Expression [, args]]`, where `Expression` is the judgment condition. If the condition is true, nothing will be done, while if the condition is false, an exception message of type `AssertError` will be thrown. The `args` are user-supplied exception arguments, which can usually be strings or other objects.

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

Appears normally in the output:

```text
AssertionError.
```

### pass Statements

The `pass` statement doesn't do anything and is usually used as a placeholder to maintain structural integrity. For example:

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

 The result is as follows:

```text
ret: 50.625
```

### return Statements

The `return` statement usually returns the result to the place where it was called, and statements after the `return` statement are not executed. If the return statement does not have any expression or the function does not have a `return` statement, a `None` object is returned by default. There can be more than one `return` statement within a function, depending on the situation. For example:

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

As above, there can be multiple `return` statements in a control flow scenario statement. If there is no `return` statement in a function, the None object is returned by default, as in the following use case:

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

### break Statements

The `break` statement is used to terminate a loop statement, i.e., it stops execution of the loop statement even if the loop condition does not have a `False` condition or if the sequence is not fully recursive, usually used in `while` and `for` loops. In nested loops, the `break` statement stops execution of the innermost loop.

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

The result is as follows:

```text
ret: 1920
```

### continue Statements

The `continue` statement is used to jump out of the current loop statement and into the next round of the loop. This is different from the `break` statement, which is used to terminate the entire loop statement. `continue` is also used in `while` and `for` loops. For example:

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

The result is as follows:

```text
ret: 9
```

## Compound Statements

### Conditional Control Statements

#### if Statements

Usage:

- `if (cond): statements...`

- `x = y if (cond) else z`

Parameter: `cond` -- Variables of `Bool` type and constants of `Bool`, `List`, `Tuple`, `Dict` and `String` types are supported.

Restrictions:

- If `cond` is not a constant, the variable or constant assigned to a same sign in different branches should have same data type. If the data type of assigned variables or constants is `Tensor`, the variables and constants should have same shape and element type. For shape consistency restrictions, please refer to [ShapeJoin Rules](https://www.mindspore.cn/tutorials/experts/en/r2.1/network/control_flow.html#shapejoin-rules).

Example 1:

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

The data type of `m` returned by the `if` branch and `n` returned by the `else` branch must be same.

The result is as follows:

```text
ret:1
```

Example 2:

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

The variable or constant `m` assigned to `out` in `if` branch and the variable or constant `n` assigned to out in `false` branch must have same data type.

The result is as follows:

```text
ret:1
```

Example 3:

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

The variable or constant `m` assigned to `out` in `if` branch and the variable or constant `init` initially assigned to `out` must have same data type.

The result is as follows:

```text
ret:1
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

- If `cond` is not a constant, the variable or constant assigned to a same sign inside body of `while` and outside body of `while` should have same data type.If the data type of assigned variables or constants is `Tensor`, the variables and constants should have same shape and element type. For shape consistency restrictions, please refer to [ShapeJoin Rules](https://www.mindspore.cn/tutorials/experts/en/r2.1/network/control_flow.html#shapejoin-rules).

- The `while...else...` statement is not supported.

Example 1:

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

The data type of `m` returned inside `while` and data type of `n` returned outside `while` must have same data type.

The result is as follows:

```text
ret:1
```

Example 2:

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

The variable `op1` assigned to `out` inside `while` and the variable or constant `init` initially assigned to `out` must have same data type.

The result is as follows:

```text
ret:15
```

### Function Definition Statements

#### def Keyword

`def` is used to define a function, followed by the function identifier name and the original parentheses `()`, which may contain the function parameters.
Usage: `def function_name(args): statements...`.

For example:

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

The result is as follows:

```text
ret:6
```

Instructions:

- The defined function supported has no `return` statement. That means the return value of default functions is None.
- `Construct` function of the outermost network and the inner network function is support kwargs, like:`def construct(**kwargs):`.
- Mixed use of variable argument and non-variable argument is supported, like:`def function(x, y, *args)` and `def function(x = 1, y = 1, **kwargs)`.

#### lambda Expression

A `lambda` expression is used to generate an anonymous function. Unlike normal functions, it computes and returns only one expression. Usage: `lambda x, y: x + y`.

For example:

```python
import mindspore as ms

@ms.jit()
def test(x, y):
    number_add = lambda x, y: x + y
    return number_add(x, y)

ret = test(1, 5)
print('ret:{}'.format(ret))
```

The result is as follows:

```text
ret:6
```

#### Partial function partial

Function: partial function, fixed function input parameter. Usage: `partial(func, arg, ...)`.

Input parameter:

- `func` -- function.

- `arg` -- One or more parameters to be fixed, support positional parameters and key-value pair parameters.

Return Value: Returns some functions with fixed input value.

The example is as follows:

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

The result is as follows:

```text
m:5
n:7
```

#### Function Parameters

- Default parameter value: The default value set to `Tensor` type data is currently not supported, and `int`, `float`, `bool`, `None`, `str`, `tuple`, `list`, `dict` type data is supported.
- Variable parameters: Inference and training of networks with variable parameters are supported.
- Key-value pair parameter: Functions with key-value pair parameters cannot be used for backward propagation.
- Variable key-value pair parameter: Functions with variable key-value pairs cannot be used for backward propagation.

### List Comprehension and Generator Expression

Support for List Comprehension and Generator Expression. Support for constructing a new sequence. List Comprehension is used to generate a new list `List` and Generator Expression is used to generate a new tuple `Tuple`.

#### List Comprehension

List comprehension are used to generate lists. Usage: `[arg for loop if statements]`.

The example is as follows:

```python
import mindspore as ms

@ms.jit()
def test(x, y):
    l = [x * x for x in range(1, 11) if x % 2 == 0]
    return l

ret = test(1, 5)
print('ret:{}'.format(ret))
```

The result is as follows:

```text
ret:[4, 16, 36, 64, 100]
```

Restrictions:

The use of multiple levels of nested iterators is not supported in graph mode.

The example usage of the restriction is as follows (two levels of iterators are used):

```python
l = [y for x in ((1, 2), (3, 4), (5, 6)) for y in x]
```

An error will be prompted:

```text
TypeError:  The `generators` supports one `comprehension` in ListComp/GeneratorExp, but got 2 comprehensions.
```

#### Generator Expression

Generator expressions are used to generate lists. Usage: `(arg for loop if statements)`.

For example:

```python
import mindspore as ms

@ms.jit()
def test(x, y):
    l = (x * x for x in range(1, 11) if x % 2 == 0)
    return l

ret = test(1, 5)
print('ret:{}'.format(ret))
```

The result is as follows:

```text
ret:[4, 16, 36, 64, 100]
```

Usage restrictions are the same as list comprehension, i.e., the use of multiple levels of nested iterators is not supported in graph mode.

### With Statement

In graph mode, the `with` statement is supported with limitations. The `with` statement requires that the object must have two magic methods: `__enter__()` and `__exit__()`.

For example:

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

The result is as follows:

```text
out1: [5]
out2: [2]
```

## Python Built-in Functions

Currently supported Python built-in functions include `int`, `float`, `bool`, `str`, `list`, `tuple`, `getattr`, `hasattr`, `len`, `isinstance`, `all`, `any`, `round`, `max`, `min` , `sum`, `abs`, `partial`, `map`, `range`, `enumerate`, `super`, `pow`, `filter`. The use of built-in functions in graph mode is similar to the corresponding Python built-in functions.

### int

Function: Return the integer value based on the input number or string.

Call: `int(x=0, base=10)`, converted to decimal by default.

Input parameter:

- `x` -- the object need to be converted to integer, the valid type of x includes `int`, `float`, `bool`, `str`, `Tensor` and third-party object (such as `numpy.ndarray`).

- `base` -- the base to convert. `base` is only allowed when `x` is constant `str`.

Return value: the converted integer.

For example:

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

The result is as follows:

```text
a: 3
b: 3
c: 18
d: 10
e: 8
```

### float

Function: Return the floating-point number based on the input number or string.

Calling: `float(x=0)`.

Input parameter: `x` -- the object need to be converted to floating number, the valid type of x includes `int`, `float`, `bool`, `str`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: the converted floating-point number.

For example:

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

x = ms.Tensor([-1.0], ms.float32)
a, b, c, d = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
```

The result is as follows:

```text
a: 1.0
b: 112.0
c: -123.6
d: 123.0
e: -1.0
```

### bool

Function: Return the boolean value based on the input.

Calling: `bool(x=false)`

Input parameter: `x` -- the object need to be converted to boolean value, the valid type of x includes `int`, `float`, `bool`, `str`, `list`, `tuple`, `dict`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: the converted boolean scalar.

For example:

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

The result is as follows:

```text
a: False
b: False
c: True
d: True
e: True
```

### str

Function: Return the string value based on the input.

Calling: `str(x='')`

Input parameter: `x` -- the object need to be converted to string value, the valid type of x includes `int`, `float`, `bool`, `str`, `list`, `tuple`, `dict`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: string converted from `x`.

For example:

```python
import numpy as np
import mindspore as ms

@ms.jit
def func(x):
   a = str()
   b = str(0)
   c = str([1, 2, 3, 4])
   d = str(ms.Tensor([10]))
   e = str(np.array([1, 2, 3, 4]))
   f = str(x.asnumpy())
   g = str(2 * x)
   return a, b, c, d, e, f, g

x = ms.Tensor([-1.0], ms.float32)
a, b, c, d, e = func(x)
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
print("e: ", e)
print("f: ", f)
```

The result is as follows:

```text
a:                                             # a is empty string
b: 0
c: [1, 2, 3, 4]
d: Tensor(shape=[1], dtype=Int64, value=[10])
e: [1 2 3 4]
f: [-1.0]
g: [-2.0]
```

### tuple

Function: Return a tuple based on the input object.

Calling: `tuple(x=())`.

Input parameter: `x` -- the object that need to be converted to tuple, the valid type of x includes `list`, `tuple`, `dict`, `Tensor` or third-party object (such as `numpy.ndarray`).

Return value: tuple with elements of `x`, `x` is cut based on zero dimension.

For example:

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

The result is as follows:

```text
a: (1, 2, 3)
b: (1, 2, 3)
c: ('a', 'b', 'c')
d: (Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 2), Tensor(shape=[], dtype=Int64, value= 3))
```

### list

Function: Return a list based on the input object.

Calling: `list(x=())`.

Input parameter: `x` -- the object that need to be converted to list, the valid type of x includes `list`, `tuple`, `dict`, `Tensor` or third-party object (such as `numpy.ndarray`).

Return value: list with elements of `x`, `x` is cut based on zero dimension.

For example:

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

The result is as follows:

```text
a_t: [1, 2, 3]
b_t: [1, 2, 3]
c_t: ['a', 'b', 'c']
d_t: [Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 2), Tensor(shape=[], dtype=Int64, value= 3)]
```

### getattr

Function: Get the attribute of python object.

Calling: `getattr(x, attr, default)`.

Input parameter:

- `x` -- The object to get attribute, `x` can be all types that graph mode supports. `x` can not be third-party object.

- `attr` -- The name of the attribute, the type of `attr` should be `str`.

- `default` -- Optional input. If `x` do not have `attr`, `default` will be returned. `default` can be all types that graph mode supports but can not be third-party object. If `default` is not set and `x` does not have attribute `attr`, AttributeError will be raised.

Return value: Target attribute or `default`.

For example:

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

The result is as follows:

```text
a: 0
b: 2
```

The attribute of object in graph mode may be different from that in pynative mode. It is suggested to use `default` input or call `hasattr` before using `getattr` to avoid AttributeError.

### hasattr

Function: Judge whether an object has an attribute.

Calling: `hasattr(x, attr)`.

Input parameter:

- `x` -- The object to get attribute, `x` can be all types that graph mode supports and also can be third-party object.

- `attr` -- The name of the attribute, the type of `attr` should be `str`.

Return value: boolean value indicates whether `x` has `attr`.

For example:

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

The result is as follows:

```text
a: True
b: False
```

### len

Function: Return the length of an object (string or other iterable object).

Calling: `len(sequence)`.

Input parameter: `sequence` -- `Tuple`, `List`, `Dictionary`, `Tensor` or third-party object (such as numpy.ndarray).

Return value: length of the sequence, which is of the `int` type. If the input parameter is `Tensor`, the length of dimension 0 is returned.

For example:

```python
import mindspore as ms
import numpy as np

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

input_x = Tensor([1, 2, 3, 4])
x_len, y_len, d_len, z_len, n_len, w_len = test(input_x)
print('x_len:{}'.format(x_len))
print('y_len:{}'.format(y_len))
print('d_len:{}'.format(d_len))
print('z_len:{}'.format(z_len))
print('n_len:{}'.format(n_len))
print('w_len:{}'.format(w_len))
```

The result is as follows:

```text
x_len:3
y_len:3
d_len:2
z_len:6
z_len:4
w_len:1
```

### isinstance

Function: Determines whether an object is an instance of a class.

Calling: `isinstance(obj, type)`.

Input parameters:

- `obj` -- Any instance of any supported type.

- `type` -- `bool`, `int`, `float`, `str`, `list`, `tuple`, `dict`, `Tensor`, `Parameter`, or the types of third-party libraries (e.g. numpy.ndarray) or a `tuple` containing only those types.

Return value: If `obj` is an instance of `type`, return `True`. Otherwise, return `False`.

For example:

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

w = Tensor(np.array([-1, 2, 4]))
x_is_tuple, y_is_list, z_is_tensor, w_is_ndarray = test(w)
print('x_is_tuple:{}'.format(x_is_tuple))
print('y_is_list:{}'.format(y_is_list))
print('z_is_tensor:{}'.format(z_is_tensor))
print('w_is_ndarray:{}'.format(w_is_ndarray))
```

The result is as follows:

```text
x_is_tuple:True
y_is_list:True
z_is_tensor:True
w_is_ndarray:True
```

### all

Function: Judge whether all of the elements in the input is true.

Calling: `all(x)`.

Input parameter: - `x` -- Iterable object, the valid types include `list`, `tuple`, `dict`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: boolean, return `True` if all elements are `True`, otherwise `False`.

For example:

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
i: False
```

### any

Function: Judge whether any of the elements in the input is true.

Calling: `any(x)`.

Input parameter: - `x` -- Iterable object, the valid types include `list`, `tuple`, `dict`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: boolean, return `False` if all elements are `False`, otherwise `True`. Elements count as `True` except for 0, null, and `False`.

For example:

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
   i = all(x.asnumpy())
   return a, b, c, d, e, f, g, h, i

a, b, c, d, e, f, g, h = func()
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
i: True
```

### round

Function: Return the rounding value of input.

Calling: `round(x, digit=0)`

Input parameter:

- `x` -- the object to rounded, the valid types include `int`, `float`, `bool`, `Tensor` and third-party object that defines magic function `__round__()`.

- `digit` -- the number of decimal places to round, the default value is 0. `digit` can be `int` object or `None`. If `x` is `Tensor`, then `round()` does not support input `digit`.

Return value: the value after rounding.

For example:

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

### max

Function: Return the maximum of inputs.

Calling: `max(*data)`.

Input parameter: - `*data` -- If `*data` is single input, `max` will compare all elements within `data` and `data` must be iterable object. If there are multiple inputs, then `max()` will compare each of them. The valid types of `data` include `int`, `float`, `bool`, `list`, `tuple`, `dict`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: boolean, the maximum of the inputs.

For example:

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

### min

Function: Return the minimum of inputs.

Calling: `min(*data)`.

Input parameter: - `*data` -- If `*data` is single input, then `min()` will compare all elements within `data` and `data` must be iterable object. If there are multiple inputs, then `min()` will compare each of them. The valid types of `data` include `int`, `float`, `bool`, `list`, `tuple`, `dict`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: boolean, the minimum of the inputs.

For example:

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

### sum

Function: Return the sum of input sequence.

Calling: `sum(x, n=0)`.

Input parameter:

- `x` -- iterable with numbers, the valid types include `list`, `tuple`, `Tensor` and third-party object (such as `numpy.ndarray`).

- `n` -- the number that will be added to the sum of `x`, which is assumed to be 0 if not given.

Return value: the value obtained by summing `x` and adding it to `n`.

For example:

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

### abs

Function: Return the absolute value of the input.

Calling: `abs(x)`.

Input parameter: - `x` -- The valid types of `x` include `int`, `float`, `bool`, `complex`, `Tensor` and third-party object (such as `numpy.ndarray`).

Return value: the absolute value of the input.

For example:

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

The result is as follows:

```text
a: 45
b: 100.12
```

### map

Function: Maps one or more sequences based on the provided functions and generates a new sequence based on the mapping result. The current requirement is that the number of elements in multiple sequences be the same.

Calling: `map(func, sequence, ...)`.

Input parameters:

- `func` -- Function.

- `sequence` -- One or more sequences (`Tuple` or `List`).

Return value: Return a new sequence.

For example:

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

ret1，ret2 = test()
print('ret1:{}'.format(ret1))
print('ret2:{}'.format(ret2))
```

The result is as follows:

```text
ret1: (5, 7, 9)
ret2: [6, 8, 10]
```

### zip

Function: Packs elements in the corresponding positions in multiple sequences into tuples, and then uses these tuples to form a new sequence. If the number of elements in each sequence is inconsistent, the length of the new sequence is the same as that of the shortest sequence.

Calling: `zip(sequence, ...)`.

Input parameter: `sequence` -- One or more sequences (`Tuple` or `List`)`.

Return value: Return a new sequence.

For example:

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

The result is as follows:

```text
ret:((1, 4), (2, 5), (3, 6))
```

### range

Function: Creates a `Tuple` based on the start value, end value, and step.

Calling:

- `range(start, stop, step)`

- `range(start, stop)`

- `range(stop)`

Input parameters:

- `start` -- start value of the count. The type is `int`. The default value is 0.

- `stop` -- end value of the count (exclusive). The type is `int`.

- `step` -- Step. The type is `int`. The default value is 1.

Return value: Return a `Tuple`.

For example:

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

The result is as follows:

```text
x:(0, 2, 4)
y:(0, 1, 2, 3, 4)
z:(0, 1, 2)
```

### enumerate

Function: Generates an index sequence of a sequence. The index sequence contains data and the corresponding subscript.

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

The result is as follows:

```text
m:((3, 100), (4, 200), (5, 300), (6, 400))
n:((0, Tensor(shape=[2], dtype=Int64, value= [1, 2])), (1, Tensor(shape=[2], dtype=Int64, value= [3, 4])), (2, Tensor(shape=[2], dtype=Int64, value= [5, 6])))
```

### super

Function: Calls a method of the parent class (super class). Generally, the method of the parent class is called after `super`.

Calling:

- `super().xxx()`

- `super(type, self).xxx()`

Input parameters:

- `type` -- Class.

- `self` -- Object.

Return value: method of the parent class.

For example:

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

The result is as follows:

```text
out: (9, 6)
```

### pow

Function: Return the power.

Calling: `pow(x, y)`

Input parameters:

- `x` -- Base number, `Number`, or `Tensor`.

- `y` -- Power exponent, `Number`, or `Tensor`.

Return value: `y` power of `x`, `Number`, or `Tensor`

For example:

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

The result is as follows:

```text
ret:[ 1  4 27]
```

### print

Function: Prints logs.

Calling: `print(arg, ...)`.

Input parameter: `arg` -- Information to be printed (`int`, `float`, `bool`, `String` or `Tensor`, or third-party library data types).

Return value: none

Note: JIT Fallback supports printing constants in static graph mode using Python native print. See the [Printing with Python's native print](#using-native-print-printing-of-python) section of this article for more visible details.

For example:

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

The result is as follows:

```text
Tensor(shape=[3], dtype=Int32, value= [1 2 3])
Tensor(shape=[], dtype=Int32, value=3)
```

### filter

Function: According to the provided function to judge the elements of a sequence. Each element is passed into the function as a parameter in turn, and the elements whose return result is not 0 or False form a new sequence.

Calling: `filter(func, sequence)`

Input parameters:

- `func` -- Function.

- `sequence` -- A sequence (`Tuple` or `List`).

Return value: Return a new sequence.

For example:

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

The result is as follows:

```text
ret1:(1, 3, 5)
ret2:[7, 9]
```

## Network Definition

### Network Input parameters

The input parameters of the outermost network only can be `lool`, `int`, `float`, `Tensor`, `None`, `mstype.number(mstype.bool, mstype.int, mstype.float, mstype.uint)`, `List` or `Tuple` that contains these types, and `Dictionary` whose values are these types.

While calculating gradient for outermost network, only `Tensor` input could be calculated, input of other type will be ignored. For example, input parameter `(x, y,  z)` of outermost network, `x` and `z` are `Tensor` type, `y` is other type. While calculating gradient for the network, only gradients of `x` and `z` are calculated, and `(grad_x, grad_y)` is returned.

If you want to use other types of input for the network, please transfer them to the network while initializing network, and save them as network attributes, then use  in the `construct`. The input parameters of inner network do not have this restriction.

For example:

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

- Common Python function with the [@jit](https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore/mindspore.jit.html) decorator.

- Cell subclass inherited from [nn.Cell](https://www.mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.Cell.html).

### Network Construction Components

| Category                             | Content                                                      |
| :----------------------------------- | :----------------------------------------------------------- |
| `Cell` instance                      | [mindspore/nn/*](https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore.nn.html) and user-defined [Cell](https://www.mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.Cell.html). |
| Member function of a `Cell` instance | Member functions of other classes in the construct function of Cell can be called. |
| `jit_class` instance                 | Class decorated with [@jit_class](https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore/mindspore.jit_class.html). |
| `Composite` operator                 | [mindspore/ops/operations/*](https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore.ops.primitive.html) |
| `Composite` operator                 | [mindspore/ops/composite/*](https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore.ops.primitive.html) |
| `constexpr` generation operator      | Value computation operator generated by [@constexpr](https://www.mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.constexpr.html). |
| Function                             | User-defined Python functions and system functions listed in the preceding content. |

### Network Constraints

1. When an undefined class member is used in the `construct` function, `AttributeError` exception will be thrown.

   For example:

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

   The result would be:

   ```text
   AttributeError: External object has no attribute y
   ```

2. Class methods modified by `classmethod` in `nn.Cell` are not supported.

### JIT Fallback

In MindSpore static diagram mode, users need to follow MindSpore [static diagram syntax support](https://www.mindspore.cn/docs/en/r2.1/note/static_graph_syntax_support.html) when writing programs. Constraints exist on the use of the syntax.In dynamic graph mode, Python script code is executed according to the Python syntax, and users can use any Python syntax. It can be seen that the syntax constraint restrictions are different for static and dynamic graphs.

JIT Fallback considers the unification of static and dynamic graphs from the perspective of static graphs. Through the JIT Fallback feature, static graphs can support as many dynamic diagram syntaxes as possible, making static graphs provide a syntax experience close to that of dynamic graphs, thus achieving dynamic unity. To facilitate the user's ability to choose to use the JIT Fallback feature, the JIT syntax support level option 'jit_syntax_level' is provided. The value must be in [STRICT(0), COMPATIBLE(1), LAX(2)]. Default: LAX(2). All levels support all backends.
STRICT(0): Only basic syntax is supported, and execution performance is optimal.
COMPATIBLE(1): Besides basic syntax, supports more syntax, such as operations of dict, list, and scalar.
LAX(2): Compatible with all Python syntax as much as possible. However, execution performance may be affected and not optimal.

This document describes the support scope and usage notes of JIT Fallback so that you can use JIT Fallback features more effectively.

#### Support Scope

The JIT Fallback feature is still being improved, and the following is a list of static graph compilation syntaxes that are currently supported by this feature.

#### Creating and Using Tensor

JIT Fallback supports creating and using [Tensor](https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore/mindspore.Tensor.html) in static graph mode.

The code case is as follows, and `Tensor(1, dtype=mstype.int32)` is supported by JIT Fallback.

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

Output the result:

```text
1
```

The above example uses the interface of Tensor class to create a Tensor.
In some cases, it may be necessary to create a Tensor at runtime.
In this case, you can use either the aforementioned ms.Tensor interface or the [tensor function interface](https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore/mindspore.tensor.html#mindspore.tensor)to create a Tensor.
The code example is shown below.

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

Output the result:

```text
1.0
```

#### Annotation

For JIT fallback support at runtime, nodes are generated that cannot be derived by type and are called Any types. Since the correct type cannot be inferred at compile time, Any types will be operated with a default maximum precision of Float64 to prevent loss of precision. To optimize performance, it is recommended to minimize the generation of Any types. When the user knows exactly what type of statement will be generated through JIT fallback support, it is recommended to use `Annotation @jit.typing:` to specify the corresponding Python statement type, thereby determining the type of the interpretation node and avoiding the generation of Any types.

For example, the difference between the Tensor class and the tensor interface in the above example is that the annotation mechanism is used within the tensor interface. When the dtype of the tensor function is determined, the function will use annotations to specify the output type and avoid the generation of Any types. To use annotations, simply add a comment above or below the corresponding Python statement, such as # @jit.typing: () -> tensor_type[float32], where -> tensor_type[float32] indicates the output type of the annotated statement.

The code example is as follows.

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

Output the result:

```text
y1 value is 2.0, dtype is Float32
y2 value is 2.0, dtype is Float32
y3 value is 2.0, dtype is Float64
y4 value is 2.0, dtype is Float64
```

"The above examples show the differences in creating Tensors using JIT Fallback Runtime. Due to the lack of Annotation indication in the Tensor class, y3 and y4 cannot infer the correct type and can only perform operations in the highest precision Float64. For y2, the corresponding type for JIT Fallback was specified through Annotation during Tensor creation, allowing it to perform operations according to the specified type. y1 created the Tensor using the tensor function interface and passed the dtype parameter as an Annotation indication, avoiding the generation of Any type."

#### Calling the Third-party Libraries

JIT Fallback supports calling objects and methods of third-party libraries in the static graph mode.

It should be noted that for methods with return values, you need to use variables to save their results, otherwise an error may be reported. This usage will be supported in subsequent versions.

An code example to call a third-party library is shown below. The use case calls the NumPy third-party library, where `np.array([1, 2, 3])` and `np.array([4, 5, 6])` are supported via JIT Fallback.

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

Output the result:

```text
[5 7 9]
```

#### Using Native Print Printing of Python

JIT Fallback supports printing constants in static graph mode by using native print of Python, which is different from [Print operator](https://www.mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.Print.html) prints information at a different time. Python native print is triggered during compilation (at compiling time phase printing), while the Print operator requires the graph to be compiled and sent down to the device side to run before printing (at runtime phase printing).

For the sake of understanding, the following examples are given. tensor_sum involves Tensor summing, i.e. the runtime phase to get the result. When calling print, the actual call is the Print operator in the static graph mode. Refer to [static graph syntax support](https://www.mindspore.cn/docs/en/r2.1/note/static_graph_syntax_support.html). And np_num is the result of adding up two NumPy constants, i.e., the usage supported by JIT Fallback, so when calling print, the native Python print is used. Because of the different timing of the two prints, it ends up showing np_sum before tensor_sum, i.e. the print result of Python native print supported by JIT Fallback will be before the Print operator.

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

Output the result:

```text
np_sum: [2 4 6 8 10]
tensor_sum: (2, 4, 6, 8, 10)
```

#### Using the raise and assert

JIT Fallback supports the use of raise and assert in static graph mode.

Support the use of raise, the test case is as follows:

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

Output the result:

```text
ValueError: x should be greater than y.
```

Support the use of assert, the test case is as follows:

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

The output appears normally:

```text
AssertionError.
```

#### Calling Python Built-in Functions

MindSpore supports some Python built-in functions in static graph mode, including but not limited to len, isinstance, map, zip, etc. Please refer to [static graph syntax support](https://www.mindspore.cn/docs/en/r2.1/note/static_graph_syntax_support.html). With JIT Fallback, more uses of Python built-in functions can be supported in constant scenarios. Here is a brief example of some of the supported Python built-in functions.

##### dict()

Function: Used to create a dictionary.

Valid input: The Key of the dictionary supports only String type. The Value supports only constants, and does not support custom classes.

Looping over dictionaries created by `dict()` is not supported yet, including `dict.keys()`, `dict.values()` and `dict.items()`.

Examples of code usage are as follows:

```python
import mindspore as ms

@ms.jit
def func():
    a = dict()                                          # Create an empty dictionary
    b = dict(a='a', b='b', t='t')                       # Pass in keywords
    c = dict(zip(['one', 'two', 'three'], [1, 2, 3]))   # Mapping function approach to constructing dictionaries
    d = dict([('one', 1), ('two', 2), ('three', 3)])    # Iterable object approach to constructing dictionaries
    return a, b, c, d

a, b, c, d = func()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)
```

Output the result:

```text
a: {}
b: {'a': 'a', 'b': 'b', 't': 't'}
c: {'one': 1, 'two': 2, 'three': 3}
d: {'one': 1, 'two': 2, 'three': 3}
```

##### type()

Function: Output the type of the input parameter.

Valid inputs: number, list, tuples, dict, np.array, constant Tensor.

Examples of code usage are as follows:

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

Output the result:

```text
a: <class 'int'>
b: <class 'float'>
c: <class 'list'>
d: <class 'tuple'>
e: <class 'dict'>
f: <class 'numpy.ndarray'>
g: <class 'mindspore.common.tensor.Tensor'>
```

> There is another way to use type as a native Python function, i.e. type(name, bases, dict) returns a class object of type name, which is not supported currently because of the low usage scenario.

#### Supporting Control Flow

In order to improve Python standard syntax support and achieve dynamic unification, the use of control flow statements is achieved through JIT Fallback. Control flow statements are process control statements such as if, for, and while. The JIT Fallback feature supports creating and using Tensor in static graph mode, calling third-party libraries such as Numpy to create and use constants and variables, and supporting some of Python built-in functions. In theory, the syntax supported by JIT Fallback is also supported in control flow scenarios.

Examples of code usage are as follows:

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

Output the result:

```text
res: 2
```

#### Support JIT Fallback in the Runtime Phase

When JIT Fallback handles unsupported syntax expressions, it will generate corresponding nodes, and constants will derive values at compile time, otherwise these nodes will be passed to the backend runtime, where the result is obtained through capable execution of Python. The sample code is as follows. `np.add(x, y)` will generate the corresponding node, and the node, as the return value of the function, will be passed to the runtime. Currently, JIT Fallback for the runtime phase in some scenarios is supported.

```python
import numpy as np
import mindspore as ms

@ms.jit
def test_np_add():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    return np.add(x, y)

np_add_res = test_np_add()
print(np_add_res)
```

Output the result:

```text
[ 2  4  6  8  10]
```

#### The Top-level Graph Supports Returning Basic Types Such as list, dict, scalar, and none

##### The Top-level Graph Supports Returning lists

```python
import mindspore as ms

@ms.jit
def test_return_list():
    return [1, "a", True, None, ms.Tensor([2])]

res = test_return_list()
print(res)
```

Output the results:

```text
[1, 'a', True, None, Tensor(shape=[1], dtype=Int64, value= [2])]
```

##### The Top-level Graph Supports Returning dicts

```python
import mindspore as ms

@ms.jit
def test_return_dict():
    x = {'a': 1, 'b': 2}
    y = x.get('a')
    y_tensor = ms.Tensor([y])
    z = dict(a=y_tensor)
    return z

res = test_return_dict()
print(res)
```

Output the results:

```text
{'a': Tensor(shape=[1], dtype=Int64, value= [1])}
```

##### The Top-level Graph Supports Returning scalars

```python
import mindspore as ms

@ms.jit
def test_return_scalar(x, y):
    return x + y

res = test_return_scalar(ms.mutable(1), ms.mutable(2))
print(res)
```

Output the results:

```text
3
```

##### The Top-level Graph Supports Returning None

```python
import mindspore as ms

@ms.jit
def test_return_none():
    return 1, "a", None

res = test_return_none()
print(res)
```

Output the results:

```text
(1, 'a', None)
```

#### Instructions for Use

When using JIT Fallback, please note the following points:

1. The ability of JIT Fallback to support scalar dynamic graphs shall be within the scope of dynamic graph syntax, including but not limited to data types.

2. The current constant control flow scenario does not support the assignment of subscripts to Numpy Array data at this time, and the wrong code example is as follows:

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

   The error message is reported as follows:

   ```text
   RuntimeError: For operation 'setitem', current input arguments types are <External, Number, Number>. The 1-th argument type 'External' is not supported now.
   ```
