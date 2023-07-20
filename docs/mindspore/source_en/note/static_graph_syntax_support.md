# Static Graph Syntax Support

<a href="https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/static_graph_syntax_support.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png"></a>

## Overview

In graph mode, Python code is not executed by the Python interpreter. Instead, the code is compiled into a static computation graph, and then the static computation graph is executed.

There are two ways to use the graph mode. The first way is to call the `@jit` decorator to modify a function or a class member method, and then the decorated function or method will be compiled into a static computation graph. For details about how to use `jit`, click [jit API document](https://www.mindspore.cn/docs/en/r2.1/api_python/mindspore/mindspore.jit.html#mindspore.jit). The second way is to set `ms.set_context(mode=ms.GRAPH_MODE)`, then write the code in the `construct` function of the `Cell` so that the code in the `construct` function will be compiled into a static computation graph. For details about the definition of `Cell`, click [Cell API document](https://www.mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.Cell.html).

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

Supports returning numbers. For example:

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

In the above sample code, through the `append`, the inplace syntax, to change the `List` object, the address of the object has not been modified. `Tuple` does not support this inplace operation. In the case of `JIT_SYNTAX_LEVEL` set to `LAX`, the static graph mode can support inplace operation of some `List` objects.

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
      return a

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

    With `JIT_SYNTAX_LEVEL` set to `LAX`, part of the `List` built-in function in graph mode supports inplace. All methods do not support inplace operations when `JIT_SYNTAX_LEVEL` is `STRICT`.
    The `List` built-in methods supported by the graph mode are shown in the following table:

    | Method names       | Whether the inplace operation is supported （JIT_SYNTAX_LEVEL=LAX）    |  
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

        With `JIT_SYNTAX_LEVEL` set to `LAX`, `List.extend` supports the inplace operation, which does not generate a new object after the function is run.

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

        With `JIT_SYNTAX_LEVEL` set to `LAX`, `List.pop` supports the inplace operation, which does not generate a new object after the function is run.

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

        The result is as follows:

        ```text
        pop_element:3
        res_list:[1, 2]
        ```

    - List.reverse

        Basic syntax: ```list_object.reverse()```.

        Basic semantics: Reverse the order of the elements of the `List` object `list_object`.

        With `JIT_SYNTAX_LEVEL` set to `LAX`, `List.reverse` supports the inplace operation, which does not generate a new object after the function is run.

        The example is as follows:

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

        The result is as follows:

        ```text
        output1:[3, 2, 1]
        ```

    - List.insert

        Basic syntax: ```list_object.insert(index, target_obj)```.

        Basic semantics: Insert `target_obj` into the `index` position of `list_object`.

        The `index` requires to be the constant `int`. If `list_object` is of length `list_obj_size`, when `index < -list_obj_size`, insert into the first position of `List`. When `index >= -list_obj_size`, insert to the end of `List`. A negative `index` represents the number of digits from back to front.

        With `JIT_SYNTAX_LEVEL` set to `LAX`, `List.insert` supports the inplace operation, and the function runs without generating a new object.

        The example is as follows:

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

        The result is as follows:

        ```text
        output:[1, 2, 3, 4]
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

#### None

Support using and returning None.

For example:

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

### MindSpore User-defined Data Types

Currently, MindSpore supports the following user-defined data types: `Tensor`, `Primitive`, and `Cell`.

#### Tensor

For details of `Tensor`, click [Tensor API document](https://mindspore.cn/docs/en/r2.1/api_python/mindspore/mindspore.Tensor.html#mindspore-tensor).

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

## Operators

Arithmetic operators and assignment operators support the `Number` and `Tensor` operations, as well as the `Tensor` operations of different `dtype`. For more details, please refer to [Operators](https://www.mindspore.cn/docs/en/r2.1/note/static_graph_syntax/operators.html)

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

## Statements

Currently supported Python statements include raise statement, assert statement, pass statement, return statement, break statement, continue statement, if statement, for statement, while statement, with statement, list comprehension, generator expression and function definition statement. For more details, please refer to [Statements](https://www.mindspore.cn/docs/en/r2.1/note/static_graph_syntax/statements.html)

## Python Built-in Functions

Currently supported Python built-in functions include `int`, `float`, `bool`, `str`, `list`, `tuple`, `getattr`, `hasattr`, `len`, `isinstance`, `all`, `any`, `round`, `max`, `min` , `sum`, `abs`, `partial`, `map`, `range`, `enumerate`, `super`, `pow`, `filter`. The use of built-in functions in graph mode is similar to the corresponding Python built-in functions. For more details, please refer to [Python Built-in Functions](https://www.mindspore.cn/docs/en/r2.1/note/static_graph_syntax/python_builtin_functions.html)

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

## JIT Fallback

In MindSpore static diagram mode, users need to follow MindSpore [static diagram syntax support](https://www.mindspore.cn/docs/en/r2.1/note/static_graph_syntax_support.html) when writing programs. Constraints exist on the use of the syntax.In dynamic graph mode, Python script code is executed according to the Python syntax, and users can use any Python syntax. It can be seen that the syntax constraint restrictions are different for static and dynamic graphs.

JIT Fallback considers the unification of static and dynamic graphs from the perspective of static graphs. Through the JIT Fallback feature, static graphs can support as many dynamic diagram syntaxes as possible, making static graphs provide a syntax experience close to that of dynamic graphs, thus achieving dynamic unity. To facilitate the user's ability to choose to use the JIT Fallback feature, the JIT syntax support level option 'jit_syntax_level' is provided. The value must be in [STRICT, LAX]. Default: LAX. All levels support all backends.

STRICT: Only basic syntax is supported, and execution performance is optimal.

LAX: Compatible with all Python syntax as much as possible. However, execution performance may be affected and not optimal.

This document describes the support scope and usage notes of JIT Fallback so that you can use JIT Fallback features more effectively.

The JIT Fallback feature is still being improved, and the following is a list of static graph compilation syntaxes that are currently supported by this feature.

### Annotation

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

### Calling the Third-party Libraries

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

### Supporting Control Flow

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

### Support JIT Fallback in the Runtime Phase

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

### Instructions for Use

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
