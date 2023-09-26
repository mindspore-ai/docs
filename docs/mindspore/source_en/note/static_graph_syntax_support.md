# Static Graph Syntax Support

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/static_graph_syntax_support.md)

## Overview

In graph mode, Python code is not executed by the Python interpreter. Instead, the code is compiled into a static computation graph, and then the static computation graph is executed.

In static graph mode, MindSpore converts Python source code into Intermediate Representation IR by means of source code conversion and optimizes IR graphs on this basis, and finally executes the optimized graphs on hardware devices. MindSpore uses a functional IR based on graph representations, called MindIR. See [middle representation MindIR](https://www.mindspore.cn/docs/en/master/design/all_scenarios.html#mindspore-ir-mindir) for details .

MindSpore static graph execution process actually consists of two steps, corresponding to the Define and Run phases of the static graph, but in practice, the user will not perceive these two phases when the instantiated Cell object is called. MindSpore encapsulates both phases in the Cell `__call__` method, so the actual calling process is:

`model(inputs) = model.compile(inputs) + model.construct(inputs)`, where `model` is the instantiated Cell object.

There are two ways to use the graph mode. The first way is to call the `@jit` decorator to modify a function or a class member method, and then the decorated function or method will be compiled into a static computation graph. For details about how to use `jit`, click [jit API document](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.jit.html#mindspore.jit). The second way is to set `ms.set_context(mode=ms.GRAPH_MODE)`, then write the code in the `construct` function of the `Cell` so that the code in the `construct` function will be compiled into a static computation graph. For details about the definition of `Cell`, click [Cell API document](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html).

Due to syntax parsing restrictions, the supported data types, syntax, and related operations during graph building are not completely consistent with the Python syntax. As a result, some usage is restricted. Borrowing the traditional JIT compilation idea, considers the unification of static and dynamic graphs from the perspective of graph mode and extends the syntax capabilities of graph patterns. The static graph provides a syntax experience close to that of the dynamic graph, so as to realize the unity of dynamic and static. In order to facilitate users to choose whether to extend the static graph syntax, the JIT syntax support level option 'jit_syntax_level' is provided, and its value must be in the range of [STRICT,LAX], and selecting 'STRICT' is considered to use the basic syntax and do not extend the static graph syntax. The default value is 'LAX', please refer to the [Extended Syntaxes (LAX level)](#extended-syntaxes-lax-level) section of this article for more information. All backends are supported at all levels.

- STRICT: Only basic syntaxes is supported, and execution performance is optimal. Can be used for MindIR load and export.
- LAX: Supporting more complex syntaxes, compatible with all Python syntax as much as possible. Cannot be used for MindIR load and export due to some syntax that may not be able to be exported.

The following describes the data types, syntax, and related operations supported during static graph building. These rules apply only to graph mode.

## Basic Syntaxes (STRICT Level)

### Constants and Variables Within Static Graphs

In static graphs, constants and variables are an important concept for understanding static graph syntax, and many syntaxes support different methods and degrees in the case of constant input and variable input. Therefore, before introducing the specific syntax supported by static graphs, this section first explains the concepts of constants and variables in static graphs.

In static graph mode, the operation of a program is divided into compilation period and execution period. During compilation, the program is compiled into an intermediate representation graph, and the program does not actually execute, but statically parses the intermediate representation through abstract deduction. This makes it impossible to guarantee that we will get the values of all intermediate representation nodes at compile time. Constants and variables are distinguished by their true values in the compiler.

- Constant: The amount of value that can be obtained during compilation.
- Variable: The amount of value that cannot be obtained during compilation.

In some cases, it is difficult to determine whether a quantity is a constant or a variable, and we can use 'ops.isconstant' to determine whether it is a constant or not. For example:

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

In the above code, 'a' is the variable, so 'm' is 'false'. 'b' is a constant, so 'n' is 'True'.

#### Constants Generate Scenes

- Scalars, lists, and tuples entered as graph mode are constants (without using the mutable interface). For example:

  ```python
  from mindspore import Tensor, jit

  a = 1
  b = [Tensor([1]), Tensor([2])]
  c = ["a", "b", "c"]

  @jit
  def foo(a, b, c):
      return a, b, c
  ```

  In the above code, enter 'a', 'b', 'c' are constants.

- The result of the constant operation is constant. For example:

  ```python
  from mindspore import jit, Tensor

  @jit
  def foo():
      a = 1
      b = "2"
      c = Tensor([1, 2, 3])
      return a, b, c
  ```

  In the above code, enter 'a', 'b', 'c' are constants.

- Constant operations obtain a constant result. For example:

  ```python
  from mindspore import jit, Tensor

  @jit
  def foo():
      a = Tensor([1, 2, 3])
      b = Tensor([1, 1, 1])
      c = a + b
      return c
  ```

  In the above code, 'a' and 'b' are constants of Tensor generated in the graph mode, so the result of their calculation is also constant. However, if one of them is a variable, its return value will also be a variable.

#### Variables Generate Scenes

- The return value of all mutable interfaces is a variable (whether mutable is used outside the graph or inside the graph). For example:

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

  In the above code, 'a' is generated by calling the mutable interface outside the graph, 'b' and 'c' are generated by calling the mutable interface inside the graph, and 'a', 'b', and 'c' are variables.

- Tensors that are inputs to static graphs are variables. For example:

  ```python
  from mindspore import Tensor, jit

  a = Tensor([1])
  b = (Tensor([1]), Tensor([2]))

  @jit
  def foo(a, b):
      return a, b
  ```

  In the above code, 'a' is the Tensor input as the graph pattern, so it is a variable. But 'b' is a tuple that is input to the graph schema, not a Tensor type, and even if its internal elements are Tensor, 'b' is a constant.

- What is calculated by variables is the variable

  If a quantity is the output of an operator, then it is in most cases constant. For example:

  ```python
  from mindspore import Tensor, jit, ops

  a = Tensor([1])
  b = Tensor([2])

  @jit
  def foo(a, b):
      c = a + b
      return c
  ```

  In this case , 'c' is the result of calculations of 'a' and 'b' , and the inputs 'a' and 'b' used for the calculation are variables , so 'c' is also a variable.

### Data Types

#### Built-in Python Data Types

Currently, the following built-in `Python` data types are supported: `Number`, `String`, `List`, `Tuple`, and `Dictionary`.

##### Number

Supporting `int`, `float`, and `bool`, but does not support `complex` numbers.

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
res[1]: 10
res[2]: 2
```

Supporting returning numbers. For example:

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

`String` can be constructed on the network, i.e., support for using quotes (`'` or `"`) to create strings such as `x = 'abcd'` or `y = "efgh"`. Convert constants to strings by means of `str()`. Support string concatenation, truncation, and the use of membership operators (`in` or `not in`) to determine whether a string contains the specified character. Support for formatting string output by inserting a value into a string with the string format `%s`. Support for using the format string function `str.format()` in constant scenarios.

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

##### List

When 'JIT_SYNTAX_LEVEL' is set to 'LAX', static graph mode can support the inplace operation of some 'List' objects, see [Supporting List Inplace Modification Operations](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html#supporting-list-inplace-modification-operations).

The basic usage scenarios of 'List' are as follows:

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
    d = [a, b, c, (4, 5)]
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

  ```python
  import mindspore as ms

  global_list = [1, 2, 3, 4]

  @ms.jit
  def list_func():
      global_list.reverse()
      return global_list

  output = list_func()  # output: [4, 3, 2, 1]
  ```

  It should be noted that the list returned in the following pattern in the basic scenario is not the same object as the list of global variables, and when 'JIT_SYNTAX_LEVEL' is set to 'LAX', the returned object and the global object are unified objects.

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

  It should be noted that when 'List' is input as a static graph, it is always treated as a constant, regardless of the type of element inside it.

- Graph mode supports built-in methods for List

  The 'List' built-in method is described in detail below:

    - List Index Value

      Basic syntax: ```element = list_object[index]```.

      Basic semantics: Extract the element in the 'List' object in the 'index' bit ('index' starts at 0). Supporting multi-level index values.

      Index value 'index' supported types include 'int', 'Tensor', and 'slice'. Among them, inputs of type 'int' and 'Tensor' can support constants and variables, and 'slice' internal data must be constants that can be determined at compile time.

      Examples are as follows:

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

      The results are as follows:

      ```text
      a:[1, 2]
      b:2
      c:[3, 4]
      ```

    - List index assignment

      Basic syntax: ```list_object[index] = target_element```.

      Basic semantics: Assign the element in the 'List' object at bit 'index' to 'target_element' ('index' starts at 0). Support for multi-tier index assignment.

      Index value 'index' supported types include 'int', 'Tensor', and 'slice'. Among them, inputs of type 'int' and 'Tensor' can support constants and variables, and the internal data of 'slice' must be constant that can be determined at compile time.

      The index assignment object 'target_element' supports all data types supported by graph modes.

      Currently, the 'List' index assignment does not support the inplace operation, and a new object will be generated after the index is assigned. This operation will support the inplace operation in the future.

      Examples are as follows:

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

      The results are as follows:

      ```text
      output:[[0, 88], 10, 'ok', (1, 2, 3)]
      ```

    - List.append

      Basic syntax: ```list_object.append(target_element)```.

      Basic semantics: Append the element 'target_element' to the last list_object' of the 'List' object.

      Currently, 'List.append' does not support the inplace operation, and a new object will be generated after index assignment. This operation will support the inplace operation in the future.

      Examples are as follows:

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

      The results are as follows:

      ```text
      x:[1, 2, 3, 4]
      ```

    - List.clear

      Basic syntax: ```list_object.clear()```.

      Base semantics: Empty the elements contained in the 'List' object 'list_object'.

      Currently, 'List.clear' does not support inplace, and a new object will be generated after index assignment. This operation will support inplace in the future.

      Examples are as follows:

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

      The results are as follows:

      ```text
      x:[]
      ```

    - List.extend

      Basic syntax: ```list_object.extend(target)```.

      Basic semantics: Insert all elements inside the 'target' to the end of the 'List' object 'list_object'.

      The supported types for 'target' are 'Tuple', 'List', and 'Tensor'. Among them, if the 'target' type is 'Tensor', the 'Tensor' will be converted to 'List' before inserting it.

      Examples are as follows:

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

      The results are as follows:

      ```text
      output1:[1, 2, 3, 4, 'a']
      output2:[1, 2, 3, Tensor(shape=[1], dtype=Int64, value= [4]), Tensor(shape=[1], dtype=Int64, value= [5])]
      ```

    - List.pop

      Basic syntax: ```pop_element = list_object.pop(index=-1)```.

      Basic semantics: Remove the 'index' element of the 'List' object 'list_object' from the 'list_object' and return the element.

      The 'index' requires that it must be a constant 'int', and when 'list_object' has a length of 'list_obj_size', 'index' has a value range of '[-list_obj_size,list_obj_size-1]'. 'index' is a negative number representing the number of digits from back to front. When no 'index' is entered, the default value is -1, i.e. the last element is removed.

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

      The results are as follows:

      ```text
      pop_element:3
      res_list:[1, 2]
      ```

    - List.reverse

      Basic syntax: ```list_object.reverse()```.

      Basic semantics: Reverse the order of the elements of the 'List' object 'list_object'.

      Examples are as follows:

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

      The results are as follows:

      ```text
      output1:[3, 2, 1]
      ```

    - List.insert

      Basic syntax: ```list_object.insert(index, target_obj)```.

      Basic semantics: insert 'target_obj' into the 'index' bit of 'list_object'.

      The 'index' requirement must be a constant 'int'. If the length of 'list_object' is 'list_obj_size'. When 'index < -list_obj_size', insert the first place in 'List'. When 'index >= -list_obj_size', insert at the end of 'List'. A negative 'index' represents the number of digits from back to front.

      Examples are as follows:

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

      The results are as follows:

      ```text
      output:[1, 2, 3, 4]
      ```

##### Tuple

`Tuple` can be constructed on the network, that is, the syntax `y = (1, 2, 3)` is supported. The elements of the tuple `Tuple` cannot be modified, but indexed access to elements in the tuple `Tuple` is supported, and concatenated combinations of tuples are supported.

- Supported index values

  Support accessing elements in the tuple `Tuple` using square brackets plus subscripted indexes. The index value can be `int`, `slice`, `Tensor`, and multi-level index value. That is, the syntax `data = tuple_x[index0][index1]...` is supported.

  Restrictions on the index value `Tensor` are as follows:

    - `Tuple` stores `Cell`. Each `Cell` must be defined before a tuple is defined. The number of input parameters, input parameter type, and input parameter `shape` of each `Cell` must be the same. The number of outputs of each `Cell` must be the same. The output type must be the same as the output `shape`.

    - The index `Tensor` is a scalar `Tensor` whose `dtype` is `int32`. The value range is `[-tuple_len, tuple_len)`, and negative index is not supported in `Ascend` backend.

    - `CPU`, `GPU` and `Ascend` backend is supported.

  An example of the `int` and `slice` indexes is as follows:

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

##### Dictionary

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

  The example is as follows, where the 'x' and 'new_dict' in the return value are a 'Dictionary', and the support is extended under the JIT syntax support level option LAX in graph mode, for more advanced use of Dictionary, please refer to the [Supporting the high-level usage of Dictionary](#supporting-the-high-level-usage-of-dictionary) section of this article.

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

#### MindSpore User-defined Data Types

Currently, MindSpore supports the following user-defined data types: `Tensor`, `Primitive`, and `Cell`.

##### Tensor

For details of `Tensor`, click [Tensor API document](https://mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Tensor.html#mindspore-tensor).

Supporting creating and using Tensor. The code case is as follows, `Tensor(1, dtype=mstype.int32)` is extended support under the graph mode JIT syntax support level option LAX.

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

The above example uses the interface of 'Tensor' class to create a 'Tensor'. In some cases, it may be necessary to create a runtime 'Tensor', that is, the 'Tensor' data that cannot get the value at compile time, in which case you can use the above class 'ms.Tensor' interface to create 'Tensor', or [tensor function interface](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.tensor.html#mindspore.tensor)to create a Tensor. The code example is shown below.

The difference between it and the 'Tensor' class interface lies in the addition of the [Annotation Type](#annotation-type) tag inside, which can specify the output 'Tensor dtype' during type inference stage to avoid the generation of 'AnyType', with dtype set. When dynamically creating 'Tensor' at runtime, we recommend using this method to create 'Tensor' and hope that users can pass in the expected dtype type to avoid the generation of 'AnyType'.

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

For details about the defined `Primitive`, click [Primitive API document](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Primitive.html#mindspore.ops.Primitive).

##### Cell

Currently, `Cell` and its subclass instances can be constructed on the network. That is, the syntax `cell = Cell(args...)` is supported.

However, during call, the parameter can be specified only in position parameter mode, and cannot be specified in the key-value pair mode. That is, the syntax `cell = Cell(arg_name=value)` is not supported.

Currently, the attributes and APIs related to `Cell` and its subclasses cannot be called on the network unless they are called through `self` in `construct` of `Cell`.

For details about the definition of `Cell`, click [Cell API document](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html).

##### Parameter

`Parameter` is a variable tensor, indicating the parameters that need to be updated during network training.

For details about the definition of `Parameter`, click [Parameter API document](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Parameter.html#mindspore.Parameter).

### Operators

Arithmetic operators and assignment operators support the `Number` and `Tensor` operations, as well as the `Tensor` operations of different `dtype`. For more details, please refer to [Operators](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax/operators.html)

### Primaries

Primaries represent the most tightly bound operations of the language.

#### Attribute References and Attribute Modification

An attribute reference is a primary followed by a period and a name.

Using attribute references as l-values in Cell instances of MindSpore requires the following requirements:

- The modified attribute belongs to this `cell` object, i.e. it must be `self.xxx`.
- The attribute is initialized in Cell's '__init__' function and is of type Parameter.

When the JIT syntax support level option is 'LAX', can support attribute modification in more situations, see [Support Attribute Setting and Modification](#supporting-property-setting-and-modification).

Examples are as follows:

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
        self.weight = x  # The conditions are met, they can be modified
        # self.m = 3     # self.m is not of type Parameter and modification is prohibited
        # y.weight = x   # y is not self, modification is prohibited
        return x

net = Net()
ret = net(1, 2)
print('ret:{}'.format(ret))
```

The results are as follows:

```text
ret:1
```

#### Index Value

Index value of  a sequence `Tuple`, `List`, `Dictionary`, `Tensor` which called subscription in Python.

Index value of `Tuple` refers to chapter [Tuple](#tuple) of this page.

Index value of `List` refers to chapter [List](#list) of this page.

Index value of `Dictionary` refers to chapter [Dictionary](#dictionary) of this page.

Index value of `Tensor` refers to [Tensor index value document](https://www.mindspore.cn/docs/en/master/note/index_support.html#index-values).

#### Calls

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

### Statements

Currently supported Python statements include raise statement, assert statement, pass statement, return statement, break statement, continue statement, if statement, for statement, while statement, with statement, list comprehension, generator expression and function definition statement. For more details, please refer to [Statements](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax/statements.html)

### Python Built-in Functions

Currently supported Python built-in functions include `int`, `float`, `bool`, `str`, `list`, `tuple`, `getattr`, `hasattr`, `len`, `isinstance`, `all`, `any`, `round`, `max`, `min` , `sum`, `abs`, `partial`, `map`, `range`, `enumerate`, `super`, `pow`, `filter`. The use of built-in functions in graph mode is similar to the corresponding Python built-in functions. For more details, please refer to [Python Built-in Functions](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax/python_builtin_functions.html).

### Network Definition

#### Network Input parameters

While calculating gradient for outermost network, only `Tensor` input could be calculated, input of other type will be ignored.

The code example is shown below. Among the input parameter `(x, y,  z)` of outermost network, `x` and `z` are `Tensor` type but `y` is not. While `grad_net` calculating gradient of the input parameters `(x, y, z)` for the network, gradient of `y` is automatically ignored. Only gradients of `x` and `z` are calculated, and `(grad_x, grad_y)` is returned.

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

The result is as follows:

```text
ret:(Tensor(shape=[1], dtype=Int64, value= [1]), Tensor(shape=[1], dtype=Int64, value= [1]))
```

## Syntax Constraints of Basic Syntaxes

The execution graph in graph mode is converted from source code, and not all Python syntax can support it. The following describes some of the syntax constraints that exist under the basic syntax. More network compilation problems can be found in [Network compilation](https://www.mindspore.cn/docs/en/master/faq/network_compilation.html).

1. When an undefined class member is used in the `construct` function, `AttributeError` exception will be thrown. For example:

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

   The result is as follows:

   ```text
   AttributeError: External object has no attribute y
   ```

2. Class methods modified by `classmethod` in `nn.Cell` are not supported. For example:

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

   The result is as follows:

   ```Text
   TypeError: too many positional arguments
   ```

3. In graph mode, some Python syntax is difficult to convert to [intermediate MindIR](https://www.mindspore.cn/docs/en/master/design/all_scenarios.html#mindspore-ir-mindir) in graph mode. For Python keywords, there are some keywords that are not supported in graph mode: AsyncFunctionDef, ClassDef, Delete, AnnAssign, AsyncFor, AsyncWith, Match, Try, Import, ImportFrom, Nonlocal, NamedExpr, Set, SetComp, DictComp, Await, Yield, YieldFrom, Starred. If the relevant syntax is used in graph mode, an error message will alert the user.

   If you use the Try statement, the following example is used:

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

   The result is as follows:

   ```Text
   RuntimeError: Unsupported statement 'Try'.
   ```

4. Benchmarking Python built-in data types, except for [Built-in Python Data Types](#built-in-python-data-types) supported in the current graph mode, complex 'complex' and collection 'set' types are not supported. Some high-level uses of the list 'list' and dictionary 'dictionary' are not supported in the basic syntax scenario, and need to be supported when the JIT syntax support level option 'jit_syntax_level' is 'LAX', please refer to the [Extended Syntaxes (LAX level)](#extended-syntaxes-lax-level) section of this article for more information.

5. In the basic syntax scenario, in addition to the [Python Built-in Functions](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax/python_builtin_functions.html) supported in the current graph mode, there are still some built-in functions that are not supported in graph mode. For example: basestring, bin, bytearray, callable, chr, cmp, compile, delattr, dir, divmod, eval, execfile, file, frozenset, hash, hex, id, input, issubclass, iter, locals, long, memoryview, next, object, oct, open, ord, property, raw_input, reduce, reload, repr, reverse, set, slice, sorted, unichr, unicode, vars, xrange, \_\_import\_\_.

6. Python provides a number of third-party libraries that usually need to be called via import statements. In graph mode, when the JIT syntax support level is 'STRICT', you cannot directly use third-party libraries. If you need to use the data types of third-party libraries in graph mode or call methods of third-party libraries, you need to support them only if the JIT syntax support level option 'jit_syntax_level' is 'LAX', please refer to the [Calling the Third-party Libraries](#calling-the-third-party-libraries) section in [Extended Syntaxes (LAX level)](#extended-syntaxes-lax-level) of this article.

7. In graph mode, when the JIT syntax support level is 'STRICT', you cannot directly use objects, properties, and methods of custom classes. If you need to use custom classes in graph mode, refer to the [Supporting the Use of Custom Classes](#supporting-the-use-of-custom-classes) section of [Extended Syntaxes (LAX level)](#extended-syntaxes-lax-level) in this article.

   For example:

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

    There will be related errors:

    ```Text
    TypeError: Do not support to convert <class '__main__.GetattrClass'> object into graph node.
    ```

## Extended Syntaxes (LAX level)

The following mainly introduces the static graph syntax supported by the current extension.

### Calling the Third-party Libraries

- Third-party libraries.

  1. Python built-in modules and Python standard libraries, such as `os`, `sys`, `math`, `time` and other modules.

  2. Third-party code libraries. Their module paths are under the `site-packages` directory of the Python installation directory, which need to be installed first and then imported, such `NumPy` and `Scipy`. It should be noted that MindSpore suites such as `mindyolo` and `mindflow` are not treated as third-party libraries.

  3. Modules specified by the environment variable `MS_JIT_IGNORE_MODULES`. In contrast, there is the environment variable `MS_JIT_MODULES`. For more details, please refer to [Environment Variables](https://www.mindspore.cn/docs/en/master/note/env_var_list.html).

- Supporting data types of third-party libraries, allowing calling and returning objects of third-party libraries.

  The code example is as follows.

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

  The result is as follows:

  ```text
  [5 7 9]
  ```

- Supporting calling methods of third-party libraries.

  The code example is as follows.

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

  The result is as follows:

  ```text
  (2, 2)
  ```

- Supporting creating Tensor instances by using the data types of the third-party library NumPy.

  The code example is as follows.

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

  The result is as follows:

  ```text
  [2, 3, 4]
  ```

- The assignment of subscripts for data types in third-party libraries is not currently supported.

  The code example is as follows.

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

### Supporting the Use of Custom Classes

Custom classes can be used in graph mode, and classes can be instantiated and object properties and methods can be used.

For example, where 'GetattrClass' is a user-defined class that does not use the '@jit_class' decoration and does not inherit 'nn. Cell`。

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

### Basic Operators Support More Data Type

In the syntax of graph mode, the following basic operators in the list is overloaded: ['+', '-', '*', '/', '//', '%', '**', '<<', '>>', '&', '|', '^', 'not', '==', '!=', '<', '>', '<=', '>=', 'in', 'not in', 'y=x[0]']. For more details, please refer to [Operators](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax/operators.html). When getting unsupported input type, those operators need to use extended static graph syntax to support, and make the output consistent with the output in the pynative mode.

The code example is as follows.

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

In the example above, since the output of `x.asnumpy()` is `numpy.ndarray` and is an unsupported input type of `+` in the graph mode, `x.asnumpy() + y.asnumpy()` will be supported by static graph syntax.

In another example:

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

`tuple` in `tuple`is an unsupported operation in original graph mode，and will be supported by static graph syntax.

### Base Type

Use the JIT Fallback feature to extend support for Python's native data types 'List', 'Dictionary', 'None'.

#### Supporting List Inplace Modification Operations

The list 'List' and tuple 'Tuple' are the most basic sequential built-in types in Python, and the core difference between 'List' and 'Tuple' is that 'List' is an object that can be changed, while 'Tuple' cannot be changed. This means that once 'Tuple' is created, it cannot be changed without changing the object address. 'List', on the other hand, can modify an object without changing its address through a series of inplace operations. For example:

```python
a = [1, 2, 3, 4]
a_id = id(a)
a.append(5)
a_after_id = id(a)
assert a_id == a_after_id
```

In the above example code, when you change the 'List' object through the 'append' inplace syntax, the address of the object is not changed. 'Tuple' does not support this kind of inplace. With 'JIT_SYNTAX_LEVEL' set to 'LAX', static graph mode can support the inplace operation of some 'List' objects.

The specific usage scenarios are as follows:

- Support for getting the original 'List' object from a global variable

  In the following example, the static graph gets the 'List' object, performs the inplace operation 'list.reverse()' supported by graph mode on the original object, and returns the original object. It can be seen that the object returned by the graph mode has the same ID as the original global variable object, that is, the two are the same object. If 'JIT_SYNTAX_LEVEL' is set to the 'STRICT' option, the returned 'List' object and the global object are two different objects.

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

- Inplace operations on input 'List' objects are not supported

  When List' is imported as a static graph, the 'List' object is copied once, and subsequent calculations are performed using the copied object, so it is not possible to perform an inplace operation on the original input object. For example:

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

  As shown in the above use case, the 'List' object cannot be inplaced on the original object when input as a graph mode. The object returned by the graph mode is different from the object ID entered.

- Support for in-place modification of some 'List' built-in functions

  With 'JIT_SYNTAX_LEVEL' set to 'LAX', the graph mode section 'List' built-in function supports inplace. In cases where 'JIT_SYNTAX_LEVEL' is 'STRICT', none of the methods support the inplace operation.

  Currently, the built-in methods for 'List' in-place modification supported by graph mode are 'extend', 'pop', 'reverse', and 'insert'. The built-in methods 'append', 'clear' and index assignment do not support in-place modification at the moment, and will be supported in subsequent versions.

  Examples are as follows:

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

#### Supporting the High-Level Usage of Dictionary

- Support Top Graph Return Dictionary

  Examples are as follows:

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

  The results are as follows:

  ```text
  out:{'y': 'a'}
  ```

- Support Dictionary Index Value Retrieval and Assignment

  Examples are as follows:

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

  The results are as follows:

  ```text
  out1:{'a': (2, 3, 4), 'b': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6]), 'c': Tensor(shape=[3], dtype=Int64, value= [7, 8, 9])}
  out2:[4 5 6]
  ```

#### Supporting the Usage of None

'None' is a special value in Python that represents null and can be assigned to any variable. Functions that do not have a return value statement are considered to return 'None'. At the same time, 'None' is also supported as the input parameter or return value of the top graph or subgraph. Support 'None' as a subscript of a slice as input to 'List', 'Tuple', 'Dictionary'.

Examples are as follows:

```python
import mindspore as ms

@ms.jit
def test_return_none():
    return 1, "a", None

res = test_return_none()
print(res)
```

The results are as follows:

```text
(1, 'a', None)
```

For functions with no return value, the 'None' object is returned by default.

```python
import mindspore as ms

@ms.jit
def foo():
    x = 3
    print("x:", x)

res = foo()
assert res is None
```

As in the example below, 'None' is used as the default input parameter for the top graph.

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

### Built-in Functions Support More Data Types

Extend the support for built-in functions. Python built-in functions perfectly support more input types, such as third-party library data types.

For example, in the following example, 'x.asnumpy()' and 'np.ndarray' are both types supported by extensions. More support for built-in functions can be found in the [Python built-in functions](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax/python_builtin_functions.html) section.

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

### Supporting Control Flow

In order to improve the support of Python standard syntax, realize dynamic and static unification, and extend the support for more data types in the use of control flow statements. Control flow statements refer to flow control statements such as 'if', 'for', and 'while'. Theoretically, by extending the supported syntax, it is also supported in control flow scenarios. The code use cases are as follows:

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

The results are as follows:

```text
res: 2
```

### Supporting Property Setting and Modification

The specific usage scenarios are as follows:

- Set and modify properties of custom class objects and third-party types

In graph mode, you can set and modify the properties of custom class objects, such as:

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

The result is:

```text
obj.x is: 100
```

In graph mode, you can set and modify the properties of third-party library objects, such as:

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

The result is:

```text
shape is (2, 2)
```

- Make changes to the Cell's self object, for example:

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

  The result is:

  ```text
  net.m is 3
  ```

  Note that the self object only supports property modification, not property setting, that is, only supports modifying the properties set in the '__init__' function. If no attribute is defined in '__init__', it is not allowed to be set in graph mode. For example:

  ```python
  import mindspore as ms
  from mindspore import nn, set_context
  set_context(mode=ms.GRAPH_MODE)

  class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.m = 2

    def construct(self):
        self.m2 = 3 # # self.m2 is not set in the __init__, so it cannot be set in graph mode
        return

  net = Net()
  net()
  ```

- Set and modify Cell objects and jit_class objects in the static graph

  Supporting modifying the properties of the graph mode Cell object, such as:

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

  The result is:

  ```text
  net.inner.x is 100
  ```

  Supporting property modification of objects jit_class graph mode, such as:

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

  The result is:

  ```text
  net.inner.x is 100
  ```

  Note that if the same property is obtained before modifying the properties of the Cell/jit_class object in the graph mode, the obtained properties will be parsed as constants. This can cause problems when running the same network multiple times, such as:

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

  The result is:

  ```text
  value0 is 1
  value1 is 2
  value2 is 2
  ```

  But in dynamic graph mode, the value of 'value2' should be 3. However, because 'self.inner.x' in the statement 'a = self.inner.x' is solidified as a constant 2, the value of 'self.inner.x' is set to 2 both times. This issue will be resolved in a subsequent release.

### Supporting Derivation

The static graph syntax supported by the extension also supports its use in derivation, such as:

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

For syntax supported by the runtime extensions, nodes are generated that cannot be derived by type, such as dynamically created Tensors, which are called 'Any' types. Because this type cannot be inferred correctly at compile time, the Anytype will be operated on with a default maximum precision of Float64 to prevent loss of precision. In order to better optimize performance, it is necessary to reduce the generation of Any type data. When the user can clearly know the specific type that will be generated by the extended syntax, we recommend using Annotation to specify the corresponding Python statement type, thereby determining the type of the interpretation node and avoiding the generation of Any type.

For example, the difference between the [Tensor](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor) class and the [tensor](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.tensor.html#mindspore.tensor) interface lies in the use of the Annotation Type mechanism within the tensor interface. When the dtype of the tensor function is determined, the function uses Annotation to specify the output type, thereby avoiding the generation of Any type. The use of `Annotation Type` only requires adding a comment #() -> tensor_type[float32] above or after the corresponding Python statement, where tensor_type[float32] after -> indicates the output type of the annotated statement.

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

```text
y1 value is 2.0, dtype is Float32
y2 value is 2.0, dtype is Float32
y3 value is 2.0, dtype is Float64
y4 value is 2.0, dtype is Float64
```

In the above example, you can see the difference related to creating 'Tensor'. Due to the lack of Annotation indication in the Tensor class, y3 and y4 cannot infer the correct type and can only perform operations in the highest precision Float64.
For y2, the corresponding type for JIT Fallback was specified through Annotation during Tensor creation, allowing it to perform operations according to the specified type.
y1 created the Tensor using the tensor function interface and passed the dtype parameter as an Annotation indication, avoiding the generation of 'Any' type.

## Syntax Constraints of Extended Syntaxes

When using the static graph extension support syntax, note the following points:

1. In order to match the support capability of the dynamic graph. That is, it must be within the scope of dynamic graph syntax, including but not limited to data types.

2. When extending the static graph syntax, more syntax is supported, but the execution performance may be affected and is not optimal.

3. When extending the static graph syntax, more syntax is supported, and the ability to import and export cannot be used with MindIR due to use Python.

4. It is not currently supported that the repeated definition of global variables with the same name across Python files, and these global variables are used in the network.
