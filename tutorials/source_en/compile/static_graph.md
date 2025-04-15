# Introduction to Graph Mode Programming

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/compile/static_graph.md)

## Overview

In Just-In-Time Compilation (JIT) mode, Python code is not executed by the Python interpreter.Instead, the code is compiled into a static
computation graph, and then the static computation graph is executed.

In static graph mode, MindSpore converts Python source code into Intermediate Representation IR by means of source code conversion and
optimizes IR graphs on this basis, and finally executes the optimized graphs on hardware devices. MindSpore uses a functional IR based on
graph representations, called MindIR. See [middle representationMindIR](https://www.mindspore.cn/docs/en/master/design/all_scenarios.html#mindspore-ir-mindir)
for details.

Currently, there are three main methods for converting Python source code into Intermediate Representation (IR): parsing based on the
Abstract Syntax Tree (AST), parsing based on ByteCode, and the method based on operator call tracing (Trace). These three modes differ to some
extent in terms of syntax support. This document will first elaborate in detail on the syntax support in the scenario based on the Abstract
Syntax Tree (AST), and then introduce the differences in syntax support when constructing the computation graph based on ByteCode and operator
tracing (Trace) methods, respectively.

MindSpore static graph execution process actually consists of two steps, corresponding to the Define and Run phases of the static graph, but in
practice, the user will not perceive these two phases when the instantiated Cell object is called. MindSpore encapsulates both phases
in the Cell `__call__` method, so the actual calling process is:

`model(inputs) = model.compile(inputs) + model.construct(inputs)`, where `model` is the instantiated Cell object.

Just-In-Time (JIT) compilation can be achieved using the [JIT interface]{.title-ref} . Another way is to use the Graph mode by setting
`ms.set_context(mode=ms.GRAPH_MODE)`, then write the code in the
`construct` function of the `Cell` so that the code in the `construct` function will be compiled into a static computation graph. For details
about the definition of `Cell`, click [Cell API document](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html).

Due to syntax parsing restrictions, the supported data types, syntax, and related operations during graph building are not completely
consistent with the Python syntax. As a result, some usage is restricted. Borrowing the traditional JIT compilation idea, considers
the unification of static and dynamic graphs from the perspective of graph mode and extends the syntax capabilities of graph patterns. The
static graph provides a syntax experience close to that of the dynamic graph, so as to realize the unity of dynamic and static. In order to
facilitate users to choose whether to extend the static graph syntax, the JIT syntax support level option \'jit_syntax_level\' is provided,
and its value must be in the range of \[STRICT,LAX\], and selecting \'STRICT\' is considered to use the basic syntax and do not extend the
static graph syntax. The default value is \'LAX\'. All backends are supported at all
levels.

- STRICT: Only basic syntaxes is supported, and execution performance is optimal. Can be used for MindIR load and export.
- LAX: Supporting more complex syntaxes, compatible with all Python syntax as much as possible. Cannot be used for MindIR load and
    export due to some syntax that may not be able to be exported.

The following describes the data types, syntax, and related operations supported during static graph building. These rules apply only to JIT
mode. Below is an introduction to the details of syntax support based on the Abstract Syntax Tree (AST).

## AST Basic Syntaxes (STRICT Level)

### Constants and Variables Within JIT

In static graphs, constants and variables are an important concept for understanding static graph syntax, and many syntaxes support different
methods and degrees in the case of constant input and variable input. Therefore, before introducing the specific syntax supported by static
graphs, this section first explains the concepts of constants and variables in static graphs.

In static graph mode, the operation of a program is divided into compilation period and execution period. During compilation, the program
is compiled into an intermediate representation graph, and the program does not actually execute, but statically parses the intermediate
representation through abstract deduction. This makes it impossible to guarantee that we will get the values of all intermediate representation
nodes at compile time. Constants and variables are distinguished by their true values in the compiler.

- Constant: The amount of value that can be obtained during compilation.
- Variable: The amount of value that cannot be obtained during compilation.

#### Constants Generate Scenes

- Scalars, lists, and tuples entered as graph mode are constants
    (without using the mutable interface). For example:

    ``` python
    import mindspore
    from mindspore import nn

    a = 1
    b = [1, 2]
    c = ("a", "b", "c")

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self, a, b, c):
          return a, b, c

    net = Net()
    ret = net(a, b, c)
    print(ret)
    ```

    In the above code, enter `a`, `b`, `c` are constants.

- The result of the constant operation is constant. For example:

    ``` python
    import mindspore
    from mindspore import nn

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self):
          a = 1
          b = "2"
          c = mindspore.tensor([1, 2, 3])
          return a, b, c

    net = Net()
    ret = net()
    print(ret)
    ```

    In the above code, enter `a`, `b`, `c` are constants.

- Constant operations obtain a constant result. For example:

    ``` python
    import mindspore
    from mindspore import nn

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self):
          a = mindspore.tensor([1, 2, 3])
          b = mindspore.tensor([1, 1, 1])
          c = a + b
          return c

    net = Net()
    ret = net()
    print(ret)
    ```

    In the above code, `a` and `b` are constants of Tensor generated in
    the graph mode, so the result of their calculation is also constant.
    However, if one of them is a variable, its return value will also be
    a variable.

#### Variables Generate Scenes

- The return value of all mutable interfaces is a variable (whether
    mutable is used outside the graph or inside the graph). For example:

    ``` python
    import mindspore
    from mindspore import nn

    a = mindspore.mutable([mindspore.tensor([1]), mindspore.tensor([2])])

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self, a):
          b = mindspore.mutable(mindspore.tensor([3]))
          c = mindspore.mutable((mindspore.tensor([1]), mindspore.tensor([2])))
          return a, b, c

    net = Net()
    ret = net(a)
    print(ret)
    ```

    In the above code, `a` is generated by calling the mutable interface
    outside the graph, `b` and `c` are generated by calling the mutable
    interface inside the graph, and `a`, `b`, and `c` are variables.

- Tensors that are inputs to static graphs are variables. For example:

    ``` python
    import mindspore
    from mindspore import nn

    a = mindspore.tensor([1])
    b = (mindspore.tensor([1]), mindspore.tensor([2]))

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self, a, b):
          return a, b

    net = Net()
    ret = net(a, b)
    print(ret)
    ```

    In the above code, `a` is the Tensor input as the graph pattern, so
    it is a variable. But `b` is a tuple that is input to the graph
    schema, not a Tensor type, and even if its internal elements are
    Tensor, `b` is a constant.

- What is calculated by variables is the variable

    If a quantity is the output of an operator, then it is in most cases
    variable. For example:

    ``` python
    import mindspore
    from mindspore import nn

    a = mindspore.tensor([1])
    b = mindspore.tensor([2])

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self, a, b):
          c = a + b
          return c

    net = Net()
    ret = net(a, b)
    print(ret)
    ```

    In this case , `c` is the result of calculations of `a` and `b` ,
    and the inputs `a` and `b` used for the calculation are variables ,
    so `c` is also a variable.

### Data Types

#### Built-in Python Data Types

Currently, the following built-in `Python` data types are supported:
`Number`, `String`, `List`, `Tuple`, and `Dictionary`.

##### Number

Supporting `int`, `float`, and `bool`, but does not support `complex` numbers.

`Number` can be defined on the network. That is, the syntax `y = 1`, `y = 1.2`, and `y = True` are supported.

When the data is a constant, the value of the data can be achieved at compile time, the forcible conversion to `Number` is supported in the
network. The syntax `y = int(x)`, `y = float(x)`, and `y = bool(x)` are supported. When the data is a variable, i.e., you can get the value only
at runtime. It also supports data type conversion using built-in
functions [Python Built-in Functions](https://www.mindspore.cn/tutorials/en/master/compile/python_builtin_functions.html)
such as int(), float() and bool(). For example:

``` python
import mindspore
from mindspore import nn

class Net(nn.Cell):
   @mindspore.jit
   def construct(self, x):
      out1 = int(11.1)
      out2 = int(mindspore.tensor([10]))
      return out1, out2

net = Net()
res = net(mindspore.tensor(2))
print("res[0]:", res[0])
print("res[1]:", res[1])
```

The results are as follows:

``` text
res[0]: 11
res[1]: 10
```

Supporting returning Number. For example:

``` python
import mindspore
from mindspore import nn

class Net(nn.Cell):
   @mindspore.jit
   def construct(self, x, y):
      return x + y

net = Net()
res = net(mindspore.mutable(1), mindspore.mutable(2))
print(res)
```

The results are as follows:

``` text
3
```

##### String

`String` can be constructed on the network, i.e., support for using quotes (`'` or `"`) to create strings such as `x = 'abcd'` or
`y = "efgh"`. Convert constants to strings by means of `str()`. Support string concatenation, truncation, and the use of membership operators
(`in` or `not in`) to determine whether a string contains the specified character. Support for formatting string output by inserting a value
into a string with the string format `%s`. Support for using the format string function `str.format()` in constant scenarios.

For example:

``` python
import mindspore
from mindspore import nn

class Net(nn.Cell):
   @mindspore.jit
   def construct(self):
      var1 = 'Hello!'
      var2 = "MindSpore"
      var3 = str(123)
      var4 = "{} is {}".format("string", var3)
      return var1[0], var2[4:9], var1 + var2, var2 * 2, "H" in var1, "My name is %s!" % var2, var4

net = Net()
res = net()
print("res:", res)
```

The results are as follows:

``` text
res: ('H', 'Spore', 'Hello!MindSpore', 'MindSporeMindSpore', True, 'My name is MindSpore!', 'string is 123')
```

##### List

When \'JIT_SYNTAX_LEVEL\' is set to \'LAX\', static graph mode can support the inplace operation of some \'List\' objects,
see [Supporting List Inplace Modification Operations](https://www.mindspore.cn/tutorials/en/master/compile/static_graph.html#supporting-list-inplace-modification-operations-1).

The basic usage scenarios of \'List\' are as follows:

- The graph mode supports creating `Lists` in graph.

    Support creating `List` objects within graph mode, and the elements
    of the `List` objects can contain any of the types supported by the
    graph mode, as well as multiple levels of nesting. For example:

    ``` python
    import numpy as np
    import mindspore
    from mindspore import nn

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self):
          a = [1, 2, 3, 4]
          b = ["1", "2", "a"]
          c = [mindspore.tensor([1]), mindspore.tensor([2])]
          d = [a, b, c, (4, 5)]
          return d
    ```

    The above sample code, all `List` objects can be created normally.

- The graph mode supports returning `List`

    Before MindSpore version 2.0, `List` is converted to `Tuple` when the graph mode returns a `List` object. In MindSpore version 2.0,
    `List` objects can be returned. For example:

    ``` python
    import mindspore
    from mindspore import nn

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self):
          a = [1, 2, 3, 4]
          return a

    net = Net()
    output = net()  # output: [1, 2, 3, 4]
    ```

    In the same way that a `List` is created within a graph mode, the graph mode returns a `List` object that can include any of the types
    supported by the graph mode, as well as multiple levels of nesting.

- The graph mode supports obtaining `List` objects from global
    variables

    ``` python
    import mindspore
    from mindspore import nn

    global_list = [1, 2, 3, 4]

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self):
          global_list.reverse()
          return global_list

    net = Net()
    output = net()  # output: [4, 3, 2, 1]
    ```

    It should be noted that the list returned in the following pattern in the basic scenario is not the same object as the list of global
    variables, and when \'JIT_SYNTAX_LEVEL\' is set to \'LAX\', the returned object and the global object are unified objects.

- Graph mode supports `List` as input

    The graph mode supports `List` as input to static graphs. The elements of the `List` object used as input must be of an input type
    supported by the graph mode, which also supports multiple levels of nesting.

    ``` python
    import mindspore
    from mindspore import nn

    list_input = [1, 2, 3, 4]

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self, x):
          return x

    net = Net()
    output = net(list_input)  # output: [1, 2, 3, 4]
    ```

    It should be noted that when \'List\' is input as a static graph, it
    is always treated as a constant, regardless of the type of element
    inside it.

- Graph mode supports built-in methods for List

    The \'List\' built-in method is described in detail below:

    - List Index Value

        Basic syntax: `element = list_object[index]`.

        Basic semantics: Extract the element in the \'List\' object in
        the \'index\' bit (\'index\' starts at 0). Supporting multi-level index values.

        Index value \'index\' supported types include \'int\', \'Tensor\', and \'slice\'. Among them, inputs of type \'int\'
        and \'Tensor\' can support constants and variables, and \'slice\' internal data must be constants that can be determined
        at compile time.

        Examples are as follows:

        ``` python
        import mindspore
        from mindspore import nn

        class Net(nn.Cell):
           @mindspore.jit
           def construct(self):
              x = [[1, 2], 3, 4]
              a = x[0]
              b = x[0][mindspore.tensor([1])]
              c = x[1:3:1]
              return a, b, c

        net = Net()
        a, b, c = net()
        print('a:{}'.format(a))
        print('b:{}'.format(b))
        print('c:{}'.format(c))
        ```

        The results are as follows:

        ``` text
        a:[1, 2]
        b:2
        c:[3, 4]
        ```

    - List index assignment

        Basic syntax: `list_object[index] = target_element`.

        Basic semantics: Assign the element in the \'List\' object at bit \'index\' to \'target_element\' (\'index\' starts at 0).
        Support for multi-tier index assignment.

        Index value \'index\' supported types include \'int\', \'Tensor\', and \'slice\'. Among them, inputs of type \'int\'
        and \'Tensor\' can support constants and variables, and the internal data of \'slice\' must be constant that can be
        determined at compile time.

        The index assignment object \'target_element\' supports all data types supported by graph modes.

        Currently, the \'List\' index assignment does not support the inplace operation, and a new object will be generated after the
        index is assigned. This operation will support the inplace operation in the future.

        Examples are as follows:

        ``` python
        import mindspore
        from mindspore import nn

        class Net(nn.Cell):
           @mindspore.jit
           def construct(self):
              x = [[0, 1], 2, 3, 4]
              x[1] = 10
              x[2] = "ok"
              x[3] = (1, 2, 3)
              x[0][1] = 88
              return x

        net = Net()
        output = net()
        print('output:{}'.format(output))
        ```

        The results are as follows:

        ``` text
        output:[[0, 88], 10, 'ok', (1, 2, 3)]
        ```

    - List.append

        Basic syntax: `list_object.append(target_element)`.

        Basic semantics: Append the element \'target_element\' to the last list_object\' of the \'List\' object.

        Currently, \'List.append\' does not support the inplace operation, and a new object will be generated after append
        element. This operation will support the inplace operation in the future.

        Examples are as follows:

        ``` python
        import mindspore
        from mindspore import nn

        class Net(nn.Cell):
           @mindspore.jit
           def construct(self):
              x = [1, 2, 3]
              x.append(4)
              return x

        net = Net()
        x = net()
        print('x:{}'.format(x))
        ```

        The results are as follows:

        ``` text
        x:[1, 2, 3, 4]
        ```

    - List.clear

        Basic syntax: `list_object.clear()`.

        Base semantics: Empty the elements contained in the \'List\' object \'list_object\'.

        Currently, \'List.clear\' does not support inplace, and a new object will be generated after clear list. This operation will
        support inplace in the future.

        Examples are as follows:

        ``` python
        import mindspore
        from mindspore import nn

        class Net(nn.Cell):
           @mindspore.jit
           def construct(self):
              x = [1, 3, 4]
              x.clear()
              return x

        net = Net()
        x = net()
        print('x:{}'.format(x))
        ```

        The results are as follows:

        ``` text
        x:[]
        ```

    - List.extend

        Basic syntax: `list_object.extend(target)`.

        Basic semantics: Insert all elements inside the \'target\' to the end of the \'List\' object \'list_object\'.

        The supported types for \'target\' are \'Tuple\', \'List\', and \'Tensor\'. Among them, if the \'target\' type is \'Tensor\',
        the \'Tensor\' will be converted to \'List\' before inserting it.

        Examples are as follows:

        ``` python
        import mindspore
        from mindspore import nn

        class Net(nn.Cell):
           @mindspore.jit
           def construct(self):
              x1 = [1, 2, 3]
              x1.extend((4, "a"))
              x2 = [1, 2, 3]
              x2.extend(mindspore.tensor([4, 5]))
              return x1, x2

        net = Net()
        output1, output2 = net()
        print('output1:{}'.format(output1))
        print('output2:{}'.format(output2))
        ```

        The results are as follows:

        ``` text
        output1:[1, 2, 3, 4, 'a']
        output2:[1, 2, 3, Tensor(shape=[1], dtype=Int64, value= [4]), Tensor(shape=[1], dtype=Int64, value= [5])]
        ```

    - List.pop

        Basic syntax: `pop_element = list_object.pop(index=-1)`.

        Basic semantics: Remove the \'index\' element of the \'List\' object \'list_object\' from the \'list_object\' and return the element.

        The \'index\' requires that it must be a constant \'int\', and when \'list_object\' has a length of \'list_obj_size\',
        \'index\' has a value range of \'\[-list_obj_size,list_obj_size-1\]\'. \'index\' is a negative
        number representing the number of digits from back to front. When no \'index\' is entered, the default value is -1, i.e. the
        last element is removed.

        ``` python
        import mindspore
        from mindspore import nn

        class Net(nn.Cell):
           @mindspore.jit
           def construct(self):
              x = [1, 2, 3]
              b = x.pop()
              return b, x

        net = Net()
        pop_element, res_list = net()
        print('pop_element:{}'.format(pop_element))
        print('res_list:{}'.format(res_list))
        ```

        The results are as follows:

        ``` text
        pop_element:3
        res_list:[1, 2]
        ```

    - List.reverse

        Basic syntax: `list_object.reverse()`.

        Basic semantics: Reverse the order of the elements of the \'List\' object \'list_object\'.

        Examples are as follows:

        ``` python
        import mindspore
        from mindspore import nn

        class Net(nn.Cell):
           @mindspore.jit
           def construct(self):
              x = [1, 2, 3]
              x.reverse()
              return x

        net = Net()
        output = net()
        print('output:{}'.format(output))
        ```

        The results are as follows:

        ``` text
        output1:[3, 2, 1]
        ```

    - List.insert

        Basic syntax: `list_object.insert(index, target_obj)`.

        Basic semantics: insert \'target_obj\' into the \'index\' bit of \'list_object\'.

        The \'index\' requirement must be a constant \'int\'. If the length of \'list_object\' is \'list_obj_size\'.
        When \'index \<-list_obj_size\', insert the first place in \'List\'.
        When \'index \>= list_obj_size\', insert at the end of \'List\'. A
        negative \'index\' represents the number of digits from back to front.

        Examples are as follows:

        ``` python
        import mindspore
        from mindspore import nn

        class Net(nn.Cell):
           @mindspore.jit
           def construct(self):
              x = [1, 2, 3]
              x.insert(3, 4)
              return x

        net = Net()
        output = net()
        print('output:{}'.format(output))
        ```

        The results are as follows:

        ``` text
        output:[1, 2, 3, 4]
        ```

##### Tuple

`Tuple` can be constructed on the network, that is, the syntax `y = (1, 2, 3)` is supported. The elements of the tuple `Tuple` cannot
be modified, but indexed access to elements in the tuple `Tuple` is supported, and concatenated combinations of tuples are supported.

- Supported index values

    Support accessing elements in the tuple `Tuple` using square brackets plus subscripted indexes. The index value can be `int`,
    `slice`, `Tensor`, and multi-level index value. That is, the syntax `data = tuple_x[index0][index1]...` is supported.

    Restrictions on the index value `Tensor` are as follows:

    - `Tuple` stores `Cell`. Each `Cell` must be defined before a tuple is defined. The number of input parameters, input
        parameter type, and input parameter `shape` of each `Cell` must be the same. The number of outputs of each `Cell` must be the
        same. The output type must be the same as the output `shape`.
    - The index `Tensor` is a scalar `Tensor` whose `dtype` is `int32`. The value range is `[-tuple_len, tuple_len)`.
    - `CPU`, `GPU` and `Ascend` backend is supported.

    An example of the `int` and `slice` indexes is as follows:

    ``` python
    import numpy as np
    import mindspore
    from mindspore import nn

    t = mindspore.tensor(np.array([1, 2, 3]))

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self):
          x = (1, (2, 3, 4), 3, 4, t)
          y = x[1][1]
          z = x[4]
          m = x[1:4]
          n = x[-4]
          return y, z, m, n

    net = Net()
    y, z, m, n = net()
    print('y:{}'.format(y))
    print('z:{}'.format(z))
    print('m:{}'.format(m))
    print('n:{}'.format(n))
    ```

    The results are as follows:

    ``` text
    y:3
    z:[1 2 3]
    m:((2, 3, 4), 3, 4)
    n:(2, 3, 4)
    ```

    An example of the `Tensor` index is as follows:

    ``` python
    import mindspore
    from mindspore import nn

    class Net(nn.Cell):
       def __init__(self):
          super(Net, self).__init__()
          self.relu = nn.ReLU()
          self.softmax = nn.Softmax()
          self.layers = (self.relu, self.softmax)

       @mindspore.jit
       def construct(self, x, index):
          ret = self.layers[index](x)
          return ret

    x = mindspore.tensor([-1.0], mindspore.float32)

    net = Net()
    ret = net(x, 0)
    print('ret:{}'.format(ret))
    ```

    The results are as follows:

    ``` text
    ret:[0.]
    ```

- Support connection combinations

    Similar to the string `String`, tuples support combining using `+`
    and `*` to get a new tuple `Tuple`, for example:

    ``` python
    import mindspore
    from mindspore import nn

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self):
          x = (1, 2, 3)
          y = (4, 5, 6)
          return x + y, x * 2

    net = Net()
    out1, out2 = net()
    print('out1:{}'.format(out1))
    print('out2:{}'.format(out2))
    ```

    The results are as follows:

    ``` text
    out1:(1, 2, 3, 4, 5, 6)
    out2:(1, 2, 3, 1, 2, 3)
    ```

##### Dictionary

`Dictionary` can be constructed on the network. Each key value `key:value` is separated by a colon `:`, and each key value pair is
separated by a comma `,`. The entire dictionary contains the key-value
pairs using curly braces `{}`. That is, the syntax `y = {"a": 1, "b": 2}` is supported.

The `key` is unique, and if there are multiple identical `keys` in the dictionary, the duplicate `keys` are finalized with the last one and the
value `value` can be non-unique. The key `key` needs to be guaranteed to be immutable. Currently, the `key` can be `String`, `Number`, constant
`Tensor`, or `Tuple` that contains these types. The `value` can be `Number`, `Tuple`, `Tensor`, `List` or `Dictionary`.

- Supported APIs

    `keys`: extracts all `key` values from `dict` to form `Tuple` and return it.

    `values`: extracts all `value` values from `dict` to form `Tuple` and return it.

    `items`: extracts `Tuple` composed of each pair of `value` values and `key` values in `dict` to form `List` and return it.

    `get`: `dict.get(key[, value])` returns the `value` value corresponding to the specified `key`, if the specified `key` does
    not exist, the default value `None` or the set default value `value` is returned .

    `clear`: removes all elements in `dict`.

    `has_key`: `dict.has_key(key)` determines whether the specified `key` exists in `dict`.

    `update`: `dict1.update(dict2)` updates the elements in `dict2` to `dict1`.

    `fromkeys`: `dict.fromkeys(seq([, value]))` is used to create a new `Dictionary`, using the elements in the sequence `seq` as the `key`
    of the `Dictionary`, and the `value` is initial value corresponding to all `key`.

    The example is as follows, where the \'x\' and \'new_dict\' in the return value are a \'Dictionary\', and the support is extended under
    the JIT syntax support level option LAX in graph mode, for more advanced use of Dictionary, please refer to the
    [Supporting the high-level usage of Dictionary](#supporting-the-high-level-usage-of-dictionary) section
    of this article.

    ``` python
    import numpy as np
    import mindspore
    from mindspore import nn

    x = {"a": mindspore.tensor(np.array([1, 2, 3])), "b": mindspore.tensor(np.array([4, 5, 6])), "c": mindspore.tensor(np.array([7, 8, 9]))}

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self):
          x_keys = x.keys()
          x_values = x.values()
          x_items = x.items()
          value_a = x.get("a")
          check_key = x.has_key("a")
          y = {"a": mindspore.tensor(np.array([0, 0, 0]))}
          x.update(y)
          new_dict = x.fromkeys("abcd", 123)
          return x_keys, x_values, x_items, value_a, check_key, x, new_dict

    net = Net()
    x_keys, x_values, x_items, value_a, check_key, new_x, new_dict = net()
    print('x_keys:{}'.format(x_keys))
    print('x_values:{}'.format(x_values))
    print('x_items:{}'.format(x_items))
    print('value_a:{}'.format(value_a))
    print('check_key:{}'.format(check_key))
    print('new_x:{}'.format(new_x))
    print('new_dict:{}'.format(new_dict))
    ```

    The results are as follows:

    ``` text
    x_keys:('a', 'b', 'c')
    x_values:(Tensor(shape=[3], dtype=Int64, value= [1, 2, 3]), Tensor(shape=[3], dtype=Int64, value= [4, 5, 6]), Tensor(shape=[3], dtype=Int64, value= [7, 8, 9]))
    x_items:[('a', Tensor(shape=[3], dtype=Int64, value= [1, 2, 3])), ('b', Tensor(shape=[3], dtype=Int64, value= [4, 5, 6])), ('c', Tensor(shape=[3], dtype=Int64, value= [7, 8, 9]))]
    value_a:[1 2 3]
    check_key:True
    new_x:{'a': Tensor(shape=[3], dtype=Int64, value= [0, 0, 0]), 'b': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6]), 'c': Tensor(shape=[3], dtype=Int64, value= [7, 8, 9])}
    new_dict:{'a': 123, 'b': 123, 'c': 123, 'd': 123}
    ```

#### MindSpore User-defined Data Types

Currently, MindSpore supports the following user-defined data types:
`Tensor`, `Primitive`, and `Cell`.

##### Tensor

For details of `Tensor`, click [Tensor API document](https://mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Tensor.html#mindspore-tensor).

Supporting creating and using Tensor. The ways to create a `Tensor`
include using [tensor function interface](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.tensor.html#mindspore.tensor)
and using the class \'ms.Tensor\' interface. It is recommended to use
the former because users can specify the required dtype. The code case is as follows.

``` python
import mindspore
from mindspore import nn
import numpy as np

class Net(nn.Cell):
   def __init__(self):
      super(Net, self).__init__()

   @mindspore.jit
   def construct(self, x):
      return mindspore.tensor(x, dtype=mindspore.float32)

net = Net()
x = np.array([0, 1, 2, 3])
print(net(x))
```

The results are as follows:

``` text
[0., 1., 2., 3.]
```

##### Primitive

Currently, `Primitive` and its subclass instances can be constructed in
construct.

For example:

``` python
import mindspore
from mindspore import nn, ops
import numpy as np

class Net(nn.Cell):
   def __init__(self):
      super(Net, self).__init__()

   @mindspore.jit
   def construct(self, x):
      reduce_sum = ops.ReduceSum(True) #`Primitive` and its subclass instances can be constructed in construct.
      ret = reduce_sum(x, axis=2)
      return ret

x = mindspore.tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
net = Net()
ret = net(x)
print('ret.shape:{}'.format(ret.shape))
```

The results are as follows:

``` text
ret.shape:(3, 4, 1, 6)
```

Currently, the attributes and APIs related to `Primitive` and its subclasses cannot be called on the network.

For details about the defined `Primitive`, click [Primitive API
document](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Primitive.html#mindspore.ops.Primitive).

##### Cell

Currently, `Cell` and its subclass instances can be constructed on the
network. That is, the syntax `cell = Cell(args...)` is supported.

However, during call, the parameter can be specified only in position
parameter mode, and cannot be specified in the key-value pair mode. That
is, the syntax `cell = Cell(arg_name=value)` is not supported.

Currently, the attributes and APIs related to `Cell` and its subclasses
cannot be called on the network unless they are called through `self` in
`construct` of `Cell`.

For details about the definition of `Cell`, click [Cell API
document](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html).

##### Parameter

`Parameter` is a variable tensor, indicating the parameters that need to
be updated during network training.

For details about the definition of `Parameter`, click
[Parameter API document](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Parameter.html#mindspore.Parameter).

### Operators

Arithmetic operators and assignment operators support the `Number` and
`Tensor` operations, as well as the `Tensor` operations of different
`dtype`. For more details, please refer to
[Operators](https://www.mindspore.cn/tutorials/en/master/compile/operators.html)

### Primaries

Primaries represent the most tightly bound operations of the language.

#### Attribute References and Attribute Modification

An attribute reference is a primary followed by a period and a name.

Using attribute references as l-values in Cell instances of MindSpore requires the following requirements:

- The modified attribute belongs to this `cell` object, i.e. it must be `self.xxx`.
- The attribute is initialized in Cell\'s \'\*\*init\*\*\' function and is of type Parameter.

When the JIT syntax support level option is \'LAX\', can support attribute modification in more situations, see
[Support Attribute Setting and Modification](#supporting-property-setting-and-modification).

Examples are as follows:

``` python
import mindspore
from mindspore import nn

class Net(nn.Cell):
   def __init__(self):
      super().__init__()
      self.weight = mindspore.Parameter(mindspore.tensor(3, mindspore.float32), name="w")
      self.m = 2

   @mindspore.jit
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

``` text
ret:1
```

#### Index Value

Index value of a sequence `Tuple`, `List`, `Dictionary`, `Tensor` which called subscription in Python.

Index value of `Tuple` refers to chapter [Tuple](#tuple) of this page.

Index value of `List` refers to chapter [List](#list) of this page.

Index value of `Dictionary` refers to chapter [Dictionary](#dictionary) of this page.

#### Calls

A call calls a callable object (e.g., `Cell` or `Primitive`) with a possibly empty series of arguments.

For example:

``` python
import mindspore
from mindspore import nn, ops
import numpy as np

class Net(nn.Cell):
   def __init__(self):
      super().__init__()
      self.matmul = ops.MatMul()

   @mindspore.jit
   def construct(self, x, y):
      out = self.matmul(x, y)  # A call of Primitive
      return out

x = mindspore.tensor(np.ones(shape=[1, 3]), mindspore.float32)
y = mindspore.tensor(np.ones(shape=[3, 4]), mindspore.float32)
net = Net()
ret = net(x, y)
print('ret:{}'.format(ret))
```

The results are as follows:

``` text
ret:[[3. 3. 3. 3.]]
```

### Statements

Currently supported Python statements include raise statement, assert statement, pass statement, return statement, break statement, continue
statement, if statement, for statement, while statement, with statement, list comprehension, generator expression and function definition
statement. For more details, please refer to
[Statements](https://www.mindspore.cn/tutorials/en/master/compile/statements.html)

### Python Built-in Functions

Currently supported Python built-in functions include `int`, `float`, `bool`, `str`, `list`, `tuple`, `getattr`, `hasattr`, `len`,
`isinstance`, `all`, `any`, `round`, `max`, `min` , `sum`, `abs`, `partial`, `map`, `range`, `enumerate`, `super`, `pow`, `filter`. The
use of built-in functions in graph mode is similar to the corresponding
Python built-in functions. For more details, please refer to [Python Built-in Functions](https://www.mindspore.cn/tutorials/en/master/compile/python_builtin_functions.html).

### Network Definition

#### Network Input parameters

While calculating gradient for outermost network, only `Tensor` input could be calculated, input of other type will be ignored.

The code example is shown below. Among the input parameter `(x, y,  z)` of outermost network, `x` and `z` are `Tensor` type but `y` is not.
While `grad_net` calculating gradient of the input parameters `(x, y, z)` for the network, gradient of `y` is automatically ignored.
Only gradients of `x` and `z` are calculated, and `(grad_x, grad_y)` is returned.

``` python
import mindspore
from mindspore import nn

class Net(nn.Cell):
   def __init__(self):
      super(Net, self).__init__()

   def construct(self, x, y, z):
      return x + y + z

class GradNet(nn.Cell):
   def __init__(self, net):
      super(GradNet, self).__init__()
      self.forward_net = net

   @mindspore.jit
   def construct(self, x, y, z):
      return mindspore.grad(self.forward_net, grad_position=(0, 1, 2))(x, y, z)

input_x = mindspore.tensor([1])
input_y = 2
input_z = mindspore.tensor([3])

net = Net()
grad_net = GradNet(net)
ret = grad_net(input_x, input_y, input_z)
print('ret:{}'.format(ret))
```

The results are as follows:

``` text
ret:(Tensor(shape=[1], dtype=Int64, value= [1]), Tensor(shape=[1], dtype=Int64, value= [1]))
```

## Syntax Constraints of Basic Syntaxes

The execution graph in graph mode is converted from source code, and not all Python syntax can support it. The following describes some of the
syntax constraints that exist under the basic syntax. More network
compilation problems can be found in [Network compilation](https://www.mindspore.cn/docs/en/master/faq/network_compilation.html).

1. When an undefined class member is used in the `construct` function,
    `AttributeError` exception will be thrown. For example:

    ``` python
    import mindspore
    from mindspore import nn

    class Net(nn.Cell):
       def __init__(self):
          super(Net, self).__init__()

       @mindspore.jit
       def construct(self, x):
          return x + self.y

    net = Net()
    net(1)
    ```

    The error is reported as follows:

    ``` text
    AttributeError: External object has no attribute y
    ```

2. Class methods modified by `classmethod` in `nn.Cell` are not
    supported. For example:

    ``` python
    import mindspore

    class Net(nn.Cell):
       @classmethod
       def func(cls, x, y):
          return x + y

       @mindspore.jit
       def construct(self, x, y):
          return self.func(x, y)

    net = Net()
    out = net(mindspore.tensor(1), mindspore.tensor(2))
    print(out)
    ```

    The error is reported as follows:

    ``` text
    TypeError: The parameters number of the function is 3, but the number of provided arguments is 2.
    ```

3. In graph mode, some Python syntax is difficult to convert to [intermediate MindIR](https://www.mindspore.cn/docs/en/master/design/all_scenarios.html#mindspore-ir-mindir)
    in graph mode. For Python keywords, there are some keywords that are not supported in graph mode: AsyncFunctionDef, Delete, AnnAssign,
    AsyncFor, AsyncWith, Match, Try, Import, ImportFrom, Nonlocal, NamedExpr, Set, SetComp, Await, Yield, YieldFrom. If the relevant
    syntax is used in graph mode, an error message will alert the user.

    If you use the Try statement, the following example is used:

    ``` python
    import mindspore
    from mindspore import nn

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self, x, y):
          global_out = 1
          try:
             global_out = x / y
          except ZeroDivisionError:
             print("division by zero, y is zero.")
          return global_out

    net = Net()
    test_try_except_out = net(1, 0)
    print("out:", test_try_except_out)
    ```

    The error is reported as follows:

    ``` text
    RuntimeError: Unsupported statement 'Try'.
    ```

4. Benchmarking Python built-in data types, except for [Built-in Python Data Types](#built-in-python-data-types) supported in the current
    graph mode, complex \'complex\' and collection \'set\' types are not supported. Some high-level uses of the list \'list\' and dictionary
    \'dictionary\' are not supported in the basic syntax scenario, and need to be supported when the JIT syntax support level option
    \'jit_syntax_level\' is \'LAX\'.

5. In the basic syntax scenario, in addition to the [Python Built-in Functions](https://www.mindspore.cn/tutorials/en/master/compile/python_builtin_functions.html)
    supported in the current graph mode, there are still some built-in functions that are not supported in graph mode. For example:
    basestring, bin, bytearray, callable, chr, cmp, compile, delattr, dir, divmod, eval, execfile, file, frozenset, hash, hex, id, input,
    issubclass, iter, locals, long, memoryview, next, object, oct, open, ord, property, raw_input, reduce, reload, repr, reverse, set, slice,
    sorted, unichr, unicode, vars, xrange, \_\_import\_\_.

6. Python provides a number of third-party libraries that usually need to be called via import statements. In graph mode, when the JIT
    syntax support level is \'STRICT\', you cannot directly use third-party libraries. If you need to use the data types of
    third-party libraries in graph mode or call methods of third-party libraries, you need to support them only if the JIT syntax support
    level option \'jit_syntax_level\' is \'LAX\'.

7. In graph mode, the modification of the attributes of the class outside the graph is not perceived, that is, the modification of the
    attributes of the class outside the graph will not take effect. For example:

    ``` python
    import mindspore
    from mindspore import nn, ops

    class Net(nn.Cell):
       def __init__(self):
          super().__init__()
          self.len = 1

       @mindspore.jit
       def construct(self, inputs):
          x = inputs + self.len
          return x

    inputs = 2
    net = Net()
    print("out1:", net(inputs))
    net.len = 2
    print("out2:", net(inputs))
    ```

    The result of the output will not change:

    ``` text
    out1: 3
    out2: 3
    ```

## AST Extended Syntaxes (LAX level)

The following mainly introduces the static graph syntax supported by the
current extension base on AST compilation.

### Calling the Third-party Libraries

- Third-party libraries.

    1. Python built-in modules and Python standard libraries, such as `os`, `sys`, `math`, `time` and other modules.
    2. Third-party code libraries. Their module paths are under the `site-packages` directory of the Python installation directory,
        which need to be installed first and then imported, such `NumPy` and `Scipy`. It should be noted that MindSpore suites such as
        `mindyolo` and `mindflow` are not treated as third-party libraries. For a detailed list, please refer to the
        `_modules_from_mindspore` list of the
        [parser](https://gitee.com/mindspore/mindspore/blob/master/mindspore/python/mindspore/_extends/parse/parser.py) file.
    3. Modules specified by the environment variable `MS_JIT_IGNORE_MODULES`. In contrast, there is the environment
        variable `MS_JIT_MODULES`. For more details, please refer to
        [Environment Variables](https://www.mindspore.cn/docs/en/master/api_python/env_var_list.html).

- Supporting data types of third-party libraries, allowing calling and returning objects of third-party libraries.

    The code example is as follows.

    ``` python
    import numpy as np
    import mindspore
    from mindspore import nn

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self):
          a = np.array([1, 2, 3])
          b = np.array([4, 5, 6])
          out = a + b
          return out

    net = Net()
    print(net())
    ```

    The results are as follows:

    ``` text
    [5 7 9]
    ```

- Supporting calling methods of third-party libraries.

    The code example is as follows.

    ``` python
    from scipy import linalg
    import mindspore
    from mindspore import nn

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self):
          x = [[1, 2], [3, 4]]
          return linalg.qr(x)

    net = Net()
    out = net()
    print(out[0].shape)
    ```

    The results are as follows:

    ``` text
    (2, 2)
    ```

- Supporting creating Tensor instances by using the data types of the
    third-party library NumPy.

    The code example is as follows.

    ``` python
    import numpy as np
    import mindspore
    from mindspore import nn

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self):
          x = np.array([1, 2, 3])
          out = mindspore.tensor(x) + 1
          return out

    net = Net()
    print(net())
    ```

    The results are as follows:

    ``` text
    [2, 3, 4]
    ```

- The assignment of subscripts for data types in third-party libraries
    is supported.

    The code example is as follows.

    ``` python
    import numpy as np
    import mindspore
    from mindspore import nn

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self):
          x = np.array([1, 2, 3])
          x[0] += 1
          return mindspore.tensor(x)

    net = Net()
    res = net()
    print("res: ", res)
    ```

    The results are as follows:

    ``` text
    res: [2 2 3]
    ```

### Supporting the Use of Custom Classes

Custom classes can be used in graph mode, and classes can be instantiated and object properties and methods can be used.

For example, where \'GetattrClass\' is a user-defined class that does
not use the \'@jit_class\' decoration and does not inherit \'nn. Cell\`.

``` python
import mindspore

class GetattrClass():
   def __init__(self):
      self.attr1 = 99
      self.attr2 = 1

   def method1(self, x):
      return x + self.attr2

class GetattrClassNet(nn.Cell):
   def __init__(self):
      super(GetattrClassNet, self).__init__()
      self.cls = GetattrClass()

   @mindspore.jit
   def construct(self):
      return self.cls.method1(self.cls.attr1)

net = GetattrClassNet()
out = net()
assert out == 100
```

### Basic Operators Support More Data Type

In the syntax of graph mode, the following basic operators in the list
is overloaded: \[\'+\', \'-\',
\'\*\',\'/\',\'//\',\'%\',\'\*\*\',\'\<\<\',\'\>\>\',\'&\',\'\|\',\'\^\',
\'not\', \'==\', \'!=\', \'\<\', \'\>\', \'\<=\', \'\>=\', \'in\', \'not
in\', \'y=x\[0\]\'\].
For more details, please refer to
[Operators](https://www.mindspore.cn/tutorials/en/master/compile/operators.html).
When getting unsupported input type, those operators need to use
extended static graph syntax to support, and make the output consistent with the output in the pynative mode.

The code example is as follows.

``` python
import mindspore
from mindspore import nn

class InnerClass(nn.Cell):
   @mindspore.jit
   def construct(self, x, y):
      return x.asnumpy() + y.asnumpy()

net = InnerClass()
ret = net(mindspore.tensor([4, 5]), mindspore.tensor([1, 2]))
print(ret)
```

The results are as follows:

``` text
[5 7]
```

In the example above, since the output of `x.asnumpy()` is `numpy.ndarray` and is an unsupported input type of `+` in the graph
mode, `x.asnumpy() + y.asnumpy()` will be supported by static graph syntax.

In another example:

``` python
import mindspore
from mindspore import nn

class InnerClass(nn.Cell):
   @mindspore.jit
   def construct(self):
      return (None, 1) in ((None, 1), 1, 2, 3)

net = InnerClass()
print(net())
```

The results are as follows:

``` text
True
```

`tuple` in `tuple` is an unsupported operation in original graph mode, and will be supported by static graph syntax.

### Base Type

Use the JIT Fallback feature to extend support for Python\'s native data types \'List\', \'Dictionary\', \'None\'.

#### Supporting List Inplace Modification Operations

The list \'List\' and tuple \'Tuple\' are the most basic sequential built-in types in Python, and the core difference between \'List\' and
\'Tuple\' is that \'List\' is an object that can be changed, while \'Tuple\' cannot be changed. This means that once \'Tuple\' is created,
it cannot be changed without changing the object address. \'List\', on the other hand, can modify an object without changing its address
through a series of inplace operations. For example:

``` python
a = [1, 2, 3, 4]
a_id = id(a)
a.append(5)
a_after_id = id(a)
assert a_id == a_after_id
```

In the above example code, when you change the \'List\' object through the \'append\' inplace syntax, the address of the object is not changed.
\'Tuple\' does not support this kind of inplace. With \'JIT_SYNTAX_LEVEL\' set to \'LAX\', static graph mode can support the
inplace operation of some \'List\' objects.

The specific usage scenarios are as follows:

- Support for getting the original \'List\' object from a global variable

    In the following example, the static graph gets the \'List\' object,
    performs the inplace operation \'list.reverse()\' supported by graph
    mode on the original object, and returns the original object. It can
    be seen that the object returned by the graph mode has the same ID
    as the original global variable object, that is, the two are the
    same object. If \'JIT_SYNTAX_LEVEL\' is set to the \'STRICT\'
    option, the returned \'List\' object and the global object are two
    different objects.

    ``` python
    import mindspore
    from mindspore import nn

    global_list = [1, 2, 3, 4]

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self):
          global_list.reverse()
          return global_list

    net = Net()
    output = net()  # output: [4, 3, 2, 1]
    assert id(global_list) == id(output)
    ```

- Support for in-place modification of some \'List\' built-in
    functions

    With \'JIT_SYNTAX_LEVEL\' set to \'LAX\', the graph mode section
    \'List\' built-in function supports inplace. In cases where
    \'JIT_SYNTAX_LEVEL\' is \'STRICT\', none of the methods support the inplace operation.

    Currently, the built-in methods for \'List\' in-place modification
    supported by graph mode are \'extend\', \'pop\', \'reverse\', and \'insert\'. The built-in methods \'append\', \'clear\' and index
    assignment do not support in-place modification at the moment, and will be supported in subsequent versions.

    Examples are as follows:

    ``` python
    import mindspore
    from mindspore import nn

    list_input = [1, 2, 3, 4]

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self):
          list_input.reverse()
          return list_input

    net = Net()
    output = net()  # output: [4, 3, 2, 1]  list_input: [4, 3, 2, 1]
    assert id(output) == id(list_input)
    ```

#### Supporting the High-Level Usage of Dictionary

- Support Top Graph Return Dictionary

    Examples are as follows:

    ``` python
    import mindspore
    from mindspore import nn

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self):
          x = {'a': 'a', 'b': 'b'}
          y = x.get('a')
          z = dict(y=y)
          return z

    net = Net()
    out = net()
    print("out:", out)
    ```

    The results are as follows:

    ``` text
    out:{'y': 'a'}
    ```

- Support Dictionary Index Value Retrieval and Assignment

    Examples are as follows:

    ``` python
    import numpy as np
    import mindspore
    from mindspore import nn

    x = {"a": mindspore.tensor(np.array([1, 2, 3])), "b": mindspore.tensor(np.array([4, 5, 6])), "c": mindspore.tensor(np.array([7, 8, 9]))}

    class Net(nn.Cell):
       @mindspore.jit
       def construct(self):
          y = x["b"]
          x["a"] = (2, 3, 4)
          return x, y

    net = Net()
    out1, out2 = net()
    print('out1:{}'.format(out1))
    print('out2:{}'.format(out2))
    ```

    The results are as follows:

    ``` text
    out1:{'a': (2, 3, 4), 'b': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6]), 'c': Tensor(shape=[3], dtype=Int64, value= [7, 8, 9])}
    out2:[4 5 6]
    ```

#### Supporting the Usage of None

\'None\' is a special value in Python that represents null and can be assigned to any variable. Functions that do not have a return value
statement are considered to return \'None\'. At the same time, \'None\' is also supported as the input parameter or return value of the top
graph or subgraph. Support \'None\' as a subscript of a slice as input to \'List\', \'Tuple\', \'Dictionary\'.

Examples are as follows:

``` python
import mindspore
from mindspore import nn

class Net(nn.Cell):
   @mindspore.jit
   def construct(self):
      return 1, "a", None

net = Net()
res = net()
print(res)
```

The results are as follows:

``` text
(1, 'a', None)
```

For functions with no return value, the \'None\' object is returned by
default.

``` python
import mindspore
from mindspore import nn

class Net(nn.Cell):
   @mindspore.jit
   def construct(self):
      x = 3
      print("x:", x)

net = Net()
res = net()
assert res is None
```

As in the example below, \'None\' is used as the default input parameter
for the top graph.

``` python
import mindspore
from mindspore import nn

class Net(nn.Cell):
   @mindspore.jit
   def construct(self, x, y=None):
      if y is not None:
         print("y:", y)
      else:
         print("y is None")
      print("x:", x)
      return y

x = [1, 2]
net = Net()
res = net(x)
assert res is None
```

### Built-in Functions Support More Data Types

Extend the support for built-in functions. Python built-in functions
perfectly support more input types, such as third-party library data types.

For example, in the following example, \'x.asnumpy()\' and
\'np.ndarray\' are both types supported by extensions. More support for
built-in functions can be found in the [Python built-in
functions](https://www.mindspore.cn/tutorials/en/master/compile/python_builtin_functions.html) section.

``` python
import numpy as np
import mindspore
from mindspore import nn

class Net(nn.Cell):
   @mindspore.jit
   def construct(self, x):
      return isinstance(x.asnumpy(), np.ndarray)

x = mindspore.tensor(np.array([-1, 2, 4]))
net = Net()
out = net(x)
assert out
```

### Supporting Control Flow

In order to improve the support of Python standard syntax, realize dynamic and static unification, and extend the support for more data
types in the use of control flow statements. Control flow statements refer to flow control statements such as \'if\', \'for\', and \'while\'.
Theoretically, by extending the supported syntax, it is also supported in control flow scenarios. The code use cases are as follows:

``` python
import numpy as np
import mindspore
from mindspore import nn

class Net(nn.Cell):
   @mindspore.jit
   def construct(self):
      x = np.array(1)
      if x <= 1:
         x += 1
      return mindspore.tensor(x)

net = Net()
res = net()
print("res: ", res)
```

The results are as follows:

``` text
res: 2
```

### Supporting Property Setting and Modification

The specific usage scenarios are as follows:

- Set and modify properties of custom class objects and third-party types

In graph mode, you can set and modify the properties of custom class objects, such as:

``` python
import mindspore
from mindspore import nn

class AssignClass():
   def __init__(self):
      self.x = 1

obj = AssignClass()

class Net(nn.Cell):
   @mindspore.jit
   def construct(self):
      obj.x = 100

net = Net()
net()
print(f"obj.x is: {obj.x}")
```

The results are as follows:

``` text
obj.x is: 100
```

In graph mode, you can set and modify the properties of third-party library objects, such as:

``` python
import numpy as np
import mindspore
from mindspore import nn

class Net(nn.Cell):
   @mindspore.jit
   def construct(self):
      a = np.array([1, 2, 3, 4])
      a.shape = (2, 2)
      return a.shape

net = Net()
shape = net()
print(f"shape is {shape}")
```

The results are as follows:

``` text
shape is (2, 2)
```

- Make changes to the Cell\'s self object, for example:

    ``` python
    import mindspore
    from mindspore import nn

    class Net(nn.Cell):
       def __init__(self):
          super().__init__()
          self.m = 2

       @mindspore.jit
       def construct(self):
          self.m = 3
          return

    net = Net()
    net()
    print(f"net.m is {net.m}")
    ```

    The results are as follows:

    ``` text
    net.m is 3
    ```

    Note that the self object supports property modification and setting. If no attribute is defined in \'\*\*init\*\*\', align the
    PYNATIVE mode, and the graph mode also allows this attribute to be set. For example:

    ``` python
    import mindspore
    from mindspore import nn

    class Net(nn.Cell):
       def __init__(self):
          super().__init__()
          self.m = 2

       @mindspore.jit
       def construct(self):
          self.m2 = 3
          return

    net = Net()
    net()
    ```

- Set and modify Cell objects and jit_class objects in the static graph

    Supporting property modification of objects jit_class graph mode,
    such as:

    ``` python
    import mindspore
    from mindspore import nn

    @mindspore.jit_class
    class InnerClass():
       def __init__(self):
          self.x = 10

    class Net(nn.Cell):
       def __init__(self):
          super(Net, self).__init__()
          self.inner = InnerClass()

       @mindspore.jit
       def construct(self):
          self.inner.x = 100
          return

    net = Net()
    net()
    print(f"net.inner.x is {net.inner.x}")
    ```

    The results are as follows:

    ``` text
    net.inner.x is 100
    ```

### Supporting Derivation

The static graph syntax supported by the extension also supports its use
in derivation, such as:

``` python
import mindspore
from mindspore import nn, ops

class Net(nn.Cell):
   @mindspore.jit
   def construct(self, a):
      x = {'a': a, 'b': 2}
      return a, (x, (1, 2))

net = Net()
out = mindspore.grad(net)(mindspore.tensor([1]))
assert out == 2
```

### Annotation Type

For syntax supported by the runtime extensions, nodes are generated that cannot be derived by type, such as dynamically created Tensors, which
are called `Any` types. Because this type cannot be inferred correctly at compile time, the `Any` type will be operated on with a default
maximum precision of float64 to prevent loss of precision. In order to better optimize performance, it is necessary to reduce the generation of
`Any` type data. When the user can clearly know the specific type that will be generated by the extended syntax, we recommend using Annotation
to specify the corresponding Python statement type, thereby determining
the type of the interpretation node and avoiding the generation of `Any` type.

For example, the difference between the
[Tensor](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor)
class and the
[tensor](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.tensor.html#mindspore.tensor)
interface lies in the use of the Annotation Type mechanism within the tensor interface. When the dtype of the tensor function is determined,
the function uses Annotation to specify the output type, thereby avoiding the generation of Any type. The use of `Annotation Type` only
requires adding a comment `# @jit.typing: () -> tensor_type[float32]` above or after the corresponding Python statement, where
tensor_type\[float32\] after -\> indicates the output type of the annotated statement.

The code example is as follows.

``` python
import mindspore
from mindspore import nn, ops

class Net(nn.Cell):
   def __init__(self):
      super(Net, self).__init__()
      self.abs = ops.Abs()

   @mindspore.jit
   def construct(self, x, y):
      z = x.asnumpy() + y.asnumpy()
      y1 = mindspore.tensor(z, dtype=mindspore.float32)
      y2 = mindspore.tensor(z, dtype=mindspore.float32) # @jit.typing: () -> tensor_type[float32]
      y3 = mindspore.tensor(z)
      y4 = mindspore.tensor(z, dtype=mindspore.float32)
      return self.abs(y1), self.abs(y2), self.abs(y3), self.abs(y4)

net = Net()
x = mindspore.tensor(-1, dtype=mindspore.int32)
y = mindspore.tensor(-1, dtype=mindspore.float32)
y1, y2, y3, y4 = net(x, y)

print(f"y1 value is {y1}, dtype is {y1.dtype}")
print(f"y2 value is {y2}, dtype is {y2.dtype}")
print(f"y3 value is {y3}, dtype is {y3.dtype}")
print(f"y4 value is {y4}, dtype is {y4.dtype}")
```

The results are as follows:

``` text
y1 value is 2.0, dtype is Float32
y2 value is 2.0, dtype is Float32
y3 value is 2.0, dtype is Float64
y4 value is 2.0, dtype is Float64
```

In the above example, you can see the difference related to creating \'Tensor\'. Due to the lack of Annotation indication in the Tensor
class, y3 and y4 cannot infer the correct type and can only perform operations in the highest precision float64. For y2, the corresponding
type for JIT Fallback was specified through Annotation during Tensor creation, allowing it to perform operations according to the specified
type. y1 created the Tensor using the tensor function interface and passed the dtype parameter as an Annotation indication, avoiding the
generation of `Any` type.

## Syntax Constraints of Extended Syntaxes

When using the static graph extension support syntax, note the following points:

1. In order to match the support capability of the dynamic graph. That is, it must be within the scope of dynamic graph syntax, including
    but not limited to data types.
2. When extending the static graph syntax, more syntax is supported,
    but the execution performance may be affected and is not optimal.
3. When extending the static graph syntax, more syntax is supported,
    and the ability to import and export cannot be used with MindIR due to use Python.

## Syntax Based on Bytecode Graph Construction

The method of constructing computation graphs based on bytecode does not
support the relaxed mode. Its syntax support scope is largely consistent
with the strict mode of static graphs, with the main differences
including:

1. When constructing graphs based on bytecode, encountering unsupported
    syntax will not result in an error. Instead, the unsupported parts will
    be split and executed in dynamic graph mode. Therefore, the unsupported
    syntax mentioned later in this document for constructing computation
    graphs based on bytecode refers to syntax that cannot be compiled into
    static graphs, but the normal operation of the network will not be
    affected.

2. When constructing graphs based on bytecode, side-effect operations
    related to attribute settings can be included in the graph. For
    example:

``` python
import mindspore
from mindspore import nn

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.attr = 1

    @mindspore.jit(capture_mode="bytecode")
    def construct(self, x):
        self.attr = x + 1
        return self.attr

net = Net()
x = mindspore.tensor([1, 2, 3], dtype=mindspore.int32)
ret = net(x)

print("ret: ", ret)
print("net.attr: ", net.attr)
```

The results are as follows:

``` text
ret: Tensor(shape=[3], dtype=Int64, value= [2, 3, 4])

net.attr: Tensor(shape=[3], dtype=Int64, value= [2, 3, 4])
```

3\. When constructing graphs based on bytecode, control flow involving
variable scenarios cannot be included in the graph. For related information on variables, please refer to
[Variables Generate Scenes](https://www.mindspore.cn/tutorials/en/master/compile/static_graph.html#variables-generate-scenes) .
An example is as follows:

``` python
import mindspore

@mindspore.jit(capture_mode="bytecode")
def func(x):
    a = 0
    m = x * 3
    for _ in range(m):
        a = a + 1
    return a

x = mindspore.tensor([1], dtype=mindspore.int32)
ret = func(x)

print("ret: ", ret)
```

The results are as follows:

``` text
ret: 3
```

In the above example, m is a variable, so the entire for loop control
flow cannot be included in the graph and needs to be executed in dynamic graph mode.
