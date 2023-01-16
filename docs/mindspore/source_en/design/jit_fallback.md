# JIT Fallback

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/design/jit_fallback.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore framework supports both static graph mode and dynamic graph mode. In static graph mode, the Python code is first compiled into a static computational graph, and then the static computational graph is executed. Due to the limitations of syntax parsing, users need to follow MindSpore [static graph syntax support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html) when writing programs, and there are constraints on syntax usage restrictions. In the dynamic graph mode, Python code is executed through the Python interpreter and the user can use any Python syntax. As you can see, the compilation process is not the same for static and dynamic graphs, and the syntax constraints are different. For more information about static and dynamic graphs, please refer to [Static and Dynamic Graphs](https://www.mindspore.cn/tutorials/en/master/advanced/compute_graph.html).

JIT Fallback considers the unification of static and dynamic graphs from the perspective of static graphs. Through the JIT Fallback feature, static graphs can support as many dynamic diagram syntaxes as possible, making static graphs provide a syntax experience close to that of dynamic graphs, thus achieving dynamic unity. To facilitate the user's ability to choose whether to use the JIT Fallback feature, the switch `MS_DEV_ENABLE_FALLBACK` is provided and is currently turned on by default. If you need to turn it off, you can use the command: `export MS_DEV_ENABLE_FALLBACK=0`.

This document describes the support scope and usage notes of JIT Fallback so that you can use JIT Fallback features more effectively.

## Support Scope

The current JIT Fallback feature is applied to constant scenarios, which require that the actual value can be determined during compilation. The JIT Fallback feature is still being improved, and the following is a list of static graph compilation syntaxes that are currently supported by this feature.

### Creating and Using Tensor

JIT Fallback supports creating and using [Tensor](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Tensor.html) in static graph mode, and does not support Tensor.asnumpy().

The code case is as follows, and `Tensor(1, dtype=mstype.int32)` is supported by JIT Fallback.

```python
import mindspore as ms
import mindspore.nn as nn

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

### Using Native Print Printing of Python

JIT Fallback supports printing constants in static graph mode by using native print of Python, which is different from [Print operator](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Print.html) prints information at a different time. Python native print is triggered during compilation (at compiling time phase printing), while the Print operator requires the graph to be compiled and sent down to the device side to run before printing (at runtime phase printing).

For the sake of understanding, the following examples are given. tensor_sum involves Tensor summing, i.e. the runtime phase to get the result. When calling print, the actual call is the Print operator in the static graph mode. Refer to [static graph syntax support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html). And np_num is the result of adding up two NumPy constants, i.e., the usage supported by JIT Fallback, so when calling print, the native Python print is used. Because of the different timing of the two prints, it ends up showing np_sum before tensor_sum, i.e. the print result of Python native print supported by JIT Fallback will be before the Print operator.

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

Currently it is not supported to use the same print to print both compile-time and run-time execution information, for example putting np_sum and tensor_sum in the same print will report an error. An example of the error code is as follows:

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn

class Net(nn.Cell):
   def __init__(self):
      super(Net, self).__init__()

   def construct(self, input_x, input_y):
      tensor_sum = input_x + input_y
      x = np.array([1, 2, 3, 4, 5])
      y = np.array([1, 2, 3, 4, 5])
      np_sum = x + y
      print("np_sum: ", np_sum, "tensor_sum: ", tensor_sum)
      return tensor_sum, ms.Tensor(np_sum)

ms.set_context(mode=ms.GRAPH_MODE)
x = ms.Tensor(np.array([1, 2, 3, 4, 5]))
y = ms.Tensor(np.array([1, 2, 3, 4, 5]))
net = Net()
net(x, y)
```

The error message is as follows:

```text
ValueError: When using JIT Fallback to handle script 'print("np_sum: ", np_sum, "tensor_sum: ", tensor_sum)', the inputs should be constant, but found variable 'tensor_sum' to be nonconstant.
```

### Using the raise and assert

JIT Fallback supports the use of raise and assert in static graph mode.

When using raise, it is required that conditional statements and thrown exception statements conform to the conditions of the constant scenario, otherwise unpredictable results may occur. The correct code example is as follows:

```python
import mindspore.nn as nn
import mindspore as ms
class Net(nn.Cell):
   def __init__(self):
      super(Net, self).__init__()

   def construct(self, x):
      if x <= 0:
         raise ValueError("x should be greater than 0.")
      else:
         x += 1
      return x

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
net(-1)
```

Output the result:

```text
ValueError: x should be greater than 0.
```

Similarly, when using assert, the conditions of the constant scenario need to be met. The correct code example is as follows:

```python
import mindspore.nn as nn
import mindspore as ms

class Net(nn.Cell):
   def __init__(self):
      super(Net, self).__init__()

   def construct(self):
      x = 1
      assert 1 in [2, 3, 4]
      return x

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
net()
```

The output appears normally: `AssertionError`.

### Calling Python Built-in Functions

MindSpore supports some Python built-in functions in static graph mode, including but not limited to len, isinstance, map, zip, etc. Please refer to [static graph syntax support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html). With JIT Fallback, more uses of Python built-in functions can be supported in constant scenarios. Here is a brief example of some of the supported Python built-in functions.

#### dict()

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
print("a: ",a)
print("b: ",b)
print("c: ",c)
print("d: ",d)
```

Output the result:

```text
a: {}
b: {'a': 'a', 'b': 'b', 't': 't'}
c: {'one': 1, 'two': 2, 'three': 3}  
d: {'one': 1, 'two': 2, 'three': 3}
```

#### type()

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

a, b, c, d ,e, f, g = func()
print("a: ",a)
print("b: ",b)
print("c: ",c)
print("d: ",d)
print("e: ",e)
print("f: ",f)
print("g: ",g)
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

> Note: There is another way to use type as a native Python function, i.e. type(name, bases, dict) returns a class object of type name, which is not supported currently because of the low usage scenario.

### Supporting Control Flow in Constant Scenarios

In order to improve Python standard syntax support and achieve dynamic unification in constant scenarios, the use of control flow statements in constant scenarios is achieved through JIT Fallback. Control flow statements are process control statements such as if, for, and while. The JIT Fallback feature supports creating and using Tensor in static graph mode, calling third-party libraries such as Numpy to create and use constants, and supporting some of Python built-in functions. In theory, the constant syntax supported by JIT Fallback is also supported in constant control flow scenarios.

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

## Instructions for Use

When using JIT Fallback, please note the following points:

1. The current JIT Fallback only supports constant scenarios, which require that the actual value can be determined during compilation.

2. The ability of JIT Fallback to support scalar dynamic graphs shall be within the scope of dynamic graph syntax, including but not limited to data types.

3. The current constant control flow scenario does not support the assignment of subscripts to Numpy Array data at this time, and the wrong code example is as follows:

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
   RuntimeError: The 'setitem' operation does not support the type [External, Int64, Int64].
   ```

4. JIT Fallback in the Runtime phase is not supported.

   When JIT Fallback handles unsupported syntax expressions, it will generate corresponding nodes that need to be derived and executed in the compile-time phase, otherwise these nodes will raise an error when passed to the runtime. The sample code is as follows. `np.add(x, y)` will generate the corresponding node, as the return value of the function will be passed to the runtime, an error is reported. In this use case, the computed NumPy data type can be converted to a Tensor type, i.e., the Tensor() method can be called, allowing the program to execute properly.

    ```python
    import numpy as np
    import mindspore as ms

    @ms.jit
    def test_np_add():
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        return np.add(x, y)
        # return Tensor(np.add(x, y)) # If the Tensor() method is called to pass the result, the program will be able to execute normally.

    np_add_res = test_np_add()
    ```

    The error message is reported as follows:

    ```text
    Should not use Python object in runtime, node: ValueNode<InterpretedObject> InterpretedObject: '[2 4 6 8 10]'
    ```

    It should be noted that in the constant scenario, the operation results on NumPy integer data and floating-point data will be converted to constants for storage, so their results can be used as function return values. For example:

    ```python
    import numpy as np
    import mindspore as ms

    @ms.jit
    def test_np_add_constant():
        x = 1.0
        y = 2.0
        return np.add(x, y)

    res = test_np_add_constant()
    print("res:", res)
    ```

    Output the results:

    ```text
    res: 3.0
    ```

5. The NumPy third-party library supported by JIT Fallback and differs from the [mindspore.numpy](https://mindspore.cn/docs/en/master/api_python/mindspore.numpy.html) provided by MindSpore.

    mindspore.numpy is implemented through the operator capabilities of the MindSpore framework and involves operator computation in the runtime phase and cannot derive its results in the compile-time phase (the derivation of variables results in None). The sample code is as follows, using the Tensor() method on the result of `mnp.average(x)`, which does not meet the conditions of the constant scenario, will raise an error.

    ```python
    import mindspore as ms
    import mindspore.numpy as mnp

    @ms.jit
    def test_mnp_average():
        x = mnp.array(([[1., 2.], [3., 4.]]))
        x_average = mnp.average(x)
        return ms.Tensor(x_average)

    out = test_mnp_average()
    print(out)
    ```

    The error message is reported as follows:

    ```text
   TypeError: For 'Tensor', the type of input_data should be one of '['Tensor', 'ndarray', 'str_', 'list', 'tuple', 'float', 'int', 'bool', 'complex']', but got 'None' with type 'NoneType'.
   ```
