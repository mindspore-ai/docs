# JIT Fallback

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/network/jit_fallback.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

In MindSpore static diagram mode, users need to follow MindSpore [static diagram syntax support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html) when writing programs. Constraints exist on the use of the syntax.In dynamic graph mode, Python script code is executed according to the Python syntax, and users can use any Python syntax. It can be seen that the syntax constraint restrictions are different for static and dynamic graphs.

JIT Fallback considers the unification of static and dynamic graphs from the perspective of static graphs. Through the JIT Fallback feature, static graphs can support as many dynamic diagram syntaxes as possible, making static graphs provide a syntax experience close to that of dynamic graphs, thus achieving dynamic unity. To facilitate the user's ability to choose to use the JIT Fallback feature, the JIT syntax support level option 'jit_syntax_level' is provided. The value must be in [STRICT(0), COMPATIBLE(1), LAX(2)]. Default: LAX(2). All levels support all backends.

STRICT(0): Only basic syntax is supported, and execution performance is optimal.
COMPATIBLE(1): Besides basic syntax, supports more syntax, such as operations of dict, list, and scalar.
LAX(2): Compatible with all Python syntax as much as possible. However, execution performance may be affected and not optimal.

This document describes the support scope and usage notes of JIT Fallback so that you can use JIT Fallback features more effectively.

## Support Scope

The JIT Fallback feature is still being improved, and the following is a list of static graph compilation syntaxes that are currently supported by this feature.

## Creating and Using Tensor

JIT Fallback supports creating and using [Tensor](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Tensor.html) in static graph mode.

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

```text
1
```

The above example uses the interface of Tensor class to create a Tensor. In some cases, it may be necessary to create a Tensor at runtime. In this case, you can use either the aforementioned ms.Tensor interface or the [tensor function interface](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.tensor.html#mindspore.tensor)to create a Tensor.
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

```text
1.0
```

## Annotation

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

```text
y1 value is 2.0, dtype is Float32
y2 value is 2.0, dtype is Float32
y3 value is 2.0, dtype is Float64
y4 value is 2.0, dtype is Float64
```

The above examples show the differences in creating Tensors using JIT Fallback Runtime. Due to the lack of Annotation indication in the Tensor class, y3 and y4 cannot infer the correct type and can only perform operations in the highest precision Float64. For y2, the corresponding type for JIT Fallback was specified through Annotation during Tensor creation, allowing it to perform operations according to the specified type. y1 created the Tensor using the tensor function interface and passed the dtype parameter as an Annotation indication, avoiding the generation of Any type.

## Calling the Third-party Libraries

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

## Using Native Print Printing of Python

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

## Using the raise and assert

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

## Calling Python Built-in Functions

MindSpore supports some Python built-in functions in static graph mode, including but not limited to len, isinstance, map, zip, etc. Please refer to [static graph syntax support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html). With JIT Fallback, more uses of Python built-in functions can be supported in constant scenarios. Here is a brief example of some of the supported Python built-in functions.

### dict()

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

### type()

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

## Supporting Control Flow

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

## Support JIT Fallback in the Runtime Phase

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

## The Top-level Graph Supports Returning Basic Types Such as list, dict, scalar, and none

### The Top-level Graph Supports Returning lists

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
[1, "a", True, None, Tensor([2])]
```

### The Top-level Graph Supports Returning dicts

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
{'a': ms.Tensor(np.array(1), ms.int64)}
```

### The Top-level Graph Supports Returning scalars

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

### The Top-level Graph Supports Returning None

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

## Instructions for Use

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