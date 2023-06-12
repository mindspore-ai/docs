# Combination of Dynamic and Static Graphs

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/design/dynamic_graph_and_static_graph.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## The Concept of Static and Dynamic Graphs

There are two execution modes of the mainstream deep learning frameworks, namely static graph mode and dynamic graph mode.

In static graph mode, the program firstly generates the graph structure of the neural network, then executes the computational operations involved in the graph during compilation execution. Therefore, in static graph mode, the compiler uses techniques such as graph optimization to optimize the execution graph to a greater extent, resulting in better execution performance that helps scale deployment and cross-platform operation.

In dynamic graph mode, the program is executed in the order in which the code is written, and the reverse execution graph is dynamically generated during the execution of the forward process based on the principle of backward propagation. In this mode, the compiler sends down the individual operators in the neural network for execution one by one, making it easy for the user to write and debug the neural network model.\

## MindSpore Static Graph

In MindSpore, the static graph mode, also known as Graph mode, can be set to static graph mode by `set_context(mode=GRAPH_MODE)`. Static graph mode is more suitable for scenarios where the network is fixed and high performance is required. In static graph mode, the compiler can perform global optimization for the graph based on techniques such as graph optimization, whole graph offloading of computational graph. Therefore, better performance can be obtained under static graph, but the execution graph is converted from the source code. Not all Python syntax is supported under static graphs.

### Graph Mode Execution Principle

In Graph mode, MindSpore converts Python source code into IR by means of source code conversion, then performs relevant graph optimization based on this, and finally executes the optimized graph on hardware devices. MindSpore uses a functional IR based on graph representation, i.e. MindIR, by using a semantics close to the ANF functional style. The Graph mode is compiled and optimized based on MindIR. To use the Graph mode, you need to use the `nn.Cell` class and write the execution code in the `construct` function or call the `@jit` decorator.

An code example for the Graph model is shown below:

```python
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = ops.Mul()

    def construct(self, x, y):
        return self.mul(x, y)

x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
y = ms.Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))

net = Net()
print(net(x, y))
```

```text
[ 4. 10. 18.]
```

### Graph Mode Auto-differentiation Principle

In MindSpore, the principle of auto-differentiation in Graph mode can be found in [Auto-differentiation](https://www.mindspore.cn/tutorials/en/master/beginner/autograd.html).

## MindSpore Dynamic Graph

In MindSpore, dynamic graph mode is also known as PyNative mode, which can be set to dynamic graph mode by `set_context(mode=PYNATIVE_MODE)`. In script development and network flow debugging, it is recommended to use dynamic graph mode for debugging, which supports the execution of single operators, common functions and networks, and separate gradient solving operations.

### PyNative Mode Execution Principle

In PyNative mode, users can use the full Python API. In addition, for using the API provided by MindSpore, the framework will execute the operations of the operator API on the corresponding hardware platform according to the hardware platform (Ascend, GPU, CPU) selected by the user and return the corresponding results. The overall execution process of the framework is as follows:

![process](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/design/images/framework.png)

Through the front-end Python API, call to the framework layer, and finally to the corresponding hardware devices to perform calculations. For example, to complete an addition

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x = ms.Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
y = ms.Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
output = ops.add(x, y)
print(output.asnumpy())
```

```text
[[[[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]]]
```

In this example, when the Python interface ops.add(x, y) is called, the Python interface call is called to the C++ layer of the framework via Pybind11, and converted to C++ call. Then the framework will select the corresponding hardware device according to the device_target set by the users, and execute the add operation on that hardware device.

From the above principle, we can see that in PyNative mode, Python script code will be executed according to Python syntax, and the execution process involves MindSpore's API, which will be accelerated by executing on different hardware according to user settings. Therefore, in PyNative mode, users can use Python syntax and debugging methods at will, for example, you can use common IDEs such as PyCharm and VS Code to debug code.

### PyNative Mode Auto-differentiation Principle

In the previous introduction, we can see that the execution of the forward procedure under PyNative is performed exactly according to Python syntax. Under PyNative, backward propagation is implemented based on Tensor. We record all the operations applied to Tensor during the execution of the forward process, and for each operation find its reverse, and string all the reverse processes together to form the overall backward propagation graph (reverse graph for short). Eventually, the reverse graph is executed on the device to calculate the gradient.

The following code is an example of the reverse composition process: multiply the matrix x with a fixed parameter z, then perform matrix multiplication with y, and finally derives x.

```python
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.z = ms.Parameter(ms.Tensor(np.array([2.0], np.float32)), name='z')

    def construct(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return out

class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net

    def construct(self, x, y):
        gradient_function = ms.grad(self.net)
        return gradient_function(x, y)

x = ms.Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=ms.float32)
y = ms.Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=ms.float32)
output = GradNetWrtX(Net())(x, y)
print(output)
```

```text
[[9.02      5.4       7.2000003]
 [9.02      5.4       7.2000003]]
```

According to the above composition principle under PyNative, it can be seen that in the forward propagation process, we record the calculation process of Mul. According to the definition of reverse bprop corresponding to the Mul, we get the reverse MulGrad operator. According to the definition of Mul operator's bprop, as follows:

```python
from mindspore.ops._grad.grad_base import bprop_getters

@bprop_getters.register(ops.Mul)
def get_bprop_mul(self):
    """Grad definition for `Mul` operation."""
    mul_func = P.Mul()

    def bprop(x, y, out, dout):
        bc_dx = mul_func(y, dout)
        bc_dy = mul_func(x, dout)
        return binop_grad_common(x, y, bc_dx, bc_dy)

    return bprop
```

It can be seen that reversing the input to Mul requires backward propagation gradient values of two input and output, at which point z can be connected to MulGrad based on the actual input values. And so on, for the next operator Matmul, the MatmulGrad information is obtained accordingly, and then the contextual gradient propagation is connected according to the input and output of bprop.

Similarly for the input y derivation, the same procedure can be used for the derivation.

### Control flow in PyNative Mode

In the PyNative mode, scripts are executed according to the Python syntax, so in MindSpore, there is no special treatment for the control flow syntax, which is directly expanded and executed according to the Python syntax, and automatic differentiation is performed on the expanded execution operator. For example, for a for loop, the statements in the for loop are continuously executed under PyNative and automatic differentiation is performed on the operators according to the specific number of loops.

## Dynamic and Static Unification

### Overview

The industry currently supports both dynamic and static graph modes. Dynamic graphs are executed by interpretation, with dynamic syntax affinity and flexible expression, and static graphs are executed by using jit compilation optimization, more inclined to static syntax and more restrictions in syntax. For dynamic and static graph modes, firstly MindSpore unifies the API expression, uses the same API in both modes, secondly, unifies the underlying differential mechanism of dynamic and static graphs.

### Interconversion of Dynamic and Static Graphs

In MindSpore, we can switch the execution between using dynamic or static graphs by controlling the mode input parameters. For example:

```python
ms.set_context(mode=ms.PYNATIVE_MODE)
```

Since there are restrictions on Python syntax under static graphs, switching from dynamic to static graphs requires compliance with the syntax restrictions of static graphs in order to execute correctly by using static graphs. For more syntax restrictions for static graphs, refer to [Static Graph Syntax Restrictions](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html).

### Combination of Static and Dynamic

MindSpore supports mixed execution by using static compilation under dynamic graphs. The function objects that need to be executed with static graphs by using jit modification, and in this way you can achieve mixed execution of dynamic and static graphs. For more use of jit, refer to [jit documentation](https://www.mindspore.cn/tutorials/en/master/advanced/compute_graph.html#just-in-time-compilation).

For example:

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn

class AddMulMul(nn.Cell):
    def __init__(self):
        super(AddMulMul, self).__init__()
        self.param = ms.Parameter(ms.Tensor(0.5, ms.float32))

    @ms.jit
    def construct(self, x):
        x = x + x
        x = x * self.param
        x = x * x
        return x

class CellCallSingleCell(nn.Cell):
    def __init__(self):
        super(CellCallSingleCell, self).__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
        self.relu = nn.ReLU()
        self.add_mul_mul = AddMulMul()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.add_mul_mul(x)
        x = self.relu(x)
        return x

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
inputs = ms.Tensor(np.ones([1, 1, 2, 2]).astype(np.float32))
net = CellCallSingleCell()
out = net(inputs)
print(out)
```

```text
[[[[15.99984]]

  [[15.99984]]]]
```

### JIT Fallback

In MindSpore static diagram mode, users need to follow MindSpore [static diagram syntax support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html) when writing programs. Constraints exist on the use of the syntax.In dynamic graph mode, Python script code is executed according to the Python syntax, and users can use any Python syntax. It can be seen that the syntax constraint restrictions are different for static and dynamic graphs.

JIT Fallback considers the unification of static and dynamic graphs from the perspective of static graphs. Through the JIT Fallback feature, static graphs can support as many dynamic diagram syntaxes as possible, making static graphs provide a syntax experience close to that of dynamic graphs, thus achieving dynamic unity. To facilitate the user's ability to choose to use the JIT Fallback feature, the JIT syntax support level option 'jit_syntax_level' is provided. The value must be in [STRICT(0), COMPATIBLE(1), LAX(2)]. Default: LAX(2). All levels support all backends.
STRICT(0): Only basic syntax is supported, and execution performance is optimal.
COMPATIBLE(1): Besides basic syntax, supports more syntax, such as operations of dict, list, and scalar.
LAX(2): Compatible with all Python syntax as much as possible. However, execution performance may be affected and not optimal.

This document describes the support scope and usage notes of JIT Fallback so that you can use JIT Fallback features more effectively.

#### Support Scope

The JIT Fallback feature is still being improved, and the following is a list of static graph compilation syntaxes that are currently supported by this feature.

#### Creating and Using Tensor

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

Output the result:

```text
1
```

The above example uses the interface of Tensor class to create a Tensor.
In some cases, it may be necessary to create a Tensor at runtime.
In this case, you can use either the aforementioned ms.Tensor interface or the [tensor function interface](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.tensor.html#mindspore.tensor)to create a Tensor.
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

MindSpore supports some Python built-in functions in static graph mode, including but not limited to len, isinstance, map, zip, etc. Please refer to [static graph syntax support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html). With JIT Fallback, more uses of Python built-in functions can be supported in constant scenarios. Here is a brief example of some of the supported Python built-in functions.

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
