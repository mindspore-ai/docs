# Network Compilation

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/faq/network_compilation.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

<font size=3>**Q: What can I do if an error "'self.xx' should be initialized as a 'Parameter' type in the '`__init__`' function" is reported?**</font>

A: If you want to assign for a class member such as `self.xx` in the function `construct`, `self.xx` must have been defined as a [Parameter](<https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Parameter.html>) type in the `__init__` function while the other types are not supported. But the local variable `xx` is not under the regulation.

<br/>

<font size=3>**Q: What can I do if an error "For syntax like 'a is not b', b supports True, False and None" is reported?**</font>

A: For the syntax `is` or `is not`, currently `MindSpore` only supports comparisons with `True`, `False` and `None`. Other types, such as strings, are not supported.

<br/>

<font size=3>**Q: What can I do if an error "Only support comparison with 1 operator, but got 2" is reported?**</font>

A: For comparison statements, `MindSpore` supports at most one operator. For example, you can use `1 < x and x < 3` to take the place of `1 < x < 3`.

<br/>

<font size=3>**Q: What can I do if an error "TypeError: For 'Cell', the function construct requires 1 positional argument and 0 default argument, total 1, but got 2" is reported?**</font>

A: When you call the instance of a network, the function `construct` will be executed. And the program will check the number of parameters required by the function `construct` and the number of parameters actually given. If they are not equal, the above exception will be thrown.
Please check whether the number of parameters passed in when the instance of the network in the script is called matches the number of parameters required by the `construct` function in the defined network.

<br/>

<font size=3>**Q: What can I do if an error "TypeError: Do not support to convert <class xxx> object into graph node." is reported?**</font>

A: This error message indicates the object that can not be parsed is used in network compilation. For example, when using the object of the customized class in graph mode, the class needs to be decorated with `ms_class`. Otherwise, this error will be raised.

<br/>

<font size=3>**Q: What can I do if an error  “TypeError: Do not support to get attribute from xxx object xxx "  is reported?**</font>

A: In `getattr(data, attr)` syntax, `data` can not be a third-party object (e.g., `numpy.ndarray`). You can use `data.attr` instead.

<br/>

<font size=3>**Q: What can I do if an error "Unsupported expression 'Yield'" is reported?**</font>

A: MindSpore does not support the `yield` syntax in graph mode. In addition, if the unsupported syntax `net.trainable_params()` is used in graph mode, the error will also be reported, because its internal implementation uses the `list(filter(iterator))` syntax, which implicitly calls the `yield` syntax. The code sample is as follows:

```python
import mindspore as ms
from mindspore import set_context, nn

class Net(nn.Cell):
    def construct(self):
        return True

class TestNet(nn.Cell):
    def __init__(self):
        super(TestNet, self).__init__()
        self.net = Net()

    def construct(self):
        return net.trainable_params()

set_context(mode=ms.GRAPH_MODE)
net = TestNet()
out = net()
```

The result is as follows:

```text
RuntimeError: Unsupported expression 'Yield'.
More details please refer to syntax support at https://www.mindspore.cn

----------------------------------------------------
- The Traceback of Net Construct Code:
----------------------------------------------------
The function call stack (See file 'analyze_fail.dat' for more details. Get instructions about `analyze_fail.dat` at https://www.mindspore.cn/search?inputValue=analyze_fail.dat):
# 0 In file test.py:13
        return net.trainable_params()
               ^
# 1 In file /home/workspace/mindspore/build/package/mindspore/nn/cell.py:1257
        return list(filter(lambda x: x.requires_grad, self.get_parameters(expand=recurse)))
                                                      ^
```

<br/>

<font size=3>**Q: What can I do if an error "Type Join Failed" or "Shape Join Failed" is reported?**</font>

A: In the inference stage of front-end compilation, the abstract types of nodes, including `type` and `shape`, will be inferred. Common abstract types include `AbstractScalar`, `AbstractTensor`, `AbstractFunction`, `AbstractTuple`, `AbstractList`, etc. In some scenarios, such as multi-branch scenarios, the abstract types of the return values of different branches will be `join` to infer the abstract type of the returned result. If these abstract types do not match, or `type`/`shape` are inconsistent, the above exception will be thrown.

When an error similar to "Type Join Failed: dtype1 = Float32, dtype2 = Float16" appears, it means that the data types are inconsistent, resulting in an exception when joining abstract. According to the provided data types and code line, the error can be quickly located. In addition, the specific abstract information and node information are provided in the error message. You can view the MindIR information through the `analyze_fail.dat` file to locate and solve the problem. For specific introduction of MindIR, please refer to [MindSpore IR (MindIR)](https://www.mindspore.cn/docs/en/master/design/mindir.html). The code sample is as follows:

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn

ms.set_context(mode=ms.GRAPH_MODE)
class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = ops.ReLU()
        self.cast = ops.Cast()

    def construct(self, x, a, b):
        if a > b:    # The type of the two branches has inconsistent return values.
            return self.relu(x)    # shape: (2, 3, 4, 5), dtype:Float32
        else:
            return self.cast(self.relu(x), ms.float16)    # shape: (2, 3, 4, 5)， dtype:Float16

input_x = ms.Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
input_a = ms.Tensor(2, ms.float32)
input_b = ms.Tensor(6, ms.float32)
net = Net()
out_me = net(input_x, input_a, input_b)
```

The result is as follows:

```text
TypeError: Cannot join the return values of different branches, perhaps you need to make them equal.
Type Join Failed: dtype1 = Float32, dtype2 = Float16.
For more details, please refer to https://www.mindspore.cn/search?inputValue=Type%20Join%20Failed.

Inner Message:
The abstract type of the return value of the current branch is AbstractTensor(shape: (2, 3, 4, 5), element: AbstractScalar(Type: Float16, Value: AnyValue, Shape: NoShape), value_ptr: 0x55b9f289d090, value: AnyValue), and that of the previous branch is AbstractTensor(shape: (2, 3, 4, 5), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x55b9f289d090, value: AnyValue).
The node is construct.6:[CNode]13{[0]: construct.6:[CNode]12{[0]: ValueNode<Primitive> Switch, [1]: [CNode]11, [2]: ValueNode<FuncGraph> ✓construct.4, [3]: ValueNode<FuncGraph> ✗construct.5}}, true branch: ✓construct.4, false branch: ✗construct.5

The function call stack (See file 'analyze_fail.dat' for more details. Get instructions about `analyze_fail.dat` at https://www.mindspore.cn/search?inputValue=analyze_fail.dat):
# 0 In file test.py(14)
        if a > b:
        ^
```

When an error similar to "Shape Join Failed: shape1 = (2, 3, 4, 5), shape2 = ()" appears, it means that the `shape` are inconsistent, resulting in an exception when joining abstract. The code sample is as follows:

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn

ms.set_context(mode=ms.GRAPH_MODE)
class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = ops.ReLU()
        self.reducesum = ops.ReduceSum()

    def construct(self, x, a, b):
        if a > b:    # The shape of the two branches has inconsistent return values.
            return self.relu(x)    # shape: (2, 3, 4, 5), dtype:Float32
        else:
            return self.reducesum(x)    # shape:(), dype: Float32

input_x = ms.Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
input_a = ms.Tensor(2, ms.float32)
input_b = ms.Tensor(6, ms.float32)
net = Net()
out = net(input_x, input_a, input_b)
```

The result is as follows:

```text
ValueError: Cannot join the return values of different branches, perhaps you need to make them equal.
Shape Join Failed: shape1 = (2, 3, 4, 5), shape2 = ().
For more details, please refer to https://www.mindspore.cn/search?inputValue=Type%20Join%20Failed.

Inner Message:
The abstract type of the return value of the current branch is AbstractTensor(shape: (), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x55658aa9b090, value: AnyValue), and that of the previous branch is AbstractTensor(shape: (2, 3, 4, 5), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x55658aa9b090, value: AnyValue).
The node is construct.6:[CNode]13{[0]: construct.6:[CNode]12{[0]: ValueNode<Primitive> Switch, [1]: [CNode]11, [2]: ValueNode<FuncGraph> ✓construct.4, [3]: ValueNode<FuncGraph> ✗construct.5}}, true branch: ✓construct.4, false branch: ✗construct.5

The function call stack (See file 'analyze_fail.dat' for more details. Get instructions about `analyze_fail.dat` at https://www.mindspore.cn/search?inputValue=analyze_fail.dat):
# 0 In file test.py(14)
        if a > b:
        ^
```

When an error similar to "Type Join Failed: abstract type AbstractTensor can not join with AbstractTuple" appears, it means that the abstract types do not match, resulting in the failure to join abstract types. The code sample is as follows:

```python
import mindspore.ops as ops
import mindspore as ms
from mindspore import ms_function

x = ms.Tensor([1.0])
y = ms.Tensor([2.0])
grad = ops.GradOperation(get_by_list=False, sens_param=True)
sens = 1.0

def test_net(a, b):
    return a, b

@ms_function
def join_fail():
    sens_i = ops.Fill()(ops.DType()(x), ops.Shape()(x), sens)    # sens_i is a scalar shape: (1), dtype:Float64, value:1.0
    # sens_i = (sens_i, sens_i)
    a = grad(test_net)(x, y, sens_i)    # For a test_net gradient with an output type of tuple(Tensor, Tensor) requires that the type of sens_i be consistent with the output, but sens_i is a Tensor; Setting sens_i = (sens_i, sens_i) before grad can fix the problem.
    return a

join_fail()
```

The result is as follows:

```text
TypeError: Type Join Failed: abstract type AbstractTensor cannot join with AbstractTuple.
For more details, please refer to https://www.mindspore.cn/search?inputValue=Type%20Join%20Failed.

Inner Message:
This: AbstractTensor(shape: (1), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x55c969c44c60, value: Tensor(shape=[1], dtype=Float32, value=[ 1.00000000e+00])), other: AbstractTuple{element[0]: AbstractTensor(shape: (1), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x55c96a9a3bd0, value: Tensor(shape=[1], dtype=Float32, value=[ 1.00000000e+00])), element[1]: AbstractTensor(shape: (1), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x55c96a5f06a0, value: Tensor(shape=[1], dtype=Float32, value=[ 2.00000000e+00])), sequence_nodes: {test_net.3:[CNode]4{[0]: ValueNode<PrimitivePy> MakeTuple, [1]: a, [2]: b}, elements_use_flags: {ptr: 0x55c96ae83400, value: [const vector][1, 1]}}}. Please check the node: test_net.5:a{[0]: a, [1]: test_net}

The function call stack (See file 'analyze_fail.dat' for more details. Get instructions about `analyze_fail.dat` at https://www.mindspore.cn/search?inputValue=analyze_fail.dat):

The function call stack:
# 0 In file test.py(17)
    a = grad(test_net)(x, y, sens_i)
        ^
```

<br/>

<font size=3>**Q: What can I do if an error "The params of function 'bprop' of Primitive or Cell requires the forward inputs as well as the 'out' and 'dout'" is reported during compilation?**</font>

A: The inputs of user-defined back propagation function `bprop` should contain all the inputs of the forward network, `out` and `dout`. The example is as follow:

```python
from mindspore import nn, ops, Tensor
from mindspore import dtype as mstype

class BpropUserDefinedNet(nn.Cell):
        def __init__(self):
            super(BpropUserDefinedNet, self).__init__()
            self.zeros_like = ops.ZerosLike()

        def construct(self, x, y):
            return x + y

        # def bprop(self, x, y, out, dout):    # Correct usage
        def bprop(self, x, y, out):
            return self.zeros_like(out), self.zeros_like(out)

grad_fn = ops.GradOperation(get_all=True)
net = BpropUserDefinedNet()
x = Tensor(2, mstype.float32)
y = Tensor(6, mstype.float32)
output = grad_fn(net)(x, y)
print(output)
```

The result is as follows:

```text
TypeError: The params of function 'bprop' of Primitive or Cell requires the forward inputs as well as the 'out' and 'dout'.
In file test.py(13)
        def bprop(self, x, y, out):
```

<br/>

<font size=3>**Q: What can I do if an error "There isn't any branch that can be evaluated" is reported during compilation?**</font>

A: When an error similar to "There isn't any branch that can be evaluated" appears,
it means that there may be infinite recursion or loop in the code, which causes each branch of the if condition to be unable to deduce the correct type and dimension information.

The example is as follow:

```python
import mindspore as ms
from mindspore import ms_function

ZERO = ms.Tensor([0], ms.int32)
ONE = ms.Tensor([1], ms.int32)
@ms_function
def f(x):
    y = ZERO
    if x < 0:
        y = f(x - 3)
    elif x < 3:
        y = x * f(x - 1)
    elif x < 5:
        y = x * f(x - 2)
    else:
        y = f(x - 4)
    z = y + 1
    return z

def test_endless():
    ms.set_context(mode=ms.GRAPH_MODE)
    x = ms.Tensor([5], ms.int32)
    f(x)

```

<br/>

<font size=3>**Q: What can I do if an error "Exceed function call depth limit 1000" is reported during compilation?**</font>

A: When Exceed function call depth limit 1000 is displayed, this indicates that there is an infinite recursive loop in the code, or the code is too complex. The type derivation process causes the stack depth to exceed the set maximum depth.

At this time, you can set `set_context(max_call_depth = value)` to change the maximum depth of the stack, and consider simplifying the code logic or checking whether there is infinite recursion or loop in the code.

Otherwise, set max_call_depth can change the recursive depth of MindSpore, and it may also cause exceed the maximum depth of the system stack and cause segment fault. At this time, you may also need to set the system stack depth.

<br/>

<font size=3>**Q: What can I do if an error that 'could not get source code' and 'Mindspore can not compile temporary source code in terminal. Please write source code to a python file and run the file.' is displayed during compilation?**</font>

A: When compiling a network, MindSpore uses `inspect.getsourcelines(self.fn)` to get the file located in the network code. If the network is the temporary code which is edited in terminal, MindSpore will report an error as the title. It can be solved if writing the network to a Python file.

<br/>

<font size=3>**Q: What can I do when an error that 'Corresponding forward node candidate:' and 'Corresponding code candidate:' is reported?**</font>

A: "Corresponding forward node candidate:" is the code in the associated forward network, indicating that the backpropagation operator corresponds to the forward code. "Corresponding code candidate:" means that the operator is fused by these code, and the separator "-" is used to distinguish different code.

For example：

- The operator FusionOp_BNTrainingUpdate_ReLUV2 reported an error and printed the following code:

    ```text
    Corresponding code candidate:
     - In file /home/workspace/mindspore/build/package/mindspore/nn/layer/normalization.py(212)/                return self.bn_train(x,/
       In file /home/workspace/mindspore/tests/st/tbe_networks/resnet.py(265)/        x = self.bn1(x)/
       In file /home/workspace/mindspore/build/package/mindspore/nn/wrap/cell_wrapper.py(109)/        out = self._backbone(data)/
       In file /home/workspace/mindspore/build/package/mindspore/nn/wrap/cell_wrapper.py(356)/        loss = self.network(*inputs)/
       In file /home/workspace/mindspore/build/package/mindspore/train/dataset_helper.py(98)/        return self.network(*outputs)/
     - In file /home/workspace/mindspore/tests/st/tbe_networks/resnet.py(266)/        x = self.relu(x)/
       In file /home/workspace/mindspore/build/package/mindspore/nn/wrap/cell_wrapper.py(109)/        out = self._backbone(data)/
       In file /home/workspace/mindspore/build/package/mindspore/nn/wrap/cell_wrapper.py(356)/        loss = self.network(*inputs)/
       In file /home/workspace/mindspore/build/package/mindspore/train/dataset_helper.py(98)/        return self.network(*outputs)/
    ```

    The code call stack of the first separator points to 'x = self.bn1(x)' on line 265 in the network script file, and the code call stack of the second separator points to 'x = self.bn1(x)' in line 266 of the network script file. It can be seen that the operator FusionOp_BNTrainingUpdate_ReLUV2 is a fusion of these two lines of code.

- The operator Conv2DBackpropFilter reported an error and printed the following code:

    ```text
    In file /home/workspace/mindspore/build/package/mindspore/ops/_grad/grad_nn_ops.py(65)/        dw = filter_grad(dout, x, w_shape)/
    Corresponding forward node candidate:
     - In file /home/workspace/mindspore/build/package/mindspore/nn/layer/conv.py(266)/        output = self.conv2d(x, self.weight)/
       In file /home/workspace/mindspore/tests/st/tbe_networks/resnet.py(149)/        out = self.conv1(x)/
       In file /home/workspace/mindspore/tests/st/tbe_networks/resnet.py(195)/        x = self.a(x)/
       In file /home/workspace/mindspore/tests/st/tbe_networks/resnet.py(270)/        x = self.layer2(x)/
       In file /home/workspace/mindspore/build/package/mindspore/nn/wrap/cell_wrapper.py(109)/        out = self._backbone(data)/
       In file /home/workspace/mindspore/build/package/mindspore/nn/wrap/cell_wrapper.py(356)/        loss = self.network(*inputs)/
       In file /home/workspace/mindspore/build/package/mindspore/train/dataset_helper.py(98)/        return self.network(*outputs)/
    ```

    The first line is the corresponding source code of the operator. The operator is a bprop operator realized by MindSpore. The second line indicates that the operator has an associated forward node, and the fourth line points to 'out = self.conv1(x)' on line 149 of the network script file. In summary, the operator Conv2DBackpropFilter is a bprop operator, and the corresponding forward node is a convolution operator.

<br/>

<font size=3>**Q: What is "JIT Fallback"? What can I do if an error "Should not use Python object in runtime" is reported?**</font>

A: JIT Fallback is to realize the unification of static graph mode and dynamic graph mode from the perspective of static graph. With JIT Fallback feature, the static graph mode can support as many syntaxes in the dynamic graph mode as possible, so that the static graph mode can provide a syntax experience close to that of the dynamic graph mode. The environment variable switch of JIT Fallback is `DEV_ENV_ENABLE_FALLBACK`, and JIT Fallback is enabled by default.

When the errors "Should not use Python object in runtime" and "We suppose all nodes generated by JIT Fallback would not return to outside of graph" appear, it means that there is an incorrect syntax in the code. When using the JIT Fallback feature to process unsupported syntax expressions, corresponding nodes will be generated, which need to be inferred and executed at compile time. Otherwise, these nodes will throw an error when running. The current JIT Fallback conditionally supports some constant scenarios in Graph mode, and it also needs to conform to MindSpore's programming syntax. When you write the code, please refer to [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html) and the document for [JIT Fallback](https://www.mindspore.cn/docs/zh-CN/master/design/jit_fallback.html).

For example, when calling the third-party library NumPy, JIT Fallback supports the syntax of `np.add(x, y)` and `Tensor(np.add(x, y))`, but MindSpore does not support returning the NumPy type. Therefore, the program will report an error. The code sample is as follows:

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE)

class Net(nn.Cell):
    def construct(self):
        x = np.array([1, 2])
        y = np.array([3, 4])
        return np.add(x, y)

net = Net()
out = net()
```

The result is as follows:

```text
RuntimeError: Should not use Python object in runtime, node: ValueNode<InterpretedObject> InterpretedObject: '[4 6]'.
Line: In file test.py(11)
        return np.add(x, y)
        ^

We suppose all nodes generated by JIT Fallback not return to outside of graph. For more information about JIT Fallback, please refer to https://www.mindspore.cn/search?inputValue=JIT%20Fallback
```

When there is an error related to JIT Fallback, please review the code syntax and modify it according to [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html) and the provided code line. If you need to turn off JIT Fallback, you can use `export DEV_ENV_ENABLE_FALLBACK=0`.

<br/>

<font size=3>**Q: Why does screen print "Start compiling and it will take a while. Please wait..." and "End compiling." when running?**</font>

A: When accelerated execution is required, MindSpore will convert Python source code into a function-style IR based on graph representation and do dome optimizations. This process is also known as the compilation process. When printing "Start compiling and it will take a while. Please wait...", MindSpore starts the graph compilation process. When printing "End compiling.", it means the graph compilation process is over.

Currently there are the following two scenarios where the message will print:

- Running networks at the Graph Mode.
- Running functions decorated by `@ms_function`(such as the optimizer `nn.Momentum`) at the PyNative mode.

> One task may trigger multiple compilation processes.

<font size=3>**Q: What does it mean when a warning "On the Ascend platform, when the return value of the control flow subgraph is parameter, the performance may be degraded. The value of the parameter can be returned to improve the performance." is reported?**</font>

A: Since the Ascend platform cannot actually return a memory address, in the whole graph sinking mode, there will be some problems when there are parameters in the return value in the control flow scenario. In order to avoid problems, switch to the unified runtime mode for this scenario, and switch from the whole graph sinking mode to the unified runtime mode, the network performance may be degraded. If the return value of the control flow subgraph only uses the value of the parameter, you can obtain the  parameter value through the value interface of the parameter to avoid performance degradation caused by mode switching.

For example, in the following use case, only the values of "self.param1" and "self.param2" in "InnerNet" are used in the network "Net", and the properties of parameters are not used, so the value interface can be used to avoid performance caused by mode switching deterioration.

```python
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor, Parameter

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

class InnerNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param1 = Parameter(Tensor(1), name="param1")
        self.param2 = Parameter(Tensor(2), name="param2")

    def construct(self, x):
        if x > 0:
            return self.param1.value(), self.param2.value()
        return self.param2.value(), self.param1.value()

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.inner_net = InnerNet()
        self.addn = ops.AddN()

    def construct(self, x, y):
        inner_params = self.inner_net(x)
        out_res = self.addn(inner_params) + y
        return out_res, inner_params[0] + inner_params[1]

input_x = Tensor(3)
input_y = Tensor(5)
net = Net()
out = net(input_x, input_y)
print("out:", out)
```

The result is as follows:

```text
out: (Tensor(shape=[], dtype=Int64, value=8), Tensor(shape=[], dtype=Int64, value=3))
```

<br/>
