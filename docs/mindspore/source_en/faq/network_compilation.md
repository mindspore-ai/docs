# Network Compilation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/faq/network_compilation.md)

## Q: What is the set of syntaxes supported by static graph mode?

A: Static graph mode can support a subset of common Python syntax to support the construction and training of neural networks. Some Python syntax is not supported yet. For more detailed supported syntax set, please refer to [Static Graph Syntax Support](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/compile/static_graph.html). In order to facilitate users to choose whether to extend the static graph syntax, the static graph mode provides JIT syntax support level options. For some network scenarios, it is recommended to use basic syntax (nn/ops, etc.) rather than extended syntax (such as numpy third-party library). In addition, it is recommended to use [Advanced Programming Techniques with Static Graphs](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/compile/static_graph_expert_programming.html) to optimize compilation performance.

<br/>

## Q: What can I do if an error "'self.xx' should be initialized as a 'Parameter' type in the '`__init__`' function" is reported?

A: If you want to assign for a class member such as `self.xx` in the function `construct`, `self.xx` must have been defined as a [Parameter](<https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/mindspore/mindspore.Parameter.html>) type in the `__init__` function while the other types are not supported. But the local variable `xx` is not under the regulation.

<br/>

## Q: What can I do if an error "For syntax like 'a is not b', b supports True, False and None" is reported?

A: For the syntax `is` or `is not`, currently `MindSpore` only supports comparisons with `True`, `False` and `None`. Other types, such as strings, are not supported.

<br/>

## Q: What can I do if an error "TypeError: For 'Cell', the function construct requires 1 positional argument and 0 default argument, total 1, but got 2" is reported?

A: When you call the instance of a network, the function `construct` will be executed. And the program will check the number of parameters required by the function `construct` and the number of parameters actually given. If they are not equal, the above exception will be thrown.
Please check whether the number of parameters passed in when the instance of the network in the script is called matches the number of parameters required by the `construct` function in the defined network.

<br/>

## Q: What can I do if an error "Unsupported expression 'Yield'" is reported?

A: MindSpore does not support the `yield` syntax in graph mode.

<br/>

## Q: What can I do if an error "Type Join Failed" is reported?

A: In the inference stage of front-end compilation, the abstract types of nodes, including `type` and `shape`, will be inferred. Common abstract types include `AbstractScalar`, `AbstractTensor`, `AbstractFunction`, `AbstractTuple`, `AbstractList`, etc. In some scenarios, such as multi-branch scenarios, the abstract types of the return values of different branches will be `join` to infer the abstract type of the returned result. If these abstract types do not match, or `type`/`shape` are inconsistent, the above exception will be thrown.

When an error similar to `Type Join Failed: dtype1 = Float32, dtype2 = Float16` appears, it means that the data types are inconsistent, resulting in an exception when joining abstract. According to the provided data types and code line, the error can be quickly located. In addition, the specific abstract information and node information are provided in the error message. You can view the MindIR information through the `analyze_fail.ir` file to locate and solve the problem. For specific introduction of MindIR, please refer to [MindSpore IR (MindIR)](https://www.mindspore.cn/docs/en/r2.6.0rc1/design/all_scenarios.html#mindspore-ir-mindir). The code sample is as follows:

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
            return self.cast(self.relu(x), ms.float16)    # shape: (2, 3, 4, 5), dtype:Float16

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

The function call stack (See file 'analyze_fail.ir' for more details. Get instructions about `analyze_fail.ir` at https://www.mindspore.cn/search?inputValue=analyze_fail.ir):
# 0 In file test.py(14)
        if a > b:
        ^
```

When an error similar to `Type Join Failed: abstract type AbstractTensor can not join with AbstractTuple` appears, it means that the abstract types do not match, resulting in the failure to join abstract types. The code sample is as follows:

```python
import mindspore.ops as ops
import mindspore as ms

x = ms.Tensor([1.0])
y = ms.Tensor([2.0])
grad = ops.GradOperation(get_by_list=False, sens_param=True)
sens = 1.0

def test_net(a, b):
    return a, b

@ms.jit()
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

The function call stack (See file 'analyze_fail.ir' for more details. Get instructions about `analyze_fail.ir` at https://www.mindspore.cn/search?inputValue=analyze_fail.ir):

The function call stack:
# 0 In file test.py(17)
    a = grad(test_net)(x, y, sens_i)
        ^
```

<br/>

## Q: What can I do if an error `The params of function 'bprop' of Primitive or Cell requires the forward inputs as well as the 'out' and 'dout'` is reported during compilation?

A: The inputs of user-defined back propagation function `bprop` should contain all the inputs of the forward network, `out` and `dout`. The example is as follow:

```python
import mindspore as ms
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

ms.set_context(mode=ms.GRAPH_MODE)
net = BpropUserDefinedNet()
x = Tensor(2, mstype.float32)
y = Tensor(6, mstype.float32)
grad_fn = ms.grad(net, grad_position=(0, 1))
output = grad_fn(x, y)
print(output)
```

The result is as follows:

```text
TypeError: The params of function 'bprop' of Primitive or Cell requires the forward inputs as well as the 'out' and 'dout'.
In file test.py(13)
        def bprop(self, x, y, out):
```

<br/>

## Q: What can I do if an error `There isn't any branch that can be evaluated` is reported during compilation?

A: When an error `There isn't any branch that can be evaluated` appears, it means that there may be infinite recursion or loop in the code, which causes that each branch of the if condition is unable to deduce the correct type and dimension information.

<br/>

## Q: What can I do if an error `Exceed function call depth limit 1000` is reported during compilation?

A: When Exceed function call depth limit 1000 is displayed, this indicates that there is an infinite recursive loop in the code, or the code is too complex. The type derivation process causes the stack depth to exceed the set maximum depth.

At this time, you can set `mindspore.set_recursion_limit(recursion_limit=value)` to change the maximum depth of the stack, and consider simplifying the code logic or checking whether there is infinite recursion or loop in the code.

Otherwise, setting recursion_limit can change the recursive depth of MindSpore, and it may also cause exceed the maximum depth of the system stack and cause segment fault. At this time, you may also need to set the system stack depth.

<br/>

## Q: What can I do if an error that `could not get source code' and 'MindSpore can not compile temporary source code in terminal. Please write source code to a python file and run the file.` is displayed during compilation?

A: When compiling a network, MindSpore uses `inspect.getsourcelines(self.fn)` to get the file located in the network code. If the network is the temporary code which is edited in terminal, MindSpore will report an error as the title. It can be solved if writing the network to a Python file.

<br/>

## Q: What can I do when an error that `'Corresponding forward node candidate:' and 'Corresponding code candidate:'` is reported?

A: "Corresponding forward node candidate:" is the code in the associated forward network, indicating that the backpropagation operator corresponds to the forward code. "Corresponding code candidate:" means that the operator is fused by these code, and the separator "-" is used to distinguish different code.

For example:

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
    In file /home/workspace/mindspore/build/package/mindspore/ops/_grad_experimental/grad_nn_ops.py(65)/        dw = filter_grad(dout, x, w_shape)/
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

## Q: Why does screen print `Start compiling and it will take a while. Please wait...` and `End compiling.` when running?

A: When accelerated execution is required, MindSpore will convert Python source code into a function-style IR based on graph representation and do dome optimizations. This process is also known as the compilation process. When printing "Start compiling and it will take a while. Please wait...", MindSpore starts the graph compilation process. When printing "End compiling.", it means the graph compilation process is over.

Currently there are the following two scenarios where the message will print:

- Running networks at the Graph Mode.
- Running functions decorated by `@jit`(such as the optimizer `nn.Momentum`) at the PyNative mode.

> One task may trigger multiple compilation processes.

<br/>

## Q: What does it mean when a warning `On the Ascend platform, if you read-only access to the parameter, you can take the value of the parameter, so that the system can do more optimization.` is reported?

A: Since the Ascend platform cannot actually return a memory address, in the whole graph sinking mode, there will be some problems when there are parameters in the return value in the control flow scenario. In order to avoid problems, switch to the unified runtime mode for this scenario, and switch from the whole graph sinking mode to the unified runtime mode, the network performance may be degraded. If the return value of the control flow subgraph only uses the value of the parameter, you can obtain the  parameter value through the value interface of the parameter to avoid performance degradation caused by mode switching.

For example, in the following use case, only the values of "self.param1" and "self.param2" in "InnerNet" are used in the network "Net", and the properties of parameters are not used, so the value interface can be used to avoid performance caused by mode switching deterioration.

```python
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor, Parameter

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_device(device_target="Ascend")

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

## Q:What can I do if an error `The input number of parameters is not Compatible.` is reported when loading a MindIR?

A: First, check whether the number of exported parameters and the number of imported parameters match.
If the match, you need to check if a non-Tensor scenario in the exported parameters.

When the exported data input is a non-Tensor, the exported input will be solidified into MindIR as a constant, making the input in MindIR less than the Construct input for network construction.

If the data is a scalar type, you can export the scalar to Tensor type, and if the data is Tuple or List type, you can use the [mutable](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/mindspore/mindspore.mutable.html) interface to encapsulate it and export it.

<br/>

## Q: What can I do if an error `ValueError: The shape of sense must not be dynamic shape.` is reported?

A: In graph mode, when the GradOperation is called and the parameter 'sens_param' is True, and setting the dynamic shape of sense through 'nn.Cell.set_inputs' will cause an error. The code example is as follows:

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_device(device_target="CPU")

class Net(nn.Cell):
    """ReLU Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.relu = ops.ReLU()

    def construct(self, x):
        return self.relu(x)

class GradWithSense(nn.Cell):
    """Grad Net"""
    def __init__(self, network):
        super(GradWithSense, self).__init__()
        self.grad = ops.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_, sense):
        return self.grad(self.network)(input_, sense)

x = np.array([[1, 1], [1, -1]]).astype(np.float32)
sense = np.array([[2, 3], [4, 5]]).astype(np.float32)
dynamic_x = Tensor(shape=[2, None], dtype=ms.float32)
sense_x = Tensor(shape=[1, None], dtype=ms.float32)
net = GradWithSense(Net())
net.set_inputs(dynamic_x, sense_x)
print(net(Tensor(x), Tensor(sense_x))) # ValueError: The shape of sense must not be dynamic shape.
```

In graph mode, the dynamic shape of sense is not supported. It is recommended to change it to the following code:

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_device(device_target="CPU")

class Net(nn.Cell):
    """ReLU Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.relu = ops.ReLU()

    def construct(self, x):
        return self.relu(x)

class NetWithSense(nn.Cell):
    """ReLU Net"""
    def __init__(self, sense):
        super(NetWithSense, self).__init__()
        self.relu = ops.ReLU()
        self.sense = sense

    def construct(self, x):
        return self.relu(x) * self.sense  # Add sense to forward network

class Grad(nn.Cell):
    """Grad Net"""
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation(get_all=True)
        self.network = network

    def construct(self, input_):
        return self.grad(self.network)(input_)

x = np.array([[1, 1], [1, -1]]).astype(np.float32)
sense = np.array([[2, 3], [4, 5]]).astype(np.float32)
dynamic_x = Tensor(shape=[2, None], dtype=ms.float32)
net = Grad(NetWithSense(Tensor(sense)))
net.set_inputs(dynamic_x)
print(net(Tensor(x)))
```

The result is as follows:

```text
(Tensor(shape=[2, 2], dtype=Float32, value=
[[ 2.00000000e+00,  3.00000000e+00],
 [ 4.00000000e+00,  0.00000000e+00]]),)
```

<br/>

## Q: What can I do if an error "'External' TypeError" is reported?

A: The "External" type indicates that an object that cannot be natively supported is used in graph mode. For example: The third-party library object is "External" type.

<br/>

## Q: What can I do if an error `Nested execution during JIT execution for 'xxx' is not supported when 'xxx' compile and execute.` is reported?

A: When the compilation process is triggered, that is, when the code is compiled into a static computational diagram
, see [Graph Mode Execution Principle](https://www.mindspore.cn/docs/en/r2.6.0rc1/features/program_form/overview.html), using the JIT Fallback feature by default, the above exception will be thrown when entering the compilation process again.

Taking JIT Fallback support for calling objects and methods from third-party libraries as an example:

1) call the @jit decorator to modify a function or a class member method, and then the decorated function or method will be compiled into a static computation graph.

```python
from mindspore import context, Tensor, jit, nn
import numpy as np
context.set_context(mode=context.GRAPH_MODE)

class UserDefinedNet: # Customized Python classes
    def __init__(self):
        self.value = 10

    @jit
    def func(self, x):  # Method decorated by jit
        return 2 * x + self.value

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.net = UserDefinedNet()

    def construct(self, x):
        x = self.net.value + self.net.func(x)
        return x

x = np.random.randn(2, 2, 3).astype(np.float32)
net = Net()
out = net(Tensor(x))
```

The result is as follows:

```text
Nested execution during JIT execution for 'UserDefinedNet.func' is not supported when 'Net.construct' compile and execute.
```

It is recommended to remove the @jit decorator in the current scene.

2) write the code in the construct function of the Cell so that the code in the construct function will be compiled into a static computation graph.

```python
from mindspore import context, Tensor, jit, nn
import numpy as np
context.set_context(mode=context.GRAPH_MODE)

class InnerNet(nn.Cell):
    def __init__(self):
        super(InnerNet, self).__init__()

    def construct(self, x):
        return x

class UserDefinedNet: # Customized Python classes
    def __init__(self):
        self.value = 10
        self.inner_net = InnerNet()

    def func(self, x):
        return 2 * x * self.inner_net(x) + self.value

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.net = UserDefinedNet()

    def construct(self, x):
        x = self.net.value + self.net.func(x)
        return x

x = np.random.randn(2, 2, 3).astype(np.float32)
net = Net()
out = net(Tensor(x))
```

The result is as follows:

```text
Nested execution during JIT execution for 'InnerNet.construct' is not supported when 'Net.construct' compile and execute.
```

It is recommended to change it to the following code:

```python
from mindspore import context, Tensor, jit, nn
import numpy as np
context.set_context(mode=context.GRAPH_MODE)

class InnerNet(nn.Cell):
    def __init__(self):
        super(InnerNet, self).__init__()

    def construct(self, x):
        return x

class UserDefinedNet: # Customized Python classes
    def __init__(self):
        self.value = 10

    def func(self, x, y):
        return 2 * x * y + self.value

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.net = UserDefinedNet()
        self.inner_net = InnerNet()

    def construct(self, x):
        y = self.inner_net(x)
        x = self.net.value + self.net.func(x, y)
        return x

x = np.random.randn(2, 2, 3).astype(np.float32)
net = Net()
out = net(Tensor(x))
```

3) If a function decorated with a @jit decorator is called in a custom class, an error will be reported. In this scenario, it is recommended to add @jit_class decorators to custom classes in the network and avoid the JIT Fallback feature. For more use of custom classes, please refer to [Supporting the Use of Custom Classes](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/compile/static_graph.html#supporting-the-use-of-custom-classes). The use of jit_class decorators can be referred to [Use jit_class](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/compile/static_graph_expert_programming.html#using-jit-class).

```python
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE)

class InnerNet(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.value = 10

    @ms.jit
    def construct(self, x):
        return self.value + x

class CustomNet():
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, x):
        return self.model(2 * x)

class OutNet(ms.nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(self, x):
        return self.net(x)

x = ms.Tensor(2)
call_net = InnerNet()
custom_net = CustomNet(call_net)
out_net = OutNet(custom_net)
out = out_net(x)
print("out:", out)
```

The result is as follows:

```text
Nested execution during JIT execution for 'InnerNet.construct' is not supported when 'OuterNet.construct' compile and execute.
```

It is recommended to change it to the following code:

```python
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE)

class InnerNet(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.value = 10

    @ms.jit
    def construct(self, x):
        return self.value + x

@ms.jit_class
class CustomNet():
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __call__(self, x):
        return self.model(2 * x)

class OutNet(ms.nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(self, x):
        return self.net(x)

x = ms.Tensor(2)
call_net = InnerNet()
custom_net = CustomNet(call_net)
out_net = OutNet(custom_net)
out = out_net(x)
print("out:", out)
```

<br/>

## Q: What can I do if an error `ValueError: The value Parameter (name=name_a, shape=(1,), dtype=Float32, requires_grad=True) , its name 'name_a' already exists. Please set a unique name for the parameter.` is reported? What does it mean?

A: The graph mode requires the name of the parameter to be unique. If there are two or more Parameters with the same name, the network cannot distinguish different objects, which will cause errors. We can troubleshoot the Parameters with the same name in the script from the following angles, and set a unique name for the Parameter in it.

```python
import mindspore as ms
from mindspore.nn import Cell
from mindspore import Tensor, context, ParameterTuple, Parameter

context.set_context(mode=context.GRAPH_MODE)


class ParamNet(Cell):
    def __init__(self):
        super(ParamNet, self).__init__()
        self.res1 = ParameterTuple((Parameter(Tensor([2], ms.float32), name="name_a"),
                                    Parameter(Tensor([4], ms.float32), name="name_a")))
        self.param_tuple = (Parameter(Tensor([1], ms.float32), name="name_b"),
                            Parameter(Tensor([2], ms.float32)))
        self.param_list = [Parameter(Tensor([3], ms.float32), name="name_b"),
                           Parameter(Tensor([4], ms.float32))]

    def construct(self):
        out1 = self.res1[0] + self.res1[1]
        out2 = self.param_tuple[0] + self.param_tuple[1] + self.param_list[0] + self.param_listp[1]
        return out1, out2


net = ParamNet()
res = net()
```

As in the above script, ParameterTuple defines two Parameters with the same name name_a, which are not allowed. Parameters with the same name name_b defined in param_tuple and param_list are also not allowed. In another case, if a network is instantiated in the same cell in the script, such as the following example, the error "its name 'name_a' already exists." is reported.

```python
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context, ParameterTuple, Parameter


context.set_context(mode=context.GRAPH_MODE)


class InnerNet(nn.Cell):
    def __init__(self):
        super(InnerNet, self).__init__()
        self.param = Parameter(Tensor([1], ms.float32), name="name_a")

    def construct(self, x):
        return x + self.param


class OutNet1(nn.Cell):
    def __init__(self, net1, net2):
        super(OutNet1, self).__init__()
        self.param1 = ParameterTuple(net1.get_parameters())
        self.param2 = ParameterTuple(net2.get_parameters())

    def construct(self, x):
        return x + self.param1[0] + self.param2[0]


net1 = InnerNet()
net2 = InnerNet()
out_net = OutNet1(net1, net2)
res = out_net(Tensor([1], ms.float32))
print("res:", res)
```

For this case, we can use CellList to manage multiple instances of the same network.

```python
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context, ParameterTuple, Parameter


context.set_context(mode=context.GRAPH_MODE)


class InnerNet(nn.Cell):
    def __init__(self):
        super(InnerNet, self).__init__()
        self.param = Parameter(Tensor([1], ms.float32), name="name_a")

    def construct(self, x):
        return x + self.param


class OutNet1(nn.Cell):
    def __init__(self, net1, net2):
        super(OutNet1, self).__init__()
        self.cell_list = nn.CellList()
        self.cell_list.append(net1)
        self.cell_list.append(net2)
        self.param1 = ParameterTuple(self.cell_list[0].get_parameters())
        self.param2 = ParameterTuple(self.cell_list[1].get_parameters())

    def construct(self, x):
        return x + self.param1[0] + self.param2[0]


net1 = InnerNet()
net2 = InnerNet()
out_net = OutNet1(net1, net2)
res = out_net(Tensor([1], ms.float32))
print("res:", res)
```

In the following scenario, the bias parameter that has been defined in the network Net and the parameter with the same name are defined again in the network instantiation, which is not allowed in graph mode.

```python
import mindspore as ms
ms.set_context(mode=context.GRAPH_MODE)

class Net(ms.nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.bias = ms.Parameter(ms.Tensor(2.), name='bias')

class SubNet(ms.nn.Cell):
    def __init__(self):
        super().__init__()

sub = SubNet()
net = Net(sub)

net.net.bias = ms.Parameter(ms.Tensor(2.), name='bias')
a = ms.Tensor(3.)
grad_fn = ms.value_and_grad(net, None, net.trainable_params())
print(grad_fn(a))
```

<br/>

## Q: When calling the same network multiple times, under what circumstances will it be recompiled?

A: The following scenarios will trigger recompilation:

- The shape of Tensor changes.

- The scalar value changes.

- The length of Tuple or List changes.

- When the input of network is tuple[Tensor], list[Tensor] or Dict[Tensor], even if the shape and dtype of the Tensor inside do not change. For more details, please refer to [mutable](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/mindspore/mindspore.mutable.html).

<br/>

## Q: How to determine how many graphs there are in static graph mode? When will the subgraph be divided? What is the impact of multiple subgraphs? How to avoid multiple subgraphs?

A: 1. The number of subgraphs can be obtained by viewing the IR file and searching for "Total subgraphs". For how to view and analyze IR files, please refer to [MindSpore IR Introduction](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/debug/error_analysis/mindir.html)

2. Subgraph segmentation in static graph mode is common in control flow scenarios, such as if/while. In addition to manual writing by users, the control flow syntax within the MindSpore may also lead to dividing into multiple subgraphs.

3. Multiple subgraphs may affect network execution performance.

4. In order to avoid multiple subgraphs, try to avoid if/while conditions that rely on Tensor calculation results.

<br/>
