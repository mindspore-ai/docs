# Network Compilation

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/faq/network_compilation.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

<font size=3>**Q: What can I do if an error "'self.xx' should be defined in the class '__init__' function." is reported?**</font>

A: If you want to assign for a class member such as `self.xx` in the function `construct`, `self.xx` must have been defined to a [Parameter](<https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Parameter.html>) type firstly while the other types are not supported. But the local variable `xx` is not under the regulation.

<br/>

<font size=3>**Q: What can I do if an error "This comparator 'AnyValue' is not supported. For statement 'is', only support compare with 'None', 'False' or 'True'" is reported?**</font>

A: For the syntax `is` or `is not`, currently `MindSpore` only supports comparisons with `True`, `False` and `None`. Other types, such as strings, are not supported.

<br/>

<font size=3>**Q: What can I do if an error "MindSpore does not support comparison with operators more than one now, ops size =2" is reported?**</font>

A: For comparison statements, `MindSpore` supports at most one operator. For example, you can use `1 < x and x < 3` to take the place of `1 < x < 3`.

<br/>

<font size=3>**Q: What can I do if an error "TypeError: The function construct need 1 positional argument and 0 default argument, but provided 2" is reported?**</font>

A: When you call the instance of a network, the function `construct` will be executed. And the program will check the number of parameters required by the function `construct` and the number of parameters actually given. If they are not equal, the above exception will be thrown.
Please check that the number of parameters passed in when the instance of the network in the script is called matches the number of parameters required by the `construct` function in the defined network.

<br/>

<font size=3>**Q: What can I do if an error "Type Join Failed" or "Shape Join Failed" is reported?**</font>

A: In the inference stage of front-end compilation, the abstract types of nodes, including `type` and `shape`, will be inferred. Common abstract types include `AbstractScalar`, `AbstractTensor`, `AbstractFunction`, `AbstractTuple`, `AbstractList`, etc. In some scenarios, such as multi-branch scenarios, the abstract types of the return values of different branches will be `join` to infer the abstract type of the returned result. If these abstract types do not match, or `type`/`shape` are inconsistent, the above exception will be thrown.

When an error similar to "Type Join Failed: dtype1 = Float32, dtype2 = Float16" appears, it means that the data types are inconsistent, resulting in an exception when joining abstract. According to the provided data types and code line, the error can be quickly located. In addition, the specific abstract information and node information are provided in the error message. You can view the MindIR information through the `analyze_fail.dat` file to locate and solve the problem. For specific introduction of MindIR, please refer to [MindSpore IR (MindIR)](https://www.mindspore.cn/docs/en/master/design/mindir.html). The code sample is as follows:

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn, Tensor, context

context.set_context(mode=context.GRAPH_MODE)
class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = ops.ReLU()
        self.cast = ops.Cast()

    def construct(self, x, a, b):
        if a > b:    # The type of the two branches has inconsistent return values.
            return self.relu(x)    # shape: (2, 3, 4, 5), dtype:Float32
        else:
            return self.cast(self.relu(x), ms.float16)    # shape:(), dype: Float32

input_x = Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
input_a = Tensor(2, ms.float32)
input_b = Tensor(6, ms.float32)
net = Net()
out_me = net(input_x, input_a, input_b)
```

The result is as follows:

```text
TypeError: Cannot join the return values of different branches, perhaps you need to make them equal.
Type Join Failed: dtype1 = Float32, dtype2 = Float16.
For more details, please refer to the FAQ at https://www.mindspore.cn
The abstract type of the return value of the current branch is AbstractTensor(shape: (2, 3, 4, 5), element: AbstractScalar(Type: Float16, Value: AnyValue, Shape: NoShape), value_ptr: 0x55b9f289d090, value: AnyValue), and that of the previous branch is AbstractTensor(shape: (2, 3, 4, 5), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x55b9f289d090, value: AnyValue).
The node is construct.6:[CNode]13{[0]: construct.6:[CNode]12{[0]: ValueNode<Primitive> Switch, [1]: [CNode]11, [2]: ValueNode<FuncGraph> ✓construct.4, [3]: ValueNode<FuncGraph> ✗construct.5}}, true branch: ✓construct.4, false branch: ✗construct.5
The function call stack:
In file test.py(14)/        if a > b:

The function call stack (See file 'analyze_fail.dat' for more details):
# 0 In file test.py(14)
        if a > b:
        ^
```

When an error similar to "Shape Join Failed: shape1 = (2, 3, 4, 5), shape2 = ()" appears, it means that the `shape` are inconsistent, resulting in an exception when joining abstract. The code sample is as follows:

```python
import mindspore.ops as ops
from mindspore import Tensor, ms_function

x = Tensor([1.0])
y = Tensor([2.0])
grad = ops.GradOperation(get_by_list=False, sens_param=True)
sens = 1.0

def test_net(a, b):
    return a, b

@ms_function()
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
For more details, please refer to the FAQ at https://www.mindspore.cn.
This: AbstractTensor(shape: (1), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x56458a351ad0, value: Tensor(shape=[1], dtype=Float32, value=[ 1.00000000e+00])), other: AbstractTuple{element[0]: AbstractTensor(shape: (1), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x564583e3fa90, value: Tensor(shape=[1], dtype=Float32, value=[ 1.00000000e+00])), element[1]: AbstractTensor(shape: (1), element: AbstractScalar(Type: Float32, Value: AnyValue, Shape: NoShape), value_ptr: 0x564583cb00b0, value: Tensor(shape=[1], dtype=Float32, value=[ 2.00000000e+00])), sequence_nodes: {test_net.3:[CNode]4{[0]: ValueNode<PrimitivePy> MakeTuple, [1]: a, [2]: b}, elements_use_flags: {ptr: 0x5645cbc500c0, value: [const vector][1, 1]}}}
The function call stack (See file 'analyze_fail.dat' for more details):
# 0 In file test.py(16)
    a = grad(test_net)(x, y, sens_i)
        ^
```

<br/>

<font size=3>**Q: What can I do if an error "The params of function 'bprop' of Primitive or Cell requires the forward inputs as well as the 'out' and 'dout" is reported during compilation?**</font>

A: The inputs of user-defined back propagation function `bprop` should contain all the inputs of the forward network, `out` and `dout`. The example is as follow:

```python
class BpropUserDefinedNet(nn.Cell):
        def __init__(self):
            super(BpropUserDefinedNet, self).__init__()
            self.zeros_like = P.ZerosLike()

        def construct(self, x, y):
            return x + y

        def bprop(self, x, y, out, dout):
            return self.zeros_like(out), self.zeros_like(out)
```

<br/>

<font size=3>**Q: What can I do if an error "There isn't any branch that can be evaluated" is reported during compilation?**</font>

A: When an error similar to "There isn't any branch that can be evaluated" appears,
it means that there may be infinite recursion or loop in the code, which causes each branch of the if condition to be unable to deduce the correct type and dimension information.

The example is as follow:

```python
from mindspore import Tensor, ms_function
from mindspore import dtype as mstype
import mindspore.context as context
ZERO = Tensor([0], mstype.int32)
ONE = Tensor([1], mstype.int32)
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
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([5], mstype.int32)
    f(x)

```

The f(x) fails because each if branch cannot derive the correct type information.

<br/>

<font size=3>**Q: What can I do if an error "Exceed function call depth limit 1000" is reported during compilation?**</font>

When Exceed function call depth limit 1000 is displayed, this indicates that there is an infinite recursive loop in the code, or the code is too complex. The type derivation process causes the stack depth to exceed the set maximum depth.

At this time, you can set context.set_context(max_call_depth = value) to change the maximum depth of the stack, and consider simplifying the code logic or checking whether there is infinite recursion or loop in the code.

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

A: JIT Fallback is to realize the unification of static graph mode and dynamic graph mode from the perspective of static graph, so that the static graph mode can support the syntax of the dynamic mode as much as possible. It draws on the fallback idea of traditional JIT compilation. When compiling a static graph, if the syntax is not supported, the relevant sentence will be recorded and an interpret node will be generated. In the subsequent processing, the relevant sentence will be fallbacked to the Python interpreter for interpretation and execution, so that the syntax can be supported. The environment variable switch of JIT Fallback is `DEV_ENV_ENABLE_FALLBACK`, and JIT Fallback is enabled by default.

When the errors "Should not use Python object in runtime" and "We suppose all nodes generated by JIT Fallback would not return to outside of graph" appear, it means that there is an incorrect syntax in the code. The generated interpret node cannot be executed normally during the compilation phase, resulting in an error. The current JIT Fallback conditionally supports some constant scenarios in Graph mode, and it also needs to conform to MindSpore's programming syntax. When you write the code, please refer to [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html).

For example, when calling the third-party library NumPy, JIT Fallback supports the syntax of `np.add(x, y)` and `Tensor(np.add(x, y))`, but MindSpore does not support returning the NumPy type. Therefore, the program will report an error. The code sample is as follows:

```python
import numpy as np
import mindspore.nn as nn
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)

class Net(nn.Cell):
    def construct(self, x, y):
        out = np.add(x, y)
        return out

net = Net()
out = net(1, 1)
```

The result is as follows:

```text
RuntimeError: mindspore/ccsrc/pipeline/jit/validator.cc:139 ValidateValueNode] Should not use Python object in runtime, node: ValueNode<InterpretedObject> InterpretedObject: '2'

We suppose all nodes generated by JIT Fallback not return to outside of graph.

# In file test.py(9)
        out = np.add(x, y)
        ^
```

When there is an error related to JIT Fallback, please review the code syntax and modify it according to [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html) and the provided code line. If you need to turn off JIT Fallback, you can use `export DEV_ENV_ENABLE_FALLBACK=0`.

<font size=3>**Q: What can I do if an error  "Operator[AddN]  input(kNumberTypeBool,kNumberTypeBool) output(kNumberTypeBool) is not support. This error means the current input type is not supported, please refer to the MindSpore doc for supported types."**</font>

A: Currently, Tensor [subsequent abbreviation Tensor (bool)] with bool data type has weak support by MindSpore, and only a small number of operators support Tensor(bool) type data participation operations.  If an operator supporting the Tensor(bool) type is used in a forward graph and the forward graph syntax is correct, since the reverse graph solves the full derivative introduces `AddN`, `AddN` does not support the Tensor (bool) type, and the reverse graph run will throw the exception.

The example is as follow:

```python
from mindspore import context, ops, ms_function, Tensor, dtype

context.set_context(save_graphs=True, save_graphs_path='graph_path')

@ms_function
def test_logic(x, y):
    z = x and y
    return z and x

x = Tensor(True, dtype.bool_)
y = Tensor(True, dtype.bool_)
grad = ops.GradOperation(get_all=True)
grad_net = grad(test_logic)
out = grad_net(x, y)
```

The forward processing of the above code can be expressed as: the corresponding full derivative formula of `r = f(z, x), z = z(x, y)` is: `dr/dx = df/dz * dz/dx + df/dx`.  Function`f(z,x)` and `z(x,y)` are primitive `and`. Primitive `and` in the forward graph supports Tensor (bool) type, and the `AddN` introduced when reversing the full derivative of the graph does not support the Tensor(bool) type.  And the error cannot be mapped to a specific forward code line.

The result is as follows:

```text
Traceback (most recent call last):
  File "grad_fail.py", line 14, in <module>
    out = grad_net(x, y)
  File "/usr/local/python3.7/lib/python3.7/site-packages/mindspore/common/api.py", line 307, in staging_specialize
    out = _MindsporeFunctionExecutor(func, ms_create_time, input_signature, process_obj)(*args)
  File "/usr/local/python3.7/lib/python3.7/site-packages/mindspore/common/api.py", line 79, in wrapper
    results = fn(*arg, **kwargs)
  File "/usr/local/python3.7/lib/python3.7/site-packages/mindspore/common/api.py", line 221, in __call__
    phase = self.compile(args_list, arg_names, parse_method)
  File "/usr/local/python3.7/lib/python3.7/site-packages/mindspore/common/api.py", line 195, in compile
    self.enable_tuple_broaden)
TypeError: mindspore/ccsrc/runtime/device/cpu/kernel_select_cpu.cc:235 KernelNotSupportException] Operator[AddN]  input(kNumberTypeBool,kNumberTypeBool) output(kNumberTypeBool) is not support. This error means the current input type is not supported, please refer to the MindSpore doc for supported types.
Trace:
In file /usr/local/python3.7/lib/python3.7/site-packages/mindspore/ops/composite/multitype_ops/add_impl.py(287)/    return F.addn((x, y))/
```

If you encounter problems like this one, please remove the use of tensor (bool). In this example, replacing Tensor (bool) with bool can solve the problem.

<br/>

<font size=3>**Q: What can I do if an error "Side Effect Invalid: found unsupported syntax in graph mode, those side effect codes would be ignored:" is reported?**</font>

A: If you use a side-effect operator in `Cell.construct`,  `ms_function` or their callee function, you should ensure that the function not just return a constant value or inferred constant value. Since only the behavior of returning constant value will be preserved during compiling, the other operations seems invalid. Especially for side-effect operators, the result looks not correct if they're ignored, and may not correspond with user's expectation. So the compiler will throw an exception for this situation.

The example is as follow:

```python
from mindspore.nn import Cell

class Demo(Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        print('print here...')
        y = x[1]
        y[1] = 9
        return y

x = [[1, 2, 3, 4], [5, 6, 7, 8]]
net = Demo()
output = net(x)
print(output)
```

The variable `y` of the above code can be inferred as a constant value, so the function would be optimized as just return a constant value, and not really to execute `print('print here...')`. The operator `print` is an IO side effect operator, it's incorrect if it's ignored. In this case, the compiler will throw an exception to user.

The result is as follows:

```text
Traceback (most recent call last):
  File "test_print_op.py", line 20, in <module>
    output = net(x)
  File "/usr/local/python3.7/lib/python3.7/site-packages/mindspore/nn/cell.py", line 586, in __call__
    out = self.compile_and_run(*args)
  File "/usr/local/python3.7/lib/python3.7/site-packages/mindspore/nn/cell.py", line 964, in compile_and_run
    self.compile(*inputs)
  File "/usr/local/python3.7/lib/python3.7/site-packages/mindspore/nn/cell.py", line 937, in compile
    _cell_graph_executor.compile(self, *inputs, phase=self.phase, auto_parallel_mode=self._auto_parallel_mode)
  File "/usr/local/python3.7/lib/python3.7/site-packages/mindspore/common/api.py", line 1086, in compile
    result = self._graph_executor.compile(obj, args_list, phase, self._use_vm_mode())
RuntimeError: mindspore/ccsrc/pipeline/jit/static_analysis/evaluator.cc:127 CheckSideEffectNodes] Side Effect Invalid: Found unsupported syntax in graph mode, those side effect codes would be ignored:
-----
# No. 1:
In file test_print_op.py(11)
         print('print here...')
         ^

-----

If a function return a const value or inferred const value, the side effect node would be ignored.
So the codes may not run as the user's expectation, please fix it.

In this case, the const value '[[1, 2, 3, 4], [5, 6, 7, 8]]' returns:
In file test_print_op.py(10)
     def construct(self, a):
     ^

For more information about this issue, please refer to https://www.mindspore.cn/search?inputValue=Side%20Effect%20Invalid
```

If you meet this situation, please remove the side-effect operators, or not just return a constant value in the function.

