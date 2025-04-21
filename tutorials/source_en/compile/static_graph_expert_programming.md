# Graph Mode - Programming Techniques

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_en/compile/static_graph_expert_programming.md)

This chapter introduces some commonly used advanced programming techniques for static graph optimization, which can effectively improve the compilation efficiency as well as the execution efficiency of static graphs, and make the program run more stably. For a basic introduction to static graphs compilation, see [Accelerating with Static Graphs](https://www.mindspore.cn/tutorials/en/br_base/beginner/accelerate_with_static_graph.html).

## How to Optimize Compilation Performance

### Using lazy_inline Decorator

The compilation process for neural network models often uses the default inline approach, which eventually unfolds the hierarchical code representation into a flat computational graph, seeking maximize compilation optimization, and simplifying the automatic differentiation as well as the logic of execution. The computational graph formed after inline contains all the computational nodes, which can be optimized in a larger scope, such as constant folding, node fusion, parallel analysis. It can also be better implemented for memory allocation, reducing memory requests and performance overhead. Although inline optimization helps much in runtime performance improvement, excessive inline also brings burden in compilation period. For example, as the number of computational graph nodes swells, the time consumption for executing pass grows dramatically.

In order to mitigate the loss of compilation performance caused by inline, we provide a Lazy Inline mechanism to reduce compilation time for scenarios where the same computation unit is called repeatedly (typically, different instances of the same Cell class are called in a for loop).

#### Large Model Pipeline Parallel Scenarios

In the large model scenario, the compilation time consumption problem is especially prominent. One is that the model structure of the large model has a deep hierarchy and a large number of nodes; the second is that when the large model is trained, the model size and the number of nodes are further increased due to enabling pipeline parallel. If the original graph size is O, then the pipeline parallel is turned on, and the size of the single node graph becomes (O/X)*Y where X is the the number of pipeline stages, and Y is the number of micro batch. Taking the Pangu 13B network as an example, the number of computational nodes in the computational graph reaches 135,000, and the duration of a single compilation can be close to 3 hours.

The large model network structure similar to Pangu is composed of multiple layers, and when pipeline parallel is turned on, the layer structure of each micro batch is exactly the same. When pipeline parallel is turned on, `PipelineCell` uses a for loop to call the same structure of layers multiple times, as shown in the code below:

```python
from mindspore import nn

class PipelineCell(nn.Cell):
    def __init__(self, network, micro_size):
        ...
        self.network = network
        self.micro_size = micro_size
        ...

    def construct(self, ...):
        ...
        for i in range(self.micro_size):
            output = self.network(...)
        ...
```

If we think of the loop body as a subgraph that is called frequently, and tell the compiler to defer inline processing by marking it as Lazy Inline, then we can achieve performance gains by drastically reducing the number of computational graph nodes during most phases of compilation. For example, the code above can preserve the subgraph structure of the `network` instance without inlining or without early inline, for which we provide the `@lazy_inline` decorator to implement delayed inlining.

Taking the Pangu_alpha network as an example, the `network` handled in the `PipelineCell` function body is an instance of the `PanGUAlphaWithLoss` class. In order to implement a delayed inline, a `@ lazy_inline` decorator is added to the `__init__` function of the `PanGUAlphaWithLoss` class to mark that the subgraph structure of the `PanGUAlphaWithLoss` class needs to be preserved without inlining or with delayed inlining. as shown below:

```python
from mindspore import nn
from mindspore import lazy_inline

class PanGUAlphaWithLoss(nn.Cell):
    @lazy_inline
    def __init__(self, ...):
        ...

    def construct(self, ...):
```

> The full code can be found at: [Pangu_alpha](https://gitee.com/mindspore/models/tree/master/official/nlp/Pangu_alpha)

Still taking the Pangu 13B network as an example, after applying the Lazy Inline scheme, the compute graph compilation size drops from 130,000+ nodes to 20,000+ nodes, and the compilation time drops from 3 hours to 20 minutes.

#### More General Scenarios

`@lazy_inline` is a decorator for `Cell::__init__`, which generates the attribute value of the Cell `cell_init_args` with all the parameters of `__init__`, and the same value of `cell_init_args` indicates that the Cell class name and the values of the initialization parameters are the same. And for instances of the same Cell class, their weights may also be different, so for a network structure defined with `construct(self, x)`, at actual compile time we can convert to `construct(x, self.cell_init_args, self.trainable_ parameters())`. For different instances of the same Cell class, if `cell_init_args` is the same, both instances can reuse the same network structure as follows:

```python
def construct(self, x)
    reuse_construct(x, self.trainable_parameters())
```

With the introduction of reusable computation graphs, Cell instances with the same `cell_init_args` only need to be compiled and resolved once. So for more generalized scenarios of calling different instances of the same Cell class, as long as the `cell_init_args` are the same, we can add the `@lazy_inline` decorator to speed up compilation. For example, GPT networks:

```python
from mindspore import nn
from mindspore import lazy_inline

class Block(nn.Cell):
    @lazy_inline
    def __init__(self, config):
        ...

    def construct(self, x, attention_mask, layer_past):
        ...

class GPT_Model(nn.Cell):
    def __init__(self, config):
        ...
        for i in range(config.num_layers):
            self.blocks.append(Block(config))
            ...
        self.num_layers = config.num_layers

    def construct(self, input_ids, input_mask, layer_past):
        ...
        present_layer = ()
        for i in range(self.num_layers):
            hidden_states, present = self.blocks[i](...)
            present_layer = present_layer + (present,)
        ...
```

> The full code can be found at: [GPT](https://gitee.com/mindspore/models/tree/master/official/nlp/GPT)

The network structure of GPT consists of different instances of the multi-layer `Block` class, which are all initialized with the same `config` parameter, so with the addition of the `@lazy_inline` decorator, all of these `Block` instances can reuse the same network structure and are not inlined for most of the compilation phase, which can drastically reduce compilation time.

#### Usage Steps

As in the example above, add the `@lazy_inline` decorator to the `__init__` function of the Cell class that needs to delay the inline and reuse the subgraph structure in the network script.

#### Usage Limitations

1. Cell generates Cell instance identifiers based on the class name of the Cell and the value of the `__init__` parameter. This is based on the assumption that the `__init__` parameter determines all the attributes of the Cell, and that the Cell attributes at the start of the `construct` composition are the same as the attributes at the end of the `__init__` execution, therefore the composition-dependent attributes of Cell cannot be changed after `__init__` is executed. For example:

   ```python
   from mindspore import nn
   from mindspore import lazy_inline

   class Block(nn.Cell):
       @lazy_inline
       def __init__(self, ...):
           self.x = 0
           ...

       def construct(self, ...):
           if self.x == 0:
               ...
           else:
               ...
           ...

   class Model(nn.Cell):
       def __init__(self, ...):
           ...
           self.num_layers = 10
           for i in range(self.num_layers):
               self.blocks.append(Block(...)) # Here Block is initialized
               ...
           self.blocks[0].x = 1               # Here, modifying Block attributes after Block has been initialized will result in the Block not being able to reuse the same copy of the subgraph

       def construct(self, ...):
           ...
           for i in range(self.num_layers):
               res = self.blocks[i](...)
           ...
   ```

   As shown in the code above, an instance of `Block` in the network Model, whose attribute `x` has been modified after the initialization of that instance, will not be able to accurately reuse the same subgraph structure for that `Block` instance.

2. In a scenario where the network structure of a Cell class contains multiple instances of the Cell_X class, and the network structure of each Cell_X class contains multiple instances of the Cell_Y class, if you add `@lazy_inline` to the `__init__` functions of both the Cell_X and Cell_Y classes, only the outermost Cell_X instances will be compiled into a reusable computation graph and delayed inline. The computation graph of the inner Cell_Y instance will still be inline. e.g.:

   ```python
   from mindspore import nn
   from mindspore import lazy_inline

   class InnerBlock(nn.Cell):
       @lazy_inline             # InnerBlock does not get delayed inline
       def __init__(self, ...):
           ...

       def construct(self, ...):
           ...

   class OuterBlock(nn.Cell):
       @lazy_inline             # OuterBlock will be delayed inline.
       def __init__(self, ...):
           ...
           self.num_layers = 10
           for i in range(self.num_layers):
               self.blocks.append(InnerBlock(...))

       def construct(self, ...):
           ...
           for i in range(self.num_layers):
               res = self.blocks[i](...)
           ...

   class Model(nn.Cell):
       def __init__(self, ...):
           ...
           self.num_layers = 10
           for i in range(self.num_layers):
               self.blocks.append(OuterBlock(...))

       def construct(self, ...):
           ...
           for i in range(self.num_layers):
               res = self.blocks[i](...)
           ...
   ```

   There are subsequent plans to support this multi-layer Lazy Inline mechanism.

### Using HyperMap

Usage Scenario: Use HyperMap to replace for loop to optimize compilation performance.

`HyperMap` is a special class. Class object construction needs to be passed into the mapping function f, and calling the object needs to be passed into the n parameter sequence of f. For more usage see: [HyperMap](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.HyperMap.html). The mapping function f must be of type `MultitypeFuncGraph`, see [MultitypeFuncGraph](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.MultitypeFuncGraph.html). When using for loops to batch process list elements, network compilation performance can be optimized by `HyperMap`-equivalent semantic substitution.

### Using the Compilation Cache

Usage scenario: Compilation time is reduced by using a compilation cache if no changes are made to the files on which the compilation depends when training or inference is performed.

The essence of the compilation cache is to store the compilation intermediate process file of the network model. When the network model is unchanged, the production of the compilation intermediate process file is also the same, so you can reuse the intermediate process file produced by the last programming.

By setting the environment variable [MS_COMPILER_CACHE_ENABLE](https://www.mindspore.cn/docs/en/br_base/api_python/env_var_list.html?highlight=MS_COMPILER_CACHE_ENABLE), which can specify whether to save and load the compile cache.

By setting the environment variable [MS_COMPILER_CACHE_PATH](https://www.mindspore.cn/docs/en/br_base/api_python/env_var_list.html?highlight=MS_COMPILER_CACHE_PATH), you can specify the MindSpore compilation cache directory for storing cache files generated by the graph and operator compilation process.

A code sample that optimizes compilation performance by enabling compilation caching is shown below:

```python
import os
import time
from mindspore import dtype
import mindspore as ms

@ms.jit
def func(input_x, input_y):
    output = input_x
    for _ in range(200):
        output = input_x + input_x * input_y + output
    return output

os.environ['MS_COMPILER_CACHE_ENABLE'] = '0'
x = ms.Tensor([1], dtype.float32)
y = ms.Tensor([2], dtype.float32)
start_time = time.time()
out = func(x, y)
end_time = time.time()
print("Disable comile_cache cost time:", end_time - start_time)
```

The above test sample is to close the compilation cache state, execute the above test sample two times. The first time consumption and the second time consumption is as follows (the actual time consumption is related to the hardware environment, the following data is for reference only):

```text
Disable comile_cache cost time: 0.5485098361968994
Disable comile_cache cost time: 0.4614279270172119
```

It can be seen that when the compilation cache is turned off, the 2nd execution of the sample takes a little less time than the 1st. This is because the operator compilation cache is turned on by default and the 2nd execution of the sample is able to utilize the previous operator compilation cache.

```python
import os
import time
from mindspore import dtype
import mindspore as ms

@ms.jit
def func(input_x, input_y):
    output = input_x
    for _ in range(200):
        output = input_x + input_x * input_y + output
    return output

os.environ['MS_COMPILER_CACHE_ENABLE'] = '1'
os.environ['MS_COMPILER_CACHE_PATH'] = 'my_compile_cache'
x = ms.Tensor([1], dtype.float32)
y = ms.Tensor([2], dtype.float32)
start_time = time.time()
out = func(x, y)
end_time = time.time()
os.environ['MS_COMPILER_CACHE_ENABLE'] = '0'
print("Enable comile_cache cost time:", end_time - start_time)
```

The above test sample is to enable the compilation cache, execute the above test sample two times. The first time and the second time consumption is as follows (the actual time consumption is related to the hardware environment, and the following data is for reference only):

```text
Enable comile_cache cost time: 0.6357541084289551
Enable comile_cache cost time: 0.09379792213439941
```

As you can see, when compilation cache is enabled, the 2nd execution of the sample takes only about 1/7th of the time for the first execution.

Explanation: When the compile cache function is turned on, the first execution will not generate the cache yet, resulting in a warning.

```text
Warning: Check the consistency of dependency files hash failed. Execute all the compilation actions.
```

## How to Optimize Execution Performance

### Using jit_class

Usage scenario: Use `@jit_class` decorator to modify custom classes to improve execution performance. jit_class is applied to static graph mode. In dynamic graph mode, `@jit_class` is ignored and does not affect the execution logic of the dynamic graph mode.

#### Introduction to jit_class

When a user defines a class in a network script, it can be written as a class inherited from `Cell`, a custom class, or a class decorated by `@jit_class`, and their usage and differences are as follows:

- a class inherited from `Cell`

  Cell is the basic building block of a neural network in MindSpore, and models or neural network layers should inherit this class. In static graph mode, the `Cell` class is used and execution code is written in the `construct` function, which is compiled into a static computational graph.

- a custom class

  After defining a custom class, you can instantiate the class and call the attributes and methods of the class object. Please refer to [the Use of Custom Classes](https://www.mindspore.cn/tutorials/en/br_base/compile/static_graph.html#supporting-the-use-of-custom-classes). Compared to `Cell` class definitions, custom classes are closer to the user habits of calling Python classes. The implementation of custom classes in static graph mode is different from `Cell`, for example, when calling a function method of a custom class object, the code in its function method will not be compiled into a static computational graph but will be interpreted and executed by the Python interpreter.

- a class decorated by `@jit_class`

  The `@jit_class` decorator is provided in order to balance the user Python usage habits with the performance benefits of static graph compilation. After modifying the `@jit_class` decorator for a custom class, the function code of the class will be compiled into a static computational graph. Based on graph optimization, and static graph sinking, the compiler can globally optimize for the computational graph to obtain better execution performance.

In static graph mode, by modifying a custom class with `@jit_class`, the user can create, call instances of the class, and get its attributes and methods.

#### Using the jit_class Decorator

The jit_class decorator only supports modifying custom classes, not classes that inherit from `Cell`.

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class InnerNet:
    value = ms.Tensor(np.array([1, 2, 3]))

class Net(nn.Cell):
    def construct(self):
        return InnerNet().value

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
out = net()
print(out)
```

```text
[1 2 3]
```

If jit_class modifies a class that inherits from `Cell`, an error will be reported.

```python
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class Net(nn.Cell):
    def construct(self, x):
        return x

ms.set_context(mode=ms.GRAPH_MODE)
x = ms.Tensor(1)
net = Net()
net(x)
```

The error message is as follows:

```text
TypeError: Decorator jit_class is used for user-defined classes and cannot be used for nn.Cell: Net<>.
```

jit_class supports scenarios where custom classes are used nested, and custom classes are used nested with `Cell`. It should be noted that when class inherition occurs, if the parent class uses jit_class, the child class will also have the capabilities of jit_class.

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class Inner:
    def __init__(self):
        self.value = ms.Tensor(np.array([1, 2, 3]))

@ms.jit_class
class InnerNet:
    def __init__(self):
        self.inner = Inner()

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.inner_net = InnerNet()

    def construct(self):
        out = self.inner_net.inner.value
        return out

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
out = net()
print(out)
```

```text
[1 2 3]
```

#### Obtaining the Attributes and Methods of Classes

Supports calling attributes and methods by class name or class instance.

```python
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class InnerNet:
    def __init__(self, val):
        self.number = val

    def act(self, x, y):
        return self.number * (x + y)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.inner_net = InnerNet(2)

    def construct(self, x, y):
        return self.inner_net.number + self.inner_net.act(x, y)

ms.set_context(mode=ms.GRAPH_MODE)
x = ms.Tensor(2, dtype=ms.int32)
y = ms.Tensor(3, dtype=ms.int32)
net = Net()
out = net(x, y)
print(out)
```

```text
12
```

#### Creating Instances of Classes

For functions that will be compiled into static computational graphs, such as `Cell` `construct` function, `@jit`-modified functions, or subfunctions called by the first two, the parameters are required to be constants if it is necessary to create instances of the class modified by `@jit_class` within the function.

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class InnerNet:
    def __init__(self, val):
        self.number = val + 3

class Net(nn.Cell):
    def construct(self):
        net = InnerNet(2)
        return net.number

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
out = net()
print(out)
```

```text
5
```

#### Calling Instances of Classes

When calling an instance of a class modified by `@jit_class`, the `__call__` function method of that class is invoked.

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class InnerNet:
    def __init__(self, number):
        self.number = number

    def __call__(self, x, y):
        return self.number * (x + y)

class Net(nn.Cell):
    def construct(self, x, y):
        net = InnerNet(2)
        out = net(x, y)
        return out

ms.set_context(mode=ms.GRAPH_MODE)
x = ms.Tensor(2, dtype=ms.int32)
y = ms.Tensor(3, dtype=ms.int32)
net = Net()
out = net(x, y)
print(out)
```

```text
10
```

If the class does not define a `__call__` function, an error will be reported.

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class InnerNet:
    def __init__(self, number):
        self.number = number

class Net(nn.Cell):
    def construct(self, x, y):
        net = InnerNet(2)
        out = net(x, y)
        return out

ms.set_context(mode=ms.GRAPH_MODE)
x = ms.Tensor(2, dtype=ms.int32)
y = ms.Tensor(3, dtype=ms.int32)
net = Net()
out = net(x, y)
print(out)
```

The error message is as follows:

```text
ValueError: MsClassObject: 'InnerNet' has no __call__ function, please check the code.
```

### Using select Operator

Usage scenario: `Select` operator is used to replace if control flow statements, reducing static graph subgraph generation and improving execution performance (and also compilation performance).

When writing networks, you will often use if statements, if the condition of the if statement is a variable condition, each if statement will generate additional subgraphs. In static graph mode, the higher the number of subgraphs, the longer the compilation takes, so some scenarios can be optimized for compilation performance by replacing the if statement equivalently with the `Select` operator.

Note that using the `Select` operator to replace the if statement affects the performance of the network. On the one hand, the `Select` operator executes both the true branch and the false branch, whereas the if statement executes only one of its branches, so the time comsuption with if decreases compared to that with the `Select` operator; on the other hand, the `Select` operator outperforms the control-flow operator generated by the if statement, and the time comsuption with if increases compared to that with the `Select` operator. operator. Combining the above two factors, the final runtime performance changes need to be judged according to the actual situation. Generally speaking, when the number of operators in a branch is small, it is recommended to use `Select` operator, while when the number of operators in a branch is large, it is recommended to use if statement.

A code sample that uses the `Select` operator instead of an if statement to optimize compilation performance is shown below:

```python
import time
from mindspore import ops
import mindspore as ms

@ms.jit
def if_net(x, y):
    out = 0
    for _ in range(100):
        if x < y:
            x = x - y
        else:
            x = x + y
        out = out + x
    return out

start_time = time.time()
out = if_net(ms.Tensor([0]), ms.Tensor([1]))
end_time = time.time()
print("if net cost time:", end_time - start_time)

@ms.jit
def select_net(x, y):
    out = x
    for _ in range(100):
        cond = x < y
        x = ops.select(cond, x - y, x + y)
        out = out + x
    return out

start_time = time.time()
out = select_net(ms.Tensor([0]), ms.Tensor([1]))
end_time = time.time()
print("select net cost time:", end_time - start_time)
```

```text
if net cost time: 1.1603329181671143
select net cost time: 0.483151912689209
```

### Using Vmap for Batch Processing

Usage scenario: When processing batch data without dependency and the related operator supports Vmap function, you can use Vmap to replace for loop to process batch data to optimize the execution performance (and also improve the compilation performance).

MindSpore already supports the Vmap feature.

A code sample that uses Vmap to replace a for loop to process batch data to optimize compilation performance is shown below:

The running results of the above code are as follows (the actual time consumption is related to the hardware environment, and the following data is for reference only):

```python
import numpy as np
import time
from mindspore import ops, vmap
import mindspore as ms

def hswish_func(x):
    return ops.HSwish()(x)

@ms.jit
def manually_batched(xs):
    output = []
    for i in range(xs.shape[0]):
        output.append(hswish_func(xs[i]))
    return ops.stack(output)

shape = (100, 2)
prop = 100
x_np = (np.random.randn(*shape) * prop).astype(np.float32)
x = ms.Tensor(x_np)
x = ops.sub(x, 0)

start_time = time.time()
output_vmap = vmap(hswish_func, in_axes=(0,))(x)
end_time = time.time()
print("Vmap cost time:", end_time - start_time)

start_time = time.time()
output_manually = manually_batched(x)
end_time = time.time()
print("for loop cost time:", end_time - start_time)
```

```text
Vmap cost time: 0.05766916275024414
for loop cost time: 1.9284062385559082
```

The above sample corresponds to the need to batch process 100 sets of Tensor data, and it can be seen that the performance of processing with Vmap exceeds the performance of processing with for loops by a factor of 30.

## Dependency Control Guarantees Execution Order

We consider a function to have a side effect if the result of its operation depends on or affects external state, e.g., if the function changes external global variables, if the result of the function depends on the value of a global variable. We consider an operator with side effects if the operator changes the value of an input parameter or if the output of the operator depends on the value of a global parameter.

Side effects are categorized into memory side effects and IO side effects based on memory attributes and IO states. Currently memory side effects are mainly Assign, optimizer operator and so on, and IO side effects are mainly Print operator. For details, you can check the operator definitions. Memory side effect operators have side_effect_mem attribute in their definitions, and IO side effect operators have side_effect_io attribute in their definitions.

Depend is used to handle dependency operations. In most cases, if the operators have IO side effect or memory side effect, they will be executed according to the user semantics, and there is no need to additionally use the Depend operator to guarantee the execution order. In some cases, if two operators A and B have no sequential dependency and A must be executed before B, we recommend using Depend to specify their execution order. The usage is as follows:

```python
a = A(x)
b = B(y)
```

Inserting the Depend operator:

```python
a = A(x)
y = Depend(y, a)
b = B(y)
```

It is worth stating that the particular set of operators used for floating-point overflow state detection have implicit side effects, but are not IO side effects or memory side effects. In addition, there are strict order requirements for their use, i.e., you need to ensure that NPUAllocFloatStatus has been executed before using the NPUClearFloatStatus operator, and ensure that NPUClearFloatStatus has been executed before using the NPUGetFloatStatus operator. Because these operators are used less, the current scheme is to keep their definitions in a side-effect free form to ensure the order of execution with Depend. Note: the operators used for floating-point overflow state detection is only supported on the Ascend platform.

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, set_context, Tensor
from mindspore import dtype as mstype

set_context(mode=ms.GRAPH_MODE)
ms.set_device("Ascend")

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.alloc_status = ops.NPUAllocFloatStatus()
        self.get_status = ops.NPUGetFloatStatus()
        self.clear_status = ops.NPUClearFloatStatus()

    def construct(self, x):
        init = self.alloc_status()
        clear_status = self.clear_status(init)
        x = ops.Depend()(x, clear_status)
        res = ops.sub(x, ops.neg(x))
        init = ops.Depend()(init, res)
        get_status = self.get_status(init)
        res = ops.Depend()(res, get_status)
        return res

value = 5
data = np.full((2, 3), value, dtype=np.float16)
x = Tensor(data, dtype=mstype.float16)
net = Net()
res = net(x)
print(res)
```

```text
[[10. 10. 10.]
 [10. 10. 10.]]
```

## Optimizing Redundant Video Memory Copy Operations

In functional programming, a function that exchanges data with the outside world through channels other than parameters and return values is called a non-pure function and is considered to have side effects. Inside the MindSpore framework, the Load operator is inserted for side effects, which is a virtual operator that does not need to be executed in the backend, does not occupy video memory, and indicates the need to read the value of a global variable. In graph mode, you need to compile the whole graph before sending down each operator in the graph to the backend for execution, and use Load operator to read the global variables many times, instead of using real operator to save the value of global variables many times, which can reduce the consumption of video memory.

However, the value of the global variable may change, and if there is no real operator to save the value, there will be precision problems in some scenarios. In this case, the MindSpore framework inserts real operators that take up a certain amount of memory to save the values of global variables, thus avoiding precision problems.

We provide the MS_DEV_SIDE_EFFECT_LOAD_ELIM switch to optimize the level of video memory usage, i.e. set export MS_DEV_SIDE_EFFECT_LOAD_ELIM=0/1/2/3.

- When MS_DEV_SIDE_EFFECT_LOAD_ELIM is set to 0, it means that the real operator is inserted for all the Load operators inside the framework, i.e., it takes up the most video memory and ensures that there is no problem with network accuracy.
- When MS_DEV_SIDE_EFFECT_LOAD_ELIM is set to 1 or no value is set (i.e., default mode), it indicates that real operators are inserted conservatively for scenarios where the Load operator inside the framework may have accuracy problems and ensures that there is no problem with network accuracy.
- When MS_DEV_SIDE_EFFECT_LOAD_ELIM is set to 2, it inserts as few real operators as possible under the premise of losing a certain compilation performance, optimizes more memory, and ensures that there is no problem with network accuracy.
- When MS_DEV_SIDE_EFFECT_LOAD_ELIM is set to 3, no real operator is inserted. The accuracy of the network is not ensured, and the video memory is consumed the least.

We can further understand this through the use case and the generated intermediate representation (IR).

```python
import numpy as np
from mindspore.nn import Cell
from mindspore import Tensor, Parameter, ops
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE)

class ForwardNet(Cell):
    def __init__(self):
        super(ForwardNet, self).__init__()
        self.weight = Parameter(Tensor(np.array(0), ms.int32), name="param")

    def construct(self, x):
        out = 0
        i = 0
        while i < 3:
            ops.assign(self.weight, i)
            out = x * self.weight + out
            i = i + 1
        return out


class BackwardNet(Cell):
    def __init__(self, net):
        super(BackwardNet, self).__init__(auto_prefix=False)
        self.forward_net = net
        self.grad = ops.GradOperation(get_all=True)

    def construct(self, *inputs):
        grads = self.grad(self.forward_net)(*inputs)
        return grads

x = Tensor(np.array(1), ms.int32)
graph_forword_net = ForwardNet()
graph_backword_net = BackwardNet(graph_forword_net)
graph_mode_grads = graph_backword_net(x)
output_except = (Tensor(np.array(3), ms.int32),)
assert np.all(graph_mode_grads == output_except)
```

As in the above use case, save the intermediate file by setting the environment variable `MS_DEV_SAVE_GRAPHS` to 1, you can get the intermediate file IR, for easy viewing, we simplify the resulting intermediate file as follows:

The IR file when the real operator is not inserted into the Load operator inside the framework is as follows, and you can see that there are 3 Load operators, all of which take the value of para2_param this global variable at different times, and this global variable will modify the value through the Assign operator. That is, the values taken by the 3 Loads are different. And if we do not insert the real operator into the Load operator, that is, we do not save the value of the global variable para2_param at different times, then the final result obtained is incorrect. That is, this case is set to 3 in the MS_DEV_SIDE_EFFECT_LOAD_ELIM, which has the least memory footprint, but the result has precision problems.

```text
# IR entry: @BackwardNet_construct
# Total subgraphs: 1
# Total params: 2
# Params:
%para1_inputs0 : <Tensor[Int32], ()>
%para2_param : <Ref[Tensor[Int32]], (), ref_key=:param>  :  has_default

subgraph @BackwardNet_construct() {
  %0 = Assign(%para2_param, Tensor(shape=[], dtype=Int32, value=0), U)
  %1 = UpdateState(U, %0)
  %2 = Load(%para2_param, %1)
  %3 = UpdateState(%1, %2)
  %4 = Assign(%para2_param, Tensor(shape=[], dtype=Int32, value=1), %3)
  %5 = UpdateState(%3, %4)
  %6 = Load(%para2_param, %5)
  %7 = UpdateState(%5, %6)
  %8 = Assign(%para2_param, Tensor(shape=[], dtype=Int32, value=2), %7)
  %9 = UpdateState(%7, %8)
  %10 = Load(%para2_param, %9)
  %11 = MakeTuple(%10, %6)
  %12 = AddN(%11)
  %13 = MakeTuple(%12, %2)
  %14 = AddN(%13)
  %15 = MakeTuple(%14)
  %16 = UpdateState(%9, %10)
  %17 = Depend(%15, %16)
  Return(%17)
}
```

When the MS_DEV_SIDE_EFFECT_LOAD_ELIM is set to 0, 1, and 2, the simplified IR figure is shown below. Since the Load operators in this scenario need to insert real operators to save the values modified by each Assign operator, the IR files obtained MS_DEV_SIDE_EFFECT_LOAD_ELIM set to 0, 1, and 2 are consistent. In more complex cases, the MS_DEV_SIDE_EFFECT_LOAD_ELIM may be different when set to 0, 1, 2, and will not be expanded here.

```text
# IR entry: @BackwardNet_construct
# Total subgraphs: 1
# Total params: 2
# Params:
%para1_inputs0 : <Tensor[Int32], ()>
%para2_param : <Ref[Tensor[Int32]], (), ref_key=:param>  :  has_default

subgraph @BackwardNet_construct() {
  %0 = Assign(%para2_param, Tensor(shape=[], dtype=Int32, value=0), U)
  %1 = UpdateState(U, %0)
  %2 = Load(%para2_param, %1)
  %3 = TensorMove(%2)
  %4 = UpdateState(%1, %3)
  %5 = Assign(%para2_param, Tensor(shape=[], dtype=Int32, value=1), %4)
  %6 = UpdateState(%4, %5)
  %7 = Load(%para2_param, %6)
  %8 = TensorMove(%7)
  %9 = UpdateState(%6, %8)
  %10 = Assign(%para2_param, Tensor(shape=[], dtype=Int32, value=2), %9)
  %11 = UpdateState(%9, %10)
  %12 = Load(%para2_param, %11)
  %13 = TensorMove(%12)
  %14 = MakeTuple(%13, %8)
  %15 = AddN(%14)
  %16 = MakeTuple(%15, %3)
  %17 = AddN(%16)
  %18 = MakeTuple(%17)
  %19 = UpdateState(%11, %13, %15)
  %20 = Depend(%18, %19)
  Return(%20)
}
```
