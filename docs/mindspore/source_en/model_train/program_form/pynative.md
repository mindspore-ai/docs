# Dynamic Graph

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/model_train/program_form/pynative.md)

## Basic Capability

In MindSpore, dynamic graph mode is also known as PyNative mode, which is default mode, and can also be set to dynamic graph mode by `set_context(mode=PYNATIVE_MODE)`. In script development and network flow debugging, debugging is convenient in dynamic graph mode, and the dynamic graph mode supports the execution of single operators, common functions and networks, and separate gradient solving operations.

In PyNative mode, users can use the full Python API. In addition, for using the API provided by MindSpore, the framework will execute the operations of the operator API on the corresponding hardware platform according to the hardware platform (Ascend, GPU, CPU) selected by the user and return the corresponding results. The overall execution process of the framework is as follows:

![process](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/design/images/framework.png)

Through the front-end Python API, call to the framework layer, and finally to the corresponding hardware devices to perform calculations. For example, to complete an addition

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE)
ms.set_device(device_target="CPU")
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

## Dynamic and Static Combination

### JIT

Python has become the mainstream language for programming in AI due to its dynamic language and its flexible and efficient development capabilities. However, due to Python interpreted execution characteristics, its execution performance is often not optimal. Dynamic graph patterns fit Python interpreted execution characteristics, but it is difficult to further optimize the execution performance by means of operator fusion optimization and other means. Therefore, MindSpore provides JIT (just-in-time) technology to further optimize the performance. The JIT mode parses the code into an intermediate representation (IR) graph by means of AST tree parsing, Python bytecode parsing or code execution tracing, which serves as a unique representation of the code, and the compiler optimizes the code by optimizing the IR graph to improve the runtime performance. In contrast to the dynamic graph model, this JIT compilation model is called the static graph model.

Based on JIT technology, MindSpore provides a dynamic-static combination approach to improve the operational efficiency of the user's network. The combination of dynamic and static, that is, in the overall run as a dynamic graph, specifies certain code blocks to run as a static graph. Code blocks that run as static graphs are compiled first and then executed, and global optimizations are performed during the compilation period to obtain performance gains during the execution period. Users can modify functions with the `@jit` decorator to specify that they execute according to the pattern of a static graph. For the documentation on the `@jit` decorator, refer to [jit API documentation](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.jit.html#mindspore.jit). Additionally, users can configure the functions that execute static graph processes via `jit_config`, see [mindspore.JitConfig](https://mindspore.cn/docs/en/master/api_python/mindspore/mindspore.JitConfig.html#mindspore.JitConfig).

MindSpore provides three JIT compilation methods, namely, ast, bytecode and trace. The ast converts the functions that are identified by the users manually and need to be executed in accordance with the ast into a static graph through the AST tree parsing. If encountered inability to convert to a static graph, it will be directly reported as an error. The bytecode is through the Python bytecode parsing, in the dynamic graph as much as possible to build a static graph. The part that can not be converted to a static graph will be in accordance with the dynamic graph for the purpose of combining static and dynamic. The trace constructs a static graph by tracing the execution path of Python code and is currently an experimental feature. Subsequent introduction will explain in detail the difference among the three principles and their respective characteristics.

#### ast

In dynamic graph mode, the user can modify a function to execute in static graph mode by using the `@jit(capture_mode=“ast”)` decorator, which we call ast. Also, since ast is the current default configuration for jit acceleration, you can also use `@jit` to decorate directly. Users need to manually specify the functions that need to be run in static graph mode to get more precise performance benefits. At the same time, because the static graph mode requires functions to be compiled first, the syntax and data structures used inside the functions need to strictly follow the [Static Graph Syntax Specification](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph.html). If there is any unresolvable syntax or data structure in the function, it will be compiled and reported as an error.

**ast Usage**

The user can specify that the function is to be executed as a static graph via the `@jit` decorator, for example:

```python
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore import jit
from mindspore import Tensor

@jit
def tensor_cal(x, y, z):
    return ops.matmul(x, y) + z

x = Tensor(np.ones(shape=[2, 3]), ms.float32)
y = Tensor(np.ones(shape=[3, 4]), ms.float32)
z = Tensor(np.ones(shape=[2, 4]), ms.float32)
ret = tensor_cal(x, y, z)
print(ret)
```

```text
[[4. 4. 4. 4.]
 [4. 4. 4. 4.]]
```

In the above use case, the tensor_cal function is modified by the @jit decorator, and the function follows the pattern of the static graph when it is called in order to capture the performance gains during the execution period of the function.

**Advantages**

- With the ast model, users have more programming autonomy and more precise performance optimization, allowing them to tune the performance of the network to the optimal level based on function characteristics and usage experience.
- With the ast mode, if you encounter an error within a static graph, you can remove the `@jit` decorator and locate the program in dynamic graph mode. After the problem is resolved, the function is then reassigned to run in static graph mode.

**Limitations**

- Functions modified by ast must be programmed with an internal syntax that strictly adheres to strict mode of the static graph. If you use syntax or data types that are not supported by the static graph, you will get an error.

**Recommendations for the Use of the ast Model**

- In contrast to dynamic graph execution, a function modified by `@jit` consumes some time to compile a static graph the first time it is called. On subsequent calls to the function, if the original compilation result can be reused, the original compilation result will be used for execution. As a result, functions that are executed multiple times using @jit decorator usually gain more performance benefits.

- The operational efficiency advantage of the static graph pattern is that it optimizes the compilation of @jit-modified functions globally. The more operations a function contains, the higher the upper limit of optimization. Therefore, functions modified by the `@jit` decorator should ideally be large chunks of code with a lot of operations, rather than many small, fragmented functions with only a few operations tagged with a separate jit tag. Otherwise, there may be no performance gain or even degradation.

- The vast majority of calculations and optimizations for MindSpore static graphs are based on optimizations for Tensor calculations, so we recommend that the functions that are modified should be the kind of functions that are used to perform real data calculations, rather than simple scalar calculations or transformations of data structures.

- Functions modified by `@jit` that have constants in their inputs will result in a recompile each time that the function input value changes. See [Constants and Variables within Static Graphs](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph.html#constants-and-variables-within-static-graphs) for the concept of variable constants. Therefore, it is recommended that the modified function takes as input Tensor or data modified by Mutable. Avoid additional performance loss due to multiple compilations.

#### bytecode

In addition to ast, MindSpore provides another static acceleration mechanism, bytecode, which allows the user to modify a function to execute in bytecode mode via the `@jit(capture_mode=“bytecode”)` decorator. When bytecode recognizes that the syntax for entering a static graph is not supported, it will fall back to Python for execution instead of compiling directly and reporting errors. This feature combines performance and ease of use to reduce the occurrence of compilation errors. It is based on the analysis of Python bytecode, graph capture of Python execution flow, allowing subgraphs that can be run as static graphs to be run as static graphs, and allowing subgraphs that are not supported by Python syntax to be run as dynamic graphs, as well as linking the dynamic-static graphs by modifying and adjusting the bytecode, so as to achieve a mixed execution of dynamic and static. While meeting the premise of ease of use, to improve performance as much as possible.

**bytecode Operating Principle**

1. Capture the execution of Python functions based on Python VM_PyInterpreterState_SetEvalFrameFunc, which captures the execution of all Python functions in the execution area using context management.
2. Analyze the function bytecode in conjunction with the current runtime input parameters to construct a control flow graph (CFG) and a data flow graph (DFG).
3. Simulate in-stack and out-stack operations, trace bytecode by bytecode, and derive the output based on the stack inputs. Python 3.7 to Python 3.10 has a corresponding simulation implementation for each bytecode, noting that the type size of the outputs is derived, not the actual execution of the values, unless the constants are collapsed.
4. During the simulated execution of the bytecode, translate the derivation results and operations into MindIR, and finally, optimize the static graph by constant folding, UD analysis (removing useless input and output parameters), etc.
5. Before executing the equivalent static graph, compare the input parameters with the caretaker Guard conditions generated during the optimization process, and based on the runtime information, select the matching static graph for execution.
6. Dynamically manage the matching relationship between Guard and static graph buffer, recycle the unused static graph buffer, and optimize the static graph buffer through Symbolic Shape and Dynamic Shape.

**bytecode Usage**

Setting the capture_mode parameter of jit to bytecode switches the mode of operation of the modifier function to bytecode, for example:

```python
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore import jit
from mindspore import Tensor

@jit(capture_mode="bytecode")
def tensor_cal(x, y, z):
    return ops.matmul(x, y) + z

x = Tensor(np.ones(shape=[2, 3]), ms.float32)
y = Tensor(np.ones(shape=[3, 4]), ms.float32)
z = Tensor(np.ones(shape=[2, 4]), ms.float32)
ret = tensor_cal(x, y, z)
print(ret)
```

```text
[[4. 4. 4. 4.]
 [4. 4. 4. 4.]]
```

**Advantages and Limitations of bytecode**

**Advantages**

- Good user experience, no human intervention, user-written web code always runs properly, and code that can't be executed by static graphs will automatically run using dynamic graphs.
- bytecode can make more statements into the static graph by transforming the byte code. Users do not need to perceive or modify the code.

**Limitations**

- Users can't explicitly do performance acceleration for certain code, and for scenarios with more cracked graphs, the performance acceleration may not be obvious.

### Shard
