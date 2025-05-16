# Graph Construction

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/features/compile/graph_construction.md)

MindSpore provides JIT (just-in-time) technology to optimize the performance. The JIT mode parses the code into an intermediate representation (IR) graph by means of AST tree parsing, Python bytecode parsing or code execution tracing, which serves as a unique representation of the code, and the compiler optimizes the code by optimizing the IR graph to improve the runtime performance. In contrast to the dynamic graph model, this JIT compilation model is called the static graph model.

Based on JIT technology, MindSpore provides a dynamic-static combination approach to improve the operational efficiency of the user's network. The combination of dynamic and static, that is, in the overall run as a dynamic graph, specifies certain code blocks to run as a static graph. Code blocks that run as static graphs are compiled first and then executed, and global optimizations are performed during the compilation period to obtain performance gains during the execution period. Users can modify functions with the `@jit` decorator to specify that they execute according to the pattern of a static graph. For the documentation on the `@jit` decorator, refer to [jit API documentation](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/mindspore/mindspore.jit.html#mindspore.jit).

MindSpore provides three JIT compilation methods, namely, ast, bytecode and trace. The ast converts the functions that are identified by the users manually and need to be executed in accordance with the ast into a static graph through the AST tree parsing. The bytecode is through the Python bytecode parsing, in the dynamic graph as much as possible to build a static graph. The part that can not be converted to a static graph will be in accordance with the dynamic graph for the purpose of combining static and dynamic. The trace constructs a static graph by tracing the execution path of Python code and is currently an experimental feature. Subsequent introduction will explain in detail the difference among the three principles and their respective characteristics.

## Ast

In dynamic graph mode, the user can modify a function to execute in ast mode by using the `@jit(capture_mode=“ast”)` decorator. The syntax and data structures used inside the functions which decorated by ast mode need to strictly follow the [Static Graph Syntax Specification](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/compile/static_graph.html). The ast approach compiles Python code via a source-to-source method, which first parses the Python source code of model definitions into an Abstract Syntax Tree (AST), then converts the AST into MindIR. For example, the following Python code:

```python
@jit
def foo(x, y):
    z = x + y
    return z
```

The corresponding AST is as follows:

![image](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/docs/mindspore/source_zh_cn/features/compile/images/ast.png)

By parsing the above AST, we obtain the following MindIR:

```text
%para1_x: <Tensor[Int64], ()>
%para2_y: <Tensor[Int64], ()>

subgraph instance: foo
subgraph @foo() {
  %0(CNode_17) = PrimFunc_Add(%para1_x, %para2_y)
      : (<Tensor[Int64], ()>, <Tensor[Int64], ()>) -> (<Tensor[Int64], ()>)
  Return(%0)
      : (<Tensor[Int64], ()>)
}
```

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

**Limitations**

- Functions modified by ast must be programmed with an internal syntax that strictly adheres to the static graph.

**Recommendations for the Use of the ast Model**

- In contrast to dynamic graph execution, a function modified by `@jit` consumes some time to compile a static graph the first time it is called. On subsequent calls to the function, if the original compilation result can be reused, the original compilation result will be used for execution. As a result, functions that are executed multiple times using @jit decorator usually gain more performance benefits.

- The operational efficiency advantage of the static graph pattern is that it optimizes the compilation of @jit-modified functions globally. The more operations a function contains, the higher the upper limit of optimization. Therefore, functions modified by the `@jit` decorator should ideally be large chunks of code with a lot of operations, rather than many small, fragmented functions with only a few operations tagged with a separate jit tag. Otherwise, there may be no performance gain or even degradation.

- The vast majority of calculations and optimizations for MindSpore static graphs are based on optimizations for Tensor calculations, so we recommend that the functions that are modified should be the kind of functions that are used to perform real data calculations, rather than simple scalar calculations or transformations of data structures.

- Functions modified by `@jit` that have constants in their inputs will result in a recompile each time that the function input value changes. See [Constants and Variables Within JIT](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/compile/static_graph.html#constants-and-variables-within-jit) for the concept of variable constants. Therefore, it is recommended that the modified function takes as input Tensor or data modified by Mutable. Avoid additional performance loss due to multiple compilations.

## Bytecode

In addition to ast, MindSpore provides another static acceleration mechanism, bytecode, which allows the user to modify a function to execute in bytecode mode via the `@jit(capture_mode=“bytecode”)` decorator. When bytecode recognizes that the syntax for entering a static graph is not supported, it will fall back to Python for execution instead of compiling directly and reporting errors. This feature combines performance and ease of use to reduce the occurrence of compilation errors. It is based on the analysis of Python bytecode, graph capture of Python execution flow, allowing subgraphs that can be run as static graphs to be run as static graphs, and allowing subgraphs that are not supported by Python syntax to be run as dynamic graphs, as well as linking the dynamic-static graphs by modifying and adjusting the bytecode, so as to achieve a mixed execution of dynamic and static. While meeting the premise of ease of use, to improve performance as much as possible.

**bytecode Operating Principle**

1. Capture the execution of Python functions based on Python VM_PyInterpreterState_SetEvalFrameFunc, which captures the execution of all Python functions in the execution area using context management.
2. Analyze the function bytecode in conjunction with the current runtime input parameters to construct a control flow graph (CFG) and a data flow graph (DFG).
3. Simulate in-stack and out-stack operations, trace bytecode by bytecode, and derive the output based on the stack inputs. Python 3.7 to Python 3.10 has a corresponding simulation implementation for each bytecode, noting that the type size of the outputs is derived, not the actual execution of the values, unless the constants are collapsed.
4. During the simulated execution of the bytecode, translate the derivation results and operations into MindIR, and finally, optimize the static graph by constant folding, UD analysis (removing useless input and output parameters), etc.
5. Before executing the equivalent static graph, compare the input parameters with the caretaker Guard conditions generated during the optimization process, and based on the runtime information, select the matching static graph for execution.
6. Dynamically manage the matching relationship between Guard and static graph buffer, recycle the unused static graph buffer, and optimize the static graph buffer through Symbolic Shape and Dynamic Shape.

The compilation process of bytecode is illustrated in the following diagram:

![image](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/docs/mindspore/source_zh_cn/features/compile/images/bytecode.png)

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

**Advantages**

- Good user experience, no human intervention, user-written web code always runs properly, and code that can't be executed by static graphs will automatically run using dynamic graphs.
- bytecode can make more statements into the static graph by transforming the byte code. Users do not need to perceive or modify the code.

**Limitations**

- Users can't explicitly do performance acceleration for certain code, and for scenarios with more cracked graphs, the performance acceleration may not be obvious.

## Trace

MindSpore also offers another static acceleration mechanism called trace. Users can decorate a function with the `@jit(capture_mode=“trace”)` decorator to execute the function in trace mode. In this mode, the code first runs in pynative mode, during which the operators executed at runtime are recorded and captured into the computation graph. Subsequent executions of the decorated code will directly execute the computation graph constructed during the first execution. This mechanism does not parse syntax but only captures the operators called during runtime, thus avoiding syntax-related errors. It captures the operators invoked during the execution of the pynative mode, captures the Python execution flow into a graph, and compiles the captured operators into the computation graph. Operations without corresponding operators will have their return values recorded as constants in the computation graph. The generated computation graph runs in the manner of static graph execution.

**trace Usage**

Setting the capture_mode parameter of jit to trace switches the mode of operation of the modifier function to trace, for example:

```python
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore import jit
from mindspore import Tensor

@jit(capture_mode="trace")
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

**Advantages of trace**

- The graph construction capability is robust; as long as the code has corresponding operators, they can be captured into the graph without the need for additional adaptation. There will be no syntax-related errors when building the static graph.
- Good user experience, no human intervention, user-written web code always runs properly.

**Limitations of trace**

- It is unable to detect the control flow within the code, and correctness cannot be ensured in scenarios where different branches of the control flow are entered during multiple executions.
- Operations in the code that are not defined as operators, such as calls to third-party libraries, are fixed as constants in the computation graph, and correctness cannot be guaranteed across multiple runs.

