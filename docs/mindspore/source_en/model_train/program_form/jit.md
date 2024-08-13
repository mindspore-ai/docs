# Dynamic and Static Conversion

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/model_train/program_form/jit.md)

## Overview

The industry currently supports both dynamic and static graph modes. Dynamic graphs are executed by interpretation, with dynamic syntax affinity and flexible expression, and static graphs are executed by using jit compilation optimization, more inclined to static syntax and more restrictions in syntax. For dynamic and static graph modes, firstly MindSpore unifies the API expression, uses the same API in both modes, secondly, unifies the underlying differential mechanism of dynamic and static graphs.

## Interconversion of Dynamic and Static Graphs

In MindSpore, we can switch the execution between using dynamic or static graphs by controlling the mode input parameters. For example:

```python
ms.set_context(mode=ms.PYNATIVE_MODE)
```

Since there are restrictions on Python syntax under static graphs, switching from dynamic to static graphs requires compliance with the syntax restrictions of static graphs in order to execute correctly by using static graphs. For more syntax restrictions for static graphs, refer to [Static Graph Syntax Restrictions](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/static_graph_syntax_support.html).

## Combination of Static and Dynamic Graphs

MindSpore supports mixed execution by using static compilation under dynamic graphs. The function objects that need to be executed with static graphs by using jit modification, and in this way you can achieve mixed execution of dynamic and static graphs. For more use of jit, refer to [jit documentation](https://www.mindspore.cn/tutorials/en/master/beginner/accelerate_with_static_graph.html#decorator-based-startup-method).

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

## Static Graph Syntax Enhancement

In the MindSpore static graph mode, users need to follow MindSpore [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/static_graph_syntax_support.html) when writing programs, and there are constraints on the use of syntax. In dynamic graph mode, Python script code will be executed according to Python syntax, and users can use any Python syntax. It can be seen that the syntax constraints of static and dynamic graphs are different.

JIT Fallback considers the unification of static and dynamic graphs from the perspective of static graphs. When an unsupported syntax is found during compilation, the syntax is Fallback to the Python interpreter for interpretation execution. Through the JIT Fallback feature, static graphs can support as much dynamic graph syntax as possible, so that static graphs provide a syntax experience close to dynamic graphs, so as to achieve dynamic and static unity.

In the graph mode scenario, the MindSpore framework will report an error when it encounters unsupported syntax or symbols during graph compilation, mostly in the type inference stage. In the graph compilation stage, the Python source code written by the user is parsed, and then subsequent static analysis, type derivation, optimization and other steps are performed. Therefore, the JIT Fallback feature needs to be pre-detected for unsupported syntax. Common unsupported syntax mainly includes: calling methods of third-party libraries, calling class names to create objects, calling unsupported Python built-in functions, etc. Interpret execution of unsupported syntax Fallback to the Python interpreter. Since the graph mode uses [MindSpore IR (MindIR)](https://www.mindspore.cn/docs/en/master/design/all_scenarios.html#mindspore-ir-mindir), it is necessary to convert the statement executed by the interpretation to the intermediate representation and record the information required by the interpreter.

The following mainly introduces the static graph syntax supported using the JIT Fallback extension. The default value of the JIT syntax support level option jit_syntax_level is 'LAX', extending the static graph syntax with the ability of JIT Fallback.

### Calling the Third-party Libraries

Complete support for third-party libraries such as NumPy and SciPy. The static graph mode supports many third-party library data types such as np.ndarray and their operation operations, supports obtaining properties and methods that call third-party libraries, and supports interacting with third-party libraries such as NumPy through methods such as Tensor's asnumpy(). In other words, users can call MindSpore's own interface and operator in static graph mode, or directly call the interface of the three-party library, or use them together.

- Supporting data types of third-party libraries (such as NumPy and SciPy), allowing calling and returning objects of third-party libraries.
- Supporting calling methods of third-party libraries.
- Supporting creating Tensor instances by using the data types of the third-party library NumPy.
- The assignment of subscripts for data types in third-party libraries is not currently supported.

For more usage, please refer to the [Calling the Third-party Libraries](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/static_graph_syntax_support.html#calling-the-third-party-libraries) section in [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/static_graph_syntax_support.html).

### Supporting the Use of Custom Classes

Custom classes that do not use '@jit_class' decorations and do not inherit 'nn. Cell`. Through the JIT Fallback technical solution, static graph mode allows creating and referencing instances of custom classes, can directly obtain and call properties and methods of custom class instances, and allows modifying properties(Inplace operations).

For more usage, please refer to the [Supporting the Use of Custom Classes](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/static_graph_syntax_support.html#supporting-the-use-of-custom-classes) section in [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/static_graph_syntax_support.html).

### Basic Operators Support More Data Types

In the syntax of graph mode, the following basic operators in the list are overloaded: ['+', '-', '*', '/', '//', '%', '**', '<<', '>>', '&', '|', '^', 'not', '==', '!=', '<', '>', '<=', '>=', 'in', 'not in', 'y=x[0]']. For more details, please refer to [Operators](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/operators.html). When getting unsupported input type, those operators need to use extended static graph syntax to support, and make the output consistent with the output in the pynative mode.

For more usage, please refer to the [Basic Operators Support More Data Type](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/static_graph_syntax_support.html#basic-operators-support-more-data-type) section in [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/static_graph_syntax_support.html).

### Base Type

Use the JIT Fallback feature to extend support for Python's native data types 'List', 'Dictionary', 'None'. For more usage, please refer to the [Base Type](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/static_graph_syntax_support.html#base-type) section in [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/static_graph_syntax_support.html).

#### Supporting List Inplace Modification Operations

- Support for getting the original `List` object from a global variable.
- Inplace operations on input `List` objects are not supported.
- Support for in-place modification of some `List` built-in functions.

#### Supporting the High-Level Usage of Dictionary

- Supporting Top Graph Return Dictionary.
- Supporting Dictionary Index Value Retrieval and Assignment.

#### Supporting the Usage of None

`None` is a special value in Python that represents null and can be assigned to any variable. Functions that do not have a return value statement are considered to return `None`. At the same time, `None` is also supported as the input parameter or return value of the top graph or subgraph. Support `None` as a subscript of a slice as input to `List`, `Tuple`, `Dictionary`.

### Built-in Functions Support More Data Types

Extend the support for built-in functions. Python built-in functions perfectly support more input types, such as third-party library data types. More support for built-in functions can be found in the [Python built-in functions](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/python_builtin_functions.html) section.

### Supporting Control Flow

In order to improve the support of Python standard syntax, realize dynamic and static unification, and extend the support for more data types in the use of control flow statements. Control flow statements refer to flow control statements such as 'if', 'for', and 'while'. Theoretically, by extending the supported syntax, it is also supported in control flow scenarios. For more usage, please refer to [Supporting Control Flow](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/static_graph_syntax_support.html#supporting-control-flow) section in [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/static_graph_syntax_support.html).

### Supporting Property Setting and Modification

More types of inplace operations are supported. The previous version only supported value modification of the Parameter type through the Inplace operator, and in the static graph mode of MindSpore version 2.1, the properties of custom classes, Cell subclasses, and jit_class classes were supported. In addition to supporting changing the properties of class self and global variables, it also supports inplace operations such as extend(), reverse(), insert(), pop() of the List type. For more usage, please refer to the [Supporting Property Setting and Modification](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/static_graph_syntax_support.html#supporting-property-setting-and-modification) section in [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/static_graph_syntax_support.html).

- Set and modify properties of custom class objects and third-party types.
- Make changes to the Cell's self object.
- Set and modify Cell objects and jit_class objects in the static graph.

### Supporting Derivation

The static graph syntax supported by JIT Fallback also supports its use in derivation. For more usage, please refer to the [Supporting Derivation](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/static_graph_syntax_support.html#supporting-derivation) section in [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/static_graph_syntax_support.html).

### Annotation Type

For the syntax supported by the runtime extensions, nodes are generated that cannot be derived by type and are called `Any` types. Since the type cannot derive the correct type at compile time, this `Any` will be operated with a default maximum precision 'Float64' to prevent loss of precision. To optimize performance, it is recommended to minimize the generation of `Any` types. When the user knows exactly what type of statement will be generated through the extension, it is recommended to use `Annotation @jit.typing:` to specify the corresponding Python statement type, thereby determining the type of the interpretation node and avoiding the generation of `Any` types. For more usage, please refer to the [Annotation Type](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/static_graph_syntax_support.html#annotation-type) section in [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/static_graph_syntax_support.html).

### Instructions for Use

When using the static graph extension support syntax, note the following points:

1. In order to match the support capability of the dynamic graph. That is, it must be within the scope of dynamic graph syntax, including but not limited to data types.

2. When extending the static graph syntax, more syntax is supported, but the execution performance may be affected and is not optimal.

3. When extending the static graph syntax, more syntax is supported, and the ability to import and export cannot be used with MindIR due to use Python.

4. It is not currently supported that the repeated definition of global variables with the same name across Python files, and these global variables are used in the network.

## Conversion Technique from Dynamic Graph to Static Graph

MindSpore provides PIJit, a feature that directly converts a user's dynamic graph code into a static graph without code changes. This feature balances performance and ease of use, removes the cost of switching between static and dynamic modes, and truly unifies static and dynamic modes. It is based on the analysis of Python bytecode, and captures the execution flow of Python graphs. Subgraphs that can be run as static graphs are run as static graphs, and subgraphs that are not supported by Python syntax are run as dynamic graphs, and at the same time, by modifying and adjusting the bytecode to link the static graphs, it achieves the mixed execution of static and dynamic modes. In order to meet the premise of ease of use, improve performance.

### PIJit Includes the Following Features

- 1. Graph Capture: Pre-processing of bytecode, dynamically tracking the execution of the interpretation, recognizing MindSpore accessible graph operations, and providing split graph to ensure the correctness of function (bytecode) functionality.
- 2. Bytecode Support: Currently supports Python 3.7, Python 3.8, Python 3.9 and Python 3.10 version bytecode.
- 3. Graph Optimization: Optimize the bytecode generated in the graph, including branch cropping, bytecode filtering, function bytecode inlining, constant folding and other functions.
- 4. Exception Capture Mechanism: support for with, try-except syntax.
- 5. Support loop processing: implement features such as graph capture and split graph by simulating the operation stack of bytecode.
- 6. UD Analysis: The method of user-def chain analysis of variables solves the problem that some parameter types cannot be used as the return value of static graphs (Function, Bool, None), and reduces the useless parameters, improves the execution efficiency of the graphs, and reduces the copying of data.
- 7. Side effect analysis and processing: to make up for the disadvantage of side effect processing of static graphs. According to different scenarios, collect and record the variables and byte codes that generate side effects, and supplement the processing of side effects outside the static graphs on the basis of guaranteeing the semantics of the program.
- 8. Guard: The Guard records the conditions that need to be met by the inputs for the subgraph/optimization to enter, and checks if the inputs are suitable for the corresponding subgraph optimization.
- 9. Cache：The graph management caches the subgraph/optimization and Guard correspondences.
- 10. Dynamic Shape and Symbolic Shape: Use input_signature to support Dynamic Shape and Symbolic Shape for Tensor/Tensor List/Tensor Tuple as input prompts. Simultaneously supports automatic recognition of Dynamic Shape after multiple runs.
- 11. Compiling by trace: Supports operator and other type derivations during tracking and bytecode analysis processes.
- 12. Automatic mixed precision: Supports the automatic mixed precision capability of the native mindspore.nn.Cell.

### Usage

def jit(fn=None, input_signature=None, hash_args=None, jit_config=None, mode="PIJit"):

The original Jit function uses mode="PSJit", the new feature PIJit uses mode="PIJit", jit_config passes a dictionary of parameters that can provide some optimization and debugging options. For example: print_after_all can print the bytecode of the graph and split graph information, loop_unrolling can provide loop unrolling function, enable_dynamic_shape apply dynamic shape.

### Limitations

- It is not supported to run a function with decoration @jit(mode=\"PIJit\") in static graph mode, in which case the decoration @jit(mode=\"PIJit\") is considered invalid.
- Calls to functions with decoration @jit(mode=\"PIJit\") inside functions decorated with @jit(mode=\"PIJit\") are not supported, and the decorated @jit(mode=\"PIJit\") is considered invalid.