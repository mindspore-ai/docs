# Graph Optimization

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/features/compile/graph_optimization.md)

Similar to traditional compilers, MindSpore also performs compilation optimization after graph construction. The main purpose of compilation optimization is to analyze and transform MindSpore's intermediate representation [MindIR](https://www.mindspore.cn/docs/en/r2.6.0rc1/design/all_scenarios.html#mindspore-ir-mindir) by static analysis techniques to achieve goals such as reducing the size of the target code, improving execution efficiency, lowering runtime resource consumption, or enhancing other performance metrics. Compilation optimization is a crucial part of the graph compilation system and plays an extremely important role in improving the performance and resource utilization of the entire neural network model. Compared with the original code that has not been optimized, compilation optimization can bring several times or even tens of times performance improvement.

This section mainly introduces front-end compilation optimization techniques that are independent of specific hardware. Hardware-specific back-end compilation optimization techniques are not within the scope of this discussion.

## Principles of Front-End Compilation Optimization Techniques

Similar to traditional compilation optimization techniques, compilation optimization in MindSpore is also carried out through a series of Passes. Each Pass takes the MindIR produced by the previous Pass as input and generates a new MindIR representation as output after optimization. A large Pass can include multiple smaller Passes, each of which is only responsible for a single point of compilation optimization, such as arithmetic simplify, inline, redundancy elimination and etc. The optimization results produced by one Pass may create optimization opportunities for other Passes, so these Passes can be run in a loop until the MindIR no longer changes.

The selection of which Passes to run and how to arrange the execution order of these Passes has a very important impact on the final compilation result. Depending on the actual situation, the optimization actions to be performed can be adjusted by setting compilation optimization strategies (such as optimization levels, number of iterations, etc.).

## Common Front-End Compilation Optimization Techniques

There are many front-end compilation optimization techniques, such as arithmetic simplify, inline, and redundancy elimination. This section will introduce some representative compilation optimization techniques.

### Arithmetic Simplify

In traditional compilers, arithmetic simplify is a compiler optimization technique aimed at simplifying algebraic expressions in source code, eliminating redundant calculations, improving program execution efficiency, and reducing memory usage.

For example, in the following code snippet:

```cpp
int a = x * 1;
int b = x + 0;
int c = x * 0 + y * 1;
```

Traditional compilers perform equivalent substitution on recognized expressions based on algebraic rules and identities. Common algebraic rules include laws of union, commutative, and distributive, and compilers will try to replace expressions with simpler forms as much as possible. By analyzing AST or SSA analysis is used for optimization, identifying and simplifying code as follows:

```cpp
a = x;
b = x;
c = y;
```

In the MindSpore compiler, the principle of arithmetic simplify is different from traditional compilers. It processes computational graphs rather than traditional control flow graphs. By adjusting the execution order of operators in the computational graph or deleting unnecessary operators, it maintains the simplicity of the graph and improves computational efficiency.

For example, in the following Python code snippet:

```python
import numpy as np
from mindspore.common import Tensor, jit

@jit
def func(x):
    return x + 0

m = Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32))
out = func(m)
```

The MindSpore graph compiler converts Python programs into computational graphs, which consist of multiple subgraphs. The algebraic operations in the source code are converted into operator calls within the subgraph, and it can be seen that the PrimFunc_Add operator is called once.

```txt
%para1_x: <Tensor[Int32], (2, 3)>

subgraph @1_func_14() {
    %0(CNode_7) = PrimFunc_Add(%para1_x, Tensor(shape=[], dtype=Int32, value=0))
        : (<Tensor[int32], (2, 3)>, <Tensor[Int32], (), value=...>) -> (<Tensor[int32], (2, 3)>)

    Return(%0)
        : (<Tensor[int32], (2, 3)>)
}
```

By arithmetic simplify, the PrimFunc_Add operator can be directly removed to simplify the computational graph structure, reducing `x + 0` to `x`.

```txt
%para1_x: <Tensor[Int32], (2, 3)>

subgraph @1_func_14() {
    Return(%para1_x)
        : (<Tensor[int32], (2, 3)>)
}
```

Arithmetic simplify can involve more modifications to the structure of computational graphs, and it is often combined with other compiler optimization techniques such as constant folding and constant propagation to improve program performance.

### Inline

In traditional compilers, inline is an optimization technique that replaces function calls with the actual code of the called function, improving program performance. For example, consider a C++ `add` function that sums two numbers:

```cpp
int add(int a, int b) {
    return a + b;
}

int main() {
    int x = add(3, 5);
    int y = add(x, 10);
    return y;
}
```

The compiler uses inline to directly insert the function body at the call site. This eliminates function call overhead and enables follow-up optimizations (e.g., replacing `3 + 5` with its result at compile time). **Replacing calls with code** is the core idea of inline.

```cpp
int main() {
    int x = 3 + 5;   // Replace the first call.
    int y = x + 10;  // Replace the second call.
    return y;
}
```

In AI frameworks' computational graph compilers, inline serves a similar purpose but operates on "subgraphs" instead of functions. For example, consider a Python program:

```python
from mindspore import Tensor, jit, ops

def f2(x: Tensor, y: Tensor):
    return x * 0.5 + y

@jit
def f1(a: Tensor, b: Tensor, c: Tensor):
    x = f2(a, b)
    y = f2(a, c)
    return x + y

# Create three Tensors with random values, each having a shape of (2, 4).
a = ops.randn(2, 4)
b = ops.randn(2, 4)
c = ops.randn(2, 4)
out = f1(a, b, c)
```

First, MindSpore's graph compiler converts the Python program into a computational graph. The function calls in the Python program are converted into calls between calculation graphs, and the original calculation graph is similar to the following. The main graph `f1` calls the subgraph `f2` twice.

```txt
# Params:
%para1_a: <Tensor[Float32], (2, 4)>
%para2_b: <Tensor[Float32], (2, 4)>
%para3_c: <Tensor[Float32], (2, 4)>

subgraph @f2(%para1_x, %para2_y) {
    %0 = PrimFunc_Mul(%para1_x, Float32(0.5))

    %1 = PrimFunc_Add(%0, %para2_y)

    Return(%2)
}

subgraph @f1() {
  %0(x) = call @f2(%para1_a, %para2_b)  # Call subgraph f2

  %1(y) = call @f2(%para1_a, %para3_c)  # Call subgraph f2

  %2 = PrimFunc_Add(%1, %2)

  Return(%2)
}
```

With inlining, the subgraph `f2` can be expanded and merged into the main graph `f1`.

```txt
subgraph @f1() {
  # First-time subgraph inlining
  %0 = PrimFunc_Mul(%para1_a, Float32(0.5))  # Repeated computation
  %1 = PrimFunc_Add(%0, %para2_b)

  # Second-time subgraph inlining
  %2 = PrimFunc_Mul(%para1_a, Float32(0.5))  # Repeated computation
  %3 = PrimFunc_Add(%2, %para3_c)

  %4 = PrimFunc_Add(%1, %3)

  Return(%4)
}
```

Before inlining, the compiler might not detect repeated operations in the two calls to subgraph `f2` (as subgraphs are often treated as black boxes). After inlining, the compiler clearly sees `x * 0.5` calculated twice, enabling optimizations like **CSE** (Common Subexpression Elimination) to reduce redundant computations.

```txt
subgraph @f1() {
  %0 = PrimFunc_Mul(%para1_a, Float32(0.5))  # CSE merges redundant computations

  %1 = PrimFunc_Add(%0, %para2_b)

  %2 = PrimFunc_Add(%0, %para3_c)  # Directly reuse %0

  %3 = PrimFunc_Add(%1, %2)

  Return(%3)
}
```

With inlining, compilers better identify cross-subgraph optimization opportunities. In addition to CSE, it enables operator fusion, memory management optimizations, and many other optimizations. Thus, inline is a critical optimization mechanism in computational graph compilers and a foundation for many cross-subgraph optimizations.

### Redundancy Elimination

In traditional compilers, redundancy elimination encompasses various compiler optimization techniques aimed at identifying and removing redundant parts of the code during compilation. This process is designed to reduce unnecessary computations and improve the execution efficiency of programs.

Redundant code may be intentionally written by developers for readability purposes or may simply be an unintentional result of the coding process. Additionally, intermediate results generated by other optimization techniques during the compilation process (such as arithmetic simplify, inline and common subexpression elimination) may also create opportunities for redundancy elimination.

There are many techniques for redundancy elimination. This section selects and introduces some of the common ones, including dead code elimination and unreachable code elimination.

1. **Dead code elimination**

    Removing code whose results are not used. For example, in the following C++ code, the variable `c` is not used by any other code. Compilers can use data flow analysis techniques from the field of static analysis to eliminate the computation of code: `int c = x * y`.

    ```cpp
    int func(x, y) {
        int a = x + y;
        int b = x - y;
        int c = x * y; // Dead code
        int d = a / b;
        return d;
    }
    ```

2. **Unreachable code elimination**

    Removing code that is not included in any valid control flow path. For example, in the following C++ code, compilers can use control flow analysis techniques from the field of static analysis to analyze the control flow graph. They can identify that the expression `1 < 0` is always false, and thus the code within this control flow path will never be executed during actual runtime. Therefore, the code in this branch can be eliminated.

    ```cpp
    int func(x, y) {
        int a = x + y;

        int b;
        if 1 < 0 { // Unreachable branch
            b = x + y;
        } else {
            b = x - y;
        }

        int d = a / b;
        return d;
    }
    ```

In MindSpore's graph mode, the purpose and techniques of redundancy elimination are similar to those in traditional compilers. However, unlike traditional compilers, these redundancy optimization techniques are performed on MindIR. Similarly, common redundancy elimination techniques in MindSpore include:

1. **Dead code elimination**

    For example, consider the following Python code with redundant computations:

    ```python
    import mindspore as ms
    from mindspore.common import Tensor, jit

    @jit
    def func(x, y):
        a = x + y
        b = x - y
        c = x * y # Dead code
        d = a / b
        return d

    x = Tensor(20, ms.float32)
    y = Tensor(10, ms.float32)
    out = func(x, y)
    ```

    The MindSpore graph compiler will convert the Python code decorated with `@jit` into the MindIR representation through static analysis and eliminate the redundant computation `c = x * y`. The resulting MindIR is as follows:

    ```txt
    # Params:
    %para1_x: <Tensor[Float32], ()>
    %para2_y: <Tensor[Float32], ()>

    subgraph @func_1() {
    %0(a) = PrimFunc_Add(%para1_x, %para2_y)
        : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Tensor[Float32], ()>)
    %1(b) = PrimFunc_Sub(%para1_x, %para2_y)
        : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Tensor[Float32], ()>)
    %2(d) = PrimFunc_Div(%0, %1)
        : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Tensor[Float32], ()>)
    Return(%2)
        : (<Tensor[Float32], ())
    }
    ```

2. **Unreachable code elimination**

    For example, consider the following Python code with an unreachable path:

    ```python
    import mindspore as ms
    from mindspore.common import Tensor, jit

    @jit
    def func(x, y):
        a = x + y
        if 1 < 0: # Unreachable branch
            b = x + y
        else:
            b = x - y
        d = a / b
        return d

    x = Tensor(20, ms.float32)
    y = Tensor(10, ms.float32)
    out = func(x, y)
    ```

    The MindSpore graph compiler will convert the Python code decorated with `@jit` into the MindIR representation through static analysis and eliminate the redundant control flow branch `1 < 0`. The resulting MindIR is as follows:

    ```txt
    # Params:
    %para1_x: <Tensor[Float32], ()>
    %para2_y: <Tensor[Float32], ()>

    subgraph @func_1() {
    %0(a) = PrimFunc_Add(%para1_x, %para2_y)
        : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Tensor[Float32], ()>)
    %1(b) = PrimFunc_Sub(%para1_x, %para2_y)
        : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Tensor[Float32], ()>)
    %2(d) = PrimFunc_Div(%0, %1)
        : (<Tensor[Float32], ()>, <Tensor[Float32], ()>) -> (<Tensor[Float32], ()>)
    Return(%2) cnode_attrs: {checkpoint: Bool(1)}
        : (<Tensor[Float32], ())
    }
    ```

Redundancy elimination plays a crucial role in compiler optimization. Without changing the original semantics of the program, it can significantly improve execution efficiency by reducing unnecessary runtime computations and saving computing resources. Redundancy elimination is often combined with other compiler optimization techniques to create more opportunities for eliminating redundant code.
