# Inline

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/model_train/compile/inline.md)

In traditional compilers, **inline** is an optimization technique that replaces function calls with the actual code of the called function, improving program performance. For example, consider a C++ `add` function that sums two numbers:

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

First, MindSpore's graph compiler converts the Python program into a computational graph. The function calls in the Python program are converted into calls between calculation graphs, and the original calculation graph is similar to the following. The main graph `f1` calls the secondary subgraph `f2`.

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
