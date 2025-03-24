# Arithmetic Simplify

In traditional compilers, **arithmetic simplify** is a compiler optimization technique aimed at simplifying algebraic expressions in source code, eliminating redundant calculations, improving program execution efficiency, and reducing memory usage.

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

In the MindSpore compiler, the principle of **arithmetic simplify** is different from traditional compilers. It processes computational graphs rather than traditional control flow graphs. By adjusting the execution order of operators in the computational graph or deleting unnecessary operators, it maintains the simplicity of the graph and improves computational efficiency.

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

By **arithmetic simplify**, the PrimFunc_Add operator can be directly removed to simplify the computational graph structure, reducing `x + 0` to `x`.

```txt
%para1_x: <Tensor[Int32], (2, 3)>

subgraph @1_func_14() {
    Return(%para1_x)
        : (<Tensor[int32], (2, 3)>)
}
```

**Arithmetic simplify** can involve more modifications to the structure of computational graphs, and it is often combined with other compiler optimization techniques such as constant folding and constant propagation to improve program performance.