# Process Control Statements

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/network/control_flow.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

Currently, there are two execution modes of a mainstream deep learning framework: a static graph mode (`GRAPH_MODE`) and a dynamic graph mode (`PYNATIVE_MODE`).

In `PYNATIVE_MODE`, MindSpore fully supports process control statements of the native Python syntax. In `GRAPH_MODE`, MindSpore performance is optimized during build. Therefore, there are some special constraints on using process control statements when during network definition. Other constraints are the same as those in the native Python syntax.

When switching the running mode from dynamic graph to static graph, pay attention to the [static graph syntax support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html#static-graph-syntax-support). The following describes how to use process control statements when defining a network in `GRAPH_MODE`.

## Constant and Variable Conditions

When a network is defined in `GRAPH_MODE`, MindSpore classifies condition expressions in process control statements into constant and variable conditions. During graph build, a condition expression that can be determined to be either true or false is a constant condition, while a condition expression that cannot be determined to be true or false is a variable condition. **MindSpore generates control flow operators on a network only when the condition expression is a variable condition.**

It should be noted that, when a control flow operator exists in a network, the network is divided into multiple execution subgraphs, and process jumping and data transmission between the subgraphs cause performance loss to some extent.

### Constant Conditions

Check methods:

- The condition expression does not contain tensors or any list, tuple, or dict whose elements are of the tensor type.
- The condition expression contains tensors or a list, tuple, or dict of the tensor type, but the condition expression result is not affected by the tensor value.

Examples:

- `for i in range(0,10)`: `i` is a scalar. The result of the potential condition expression `i < 10` can be determined during graph build. Therefore, it is a constant condition.

- `self.flag`: A scalar of the Boolean type. Its value is determined when the Cell object is built.

- `x + 1 < 10`: `x` is a scalar: The value of `x + 1` is uncertain when the Cell object is built. MindSpore computes the results of all scalar expressions during graph build. Therefore, the value of the expression is determined during build.

- `len(my_list) < 10`: `my_list` is a list object whose element is of the tensor type. This condition expression contains tensors, but the expression result is not affected by the tensor value and is related only to the number of tensors in `my_list`.

### Variable Conditions

Check method:

- The condition expression contains tensors or a list, tuple, or dict of the tensor type, and the condition expression result is affected by the tensor value.

Examples:

- `x < y`: `x` and `y` are operator outputs.

- `x in list`: `x` is the operator output.

The operator output can be determined only when each step is executed. Therefore, the preceding two conditions are variable conditions.

## if Statement

When defining a network in `GRAPH_MODE` using the `if` statement, pay attention to the following: **When the condition expression is a variable condition, the same variable in different branches must be assigned the same data type.**

### if Statement Under a Variable Condition

In the following code, shapes of tensors assigned to the `out` variable in the `if` and `else` branches are `()` and `(2,)`, respectively. The shape of the tensor returned by the network is determined by the condition `x < y`. The result of `x < y` cannot be determined during graph build. Therefore, whether the `out` shape is `()` or `(2,)` cannot be determined during graph build. MindSpore throws an exception due to type derivation failure.

```python
import numpy as np
import mindspore as ms
from mindspore import nn

class SingleIfNet(nn.Cell):

    def construct(self, x, y, z):
        # Build an if statement whose condition expression is a variable condition.
        if x < y:
            out = x
        else:
            out = z
        out = out + 1
        return out

forward_net = SingleIfNet()

x = ms.Tensor(np.array(0), dtype=ms.int32)
y = ms.Tensor(np.array(1), dtype=ms.int32)
z = ms.Tensor(np.array([1, 2]), dtype=ms.int32)

output = forward_net(x, y, z)
```

Execute the preceding code. The error information is as follows:

```text
ValueError: mindspore/ccsrc/pipeline/jit/static_analysis/static_analysis.cc:800 ProcessEvalResults] Cannot join the return values of different branches, perhaps you need to make them equal.
Shape Join Failed: shape1 = (), shape2 = (2).
```

### if Statement Under a Constant Condition

When the condition expression in the `if` statement is a constant condition, the usage of the condition expression is the same as that of the native Python syntax, and there is no additional constraint. In the following code, the condition expression `x < y + 1` of the `if` statement is a constant condition (because `x` and `y` are scalar constants). During graph build, the `out` variable is of the scalar `int` type. The network can be built and executed properly, and the correct result `1` is displayed.

```python
import numpy as np
import mindspore as ms
from mindspore import nn

class SingleIfNet(nn.Cell):

    def construct(self, z):
        x = 0
        y = 1

        # Build an if statement whose condition expression is a constant condition.
        if x < y + 1:
            out = x
        else:
            out = z
        out = out + 1

        return out

z = ms.Tensor(np.array([0, 1]), dtype=ms.int32)
forward_net = SingleIfNet()

output = forward_net(z)
print("output:", output)
```

```text
    output: 1
```

## for Statement

The `for` statement expands the loop body. Therefore, the number of subgraphs and operators of the network that uses the `for` statement depends on the number of loops of the `for` statement. If the number of operators or subgraphs is too large, more hardware resources are consumed.

In the following sample code, the loop body in the `for` statement is executed for three times, and the output is `5`.

```python
import numpy as np
from mindspore import nn
import mindspore as ms

class IfInForNet(nn.Cell):

    def construct(self, x, y):
        out = 0

        # Build a for statement whose condition expression is a constant condition.
        for i in range(0, 3):
            # Build an if statement whose condition expression is a variable condition.
            if x + i < y:
                out = out + x
            else:
                out = out + y
            out = out + 1

        return out

forward_net = IfInForNet()

x = ms.Tensor(np.array(0), dtype=ms.int32)
y = ms.Tensor(np.array(1), dtype=ms.int32)

output = forward_net(x, y)
print("output:", output)
```

```text
    output: 5
```

The `for` statement expands the loop body. Therefore, the preceding code is equivalent to the following code:

```python
import numpy as np
from mindspore import nn
import mindspore as ms

class IfInForNet(nn.Cell):
    def construct(self, x, y):
        out = 0

        # Loop: 0
        if x + 0 < y:
            out = out + x
        else:
            out = out + y
        out = out + 1
        # Loop: 1
        if x + 1 < y:
            out = out + x
        else:
            out = out + y
        out = out + 1
        # Loop: 2
        if x + 2 < y:
            out = out + x
        else:
            out = out + y
        out = out + 1

        return out

forward_net = IfInForNet()

x = ms.Tensor(np.array(0), dtype=ms.int32)
y = ms.Tensor(np.array(1), dtype=ms.int32)

output = forward_net(x, y)
print("output:", output)
```

```text
    output: 5
```

According to the preceding sample code, using the `for` statement may cause too many subgraphs in some scenarios. To reduce hardware resource overhead and improve network build performance, you can convert the `for` statement to the `while` statement whose condition expression is a variable condition.

## while Statement

The `while` statement is more flexible than the `for` statement. When the condition of `while` is a constant, `while` processes and expands the loop body in a similar way as `for`.

When the condition expression of `while` is a variable condition, the `while` statement does not expand the loop body. Instead, a control flow operator is generated during graph execution. Therefore, the problem of too many subgraphs caused by the `for` loop can be avoided.

### while Statement Under a Constant Condition

In the following sample code, the loop body in the `for` statement is executed for three times, and the output result is `5`, which is essentially the same as the sample code in the `for` statement.

```python
import numpy as np
from mindspore import nn
import mindspore as ms

class IfInWhileNet(nn.Cell):

    def construct(self, x, y):
        i = 0
        out = x
        # Build a while statement whose condition expression is a constant condition.
        while i < 3:
            # Build an if statement whose condition expression is a variable condition.
            if x + i < y:
                out = out + x
            else:
                out = out + y
            out = out + 1
            i = i + 1
        return out

forward_net = IfInWhileNet()
x = ms.Tensor(np.array(0), dtype=ms.int32)
y = ms.Tensor(np.array(1), dtype=ms.int32)

output = forward_net(x, y)
print("output:", output)
```

```text
    output: 5
```

### while Statement Under a Variable Condition

1. Constraint 1: **When the condition expression in the while statement is a variable condition, the while loop body cannot contain computation operations of non-tensor types, such as scalar, list, and tuple.**

    To avoid too many control flow operators, you can use the `while` statement whose condition expression is a variable condition to rewrite the preceding code.

    ```python
    import numpy as np
    from mindspore import nn
    import mindspore as ms

    class IfInWhileNet(nn.Cell):

        def construct(self, x, y, i):
            out = x
            # Build a while statement whose condition expression is a variable condition.
            while i < 3:
                # Build an if statement whose condition expression is a variable condition.
                if x + i < y:
                    out = out + x
                else:
                    out = out + y
                out = out + 1
                i = i + 1
            return out

    forward_net = IfInWhileNet()
    i = ms.Tensor(np.array(0), dtype=ms.int32)
    x = ms.Tensor(np.array(0), dtype=ms.int32)
    y = ms.Tensor(np.array(1), dtype=ms.int32)

    output = forward_net(x, y, i)
    print("output:", output)
    ```

    ```text
        output: 5
    ```

    It should be noted that in the preceding code, the condition expression of the `while` statement is a variable condition, and the `while` loop body is not expanded. The expressions in the `while` loop body are computed during the running of each step. In addition, the following constraints are generated:

    > When the condition expression in the `while` statement is a variable condition, the `while` loop body cannot contain computation operations of non-tensor types, such as scalar, list, and tuple.

    These types of computation operations are completed during graph build, which conflicts with the computation mechanism of the `while` loop body during execution. The following uses sample code as an example:

    ```Python
    import numpy as np
    from mindspore import nn
    import mindspore as ms
    class IfInWhileNet(nn.Cell):

        def __init__(self):
            super().__init__()
            self.nums = [1, 2, 3]

        def construct(self, x, y, i):
            j = 0
            out = x

            # Build a while statement whose condition expression is a variable condition.
            while i < 3:
                if x + i < y:
                    out = out + x
                else:
                    out = out + y
                out = out + self.nums[j]
                i = i + 1
                # Build scalar computation in the loop body of the while statement whose condition expression is a variable condition.
                j = j + 1

            return out

    forward_net = IfInWhileNet()
    i = ms.Tensor(np.array(0), dtype=ms.int32)
    x = ms.Tensor(np.array(0), dtype=ms.int32)
    y = ms.Tensor(np.array(1), dtype=ms.int32)

    output = forward_net(x, y, i)
    ```

    In the preceding code, the `while` loop body of the condition expression `i < 3` contains scalar computation `j = j + 1`. As a result, an error occurs during graph build. The following error information is displayed during code execution:

    ```text
    IndexError: mindspore/core/abstract/prim_structures.cc:127 InferTupleOrListGetItem] list_getitem evaluator index should be in range[-3, 3), but got 3.
    ```

2. Constraint 2: **When the condition expression in the while statement is a variable condition, the input shape of the operator cannot be changed in the loop body.**

    MindSpore requires that the input shape of the same operator on the network be determined during graph build. However, changing the input shape of the operator in the `while` loop body takes effect during graph execution.

    The following uses sample code as an example:

    ```Python
    import numpy as np
    from mindspore import nn
    import mindspore as ms
    from mindspore import ops

    class IfInWhileNet(nn.Cell):

        def __init__(self):
            super().__init__()
            self.expand_dims = ops.ExpandDims()

        def construct(self, x, y, i):
            out = x
            # Build a while statement whose condition expression is a variable condition.
            while i < 3:
                if x + i < y:
                    out = out + x
                else:
                    out = out + y
                out = out + 1
                # Change the input shape of an operator.
                out = self.expand_dims(out, -1)
                i = i + 1
            return out

    forward_net = IfInWhileNet()
    i = ms.Tensor(np.array(0), dtype=ms.int32)
    x = ms.Tensor(np.array(0), dtype=ms.int32)
    y = ms.Tensor(np.array(1), dtype=ms.int32)

    output = forward_net(x, y, i)
    ```

    In the preceding code, the `ExpandDims` operator in the `while` loop body of the condition expression `i < 3` changes the input shape of the expression `out = out + 1` in the next loop. As a result, an error occurs during graph build. The following error information is displayed during code execution:

    ```text
    ValueError: mindspore/ccsrc/pipeline/jit/static_analysis/static_analysis.cc:800 ProcessEvalResults] Cannot join the return values of different branches, perhaps you need to make them equal.
    Shape Join Failed: shape1 = (1), shape2 = (1, 1).
    ```
