# Combination of Dynamic and Static Graphs

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced/compute_graph/combine.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

Currently, dynamic and static graphs are supported in the industry. Dynamic graphs are executed through explanation, with dynamic syntax affinity and flexible expression. Static graphs are executed through just in time (JIT) build, which focuses on static syntax and has many syntax constraints. The build process of the dynamic graph is different from that of the static graph. As a result, the syntax constraints are also different.

For dynamic and static graph modes, MindSpore first unifies the API expression and uses the same APIs in the two modes. Then, it unifies the underlying differentiation mechanism of dynamic and static graphs.

![dynamic](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_en/advanced/pynative_graph/images/framework1.png)

## Implementation Principle

MindSpore allows you to use the `ms_function` modifier to modify objects that need to be executed using static graphs, achieving combination of dynamic and static graphs. The following uses a simple combination example to describe the implementation principle. The sample code is as follows:

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms
from mindspore import ms_function

class Add(nn.Cell):
    """Define a class to implement the x self addition."""
    def construct(self, x):
        x = x + x
        x = x + x
        return x

class Mul(nn.Cell):
    """Define a class to implement the x self multiplication."""
    @ms_function  # Use ms_function to modify the function. This function is executed in static graph mode.
    def construct(self, x):
        x = x * x
        x = x * x
        return x

class Test(nn.Cell):
    """Define a class to implement Add(x), Mul(x), and then Add(x)."""
    def __init__(self):
        super(Test, self).__init__()
        self.add = Add()
        self.mul = Mul()

    def construct(self, x):
        x = self.add(x)
        x = self.mul(x)
        x = self.add(x)
        return x

ms.set_context(mode=ms.PYNATIVE_MODE)
x = ms.Tensor(np.ones([3, 3], dtype=np.float32))
print("init x:\n", x)
net = Test()
x = net(x)
print("\nx:\n", x)
```

```text
    init x:
     [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]

    x:
     [[1024. 1024. 1024.]
     [1024. 1024. 1024.]
     [1024. 1024. 1024.]]
```

According to the preceding information, after the test operation, the final value of x is a 3\*3 matrix whose each element is 8. The following figure shows the build method of this test case according to the execution sequence.

![msfunction](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_en/advanced/pynative_graph/images/ms_function.png)

Functions modified by `ms_function` are built and executed in static graph mode. If the network involves reverse derivation, the part modified by `ms_function` is also used to generate a backward graph in the form of an entire graph. The backward graph is connected to backward graphs of operators before and after the graph and then delivered for execution. The cache policy is the same as that of the static graph. When the input shape and type information of the same function object is the same, the built graph structure is cached.

## `ms_function` Modifier

To improve the execution speed of forward computing tasks in dynamic graph mode, MindSpore provides the `ms_function` modifier. You can modify Python functions or member functions of Python classes to build them into computational graphs. Technologies such as graph optimization are used to improve the running speed.

### Usage

MindSpore supports  static build in dynamic graphs. You can use the `ms_function` modifier to modify the function objects that need to be executed using static graphs to implement mixed execution of dynamic and static graphs.

#### 1. Modifying Independent Function

When using the `ms_function` modifier, you can modify an independently defined function so that it can run in static graph mode. The following is an example:

```python
import numpy as np
import mindspore.ops as ops
from mindspore import ms_function

# Set the running mode to dynamic graph mode.
ms.set_context(mode=ms.PYNATIVE_MODE)

# Use the modifier to specify the execution in static graph mode.
@ms_function
def add_func(x, y):
    return ops.add(x, y)

x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
y = ms.Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))

out = add_func(x, y)
print(out)
```

```text
    [5. 7. 9.]
```

In the preceding sample code, although the running mode is set to dynamic graph mode at the beginning, the `add_func(x, y)` function is modified using the `ms_function` modifier. Therefore, the `add_func(x, y)` function still runs in static graph mode.

#### 2. Modifying the Member Functions of a Class

When using the `ms_function` modifier, you can modify the member methods of the `Cell` subclass, `ms_class` class, or common user defined class. The sample code is as follows:

```python
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from mindspore import ms_function

# Set the running mode to dynamic graph mode.
ms.set_context(mode=ms.PYNATIVE_MODE)

class Add(nn.Cell):

    @ms_function # Use the modifier to specify the execution in static graph mode.
    def construct(self, x, y):
        out = x + y
        return out

x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
y = ms.Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))

grad_ops = ops.GradOperation(get_all=True)  # Define the derivation operation.
net = Add()
grad_out = grad_ops(net)(x, y)

print("Infer result:\n", net(x, y))

print("Gradient result:")
print("Grad x Tensor1:\n", grad_out[0])  # Derivation of x
print("Grad y Tensor2:\n", grad_out[1])  # Derivation of y
```

```text
    Infer result:
     [5. 7. 9.]
    Gradient result:
    Grad x Tensor1:
     [1. 1. 1.]
    Grad y Tensor2:
     [1. 1. 1.]
```

According to the preceding information, the sum of x and y is \[5, 7, 9\]. The derivation result of x is the same as that of y, that is, \[1, 1, 1\].

### Precautions

When using `ms_function` to modify functions to improve execution efficiency, pay attention to the following points:

1. Functions modified by `ms_function` must be within the syntax scope supported by static graph build, including but not limited to data types.

2. The control flow syntax supported by a function modified by `ms_function` is the same as that supported by the static graph. An acceleration effect is achieved only for a control flow structure with a fixed loop count or a branch condition.

3. When the `ms_function` function is used in PyNative mode, the parts that are not modified by `ms_function` support breakpoint debugging, and the parts modified by `ms_function` do not support breakpoint debugging because they are built in static graph mode.

4. Functions modified by `ms_function` are built and executed in static graph mode. Therefore, `ms_function` does not support the Hook operator in the modified functions or the customized Bprop function.

5. Functions modified by `ms_function` are affected by side effects of static graph functions. Side effects of a function refer to the additional effects on the main function in addition to the return value of the function, for example, modifying global variables (variables other than the function) and modifying function parameters.

    Scenario 1:

    ```python
    import numpy as np
    import mindspore as ms
    from mindspore import ms_function

    # pylint: disable=W0612

    value = 5

    @ms_function
    def func(x, y):
        out = x + y
        value = 1
        return out

    ms.set_context(mode=ms.PYNATIVE_MODE)
    x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
    y = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
    func(x, y)
    print(value)
    ```

    ```text
        5
    ```

    In this scenario, `value` is a global variable and is modified in the `func` function. In this case, if `ms_function` is used to modify the `func` function, the global variable `value` is not changed. The reason is that statements irrelevant to return values are optimized during static graph build.

    Scenario 2:

    ```python
    import numpy as np
    import mindspore.nn as nn
    import mindspore as ms
    from mindspore import ms_function

    class Func(nn.Cell):
        def __init__(self):
            super(Func, self).__init__()
            self.value = 5

        @ms_function
        def construct(self, x):
            out = self.value + x
            return out

    ms.set_context(mode=ms.PYNATIVE_MODE)
    x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
    func = Func()
    print("out1:", func(x))
    func.value = 1
    print("out2:", func(x))
    ```

    ```text
        out1: [6. 7. 8.]
        out2: [6. 7. 8.]
    ```

    According to the preceding information, after the member variable `value` of the `func` class is changed to 1, the `construct` operation of the member function is not affected. In this scenario, `ms_function` is used to modify the `construct` member function of the `func` object. When `construct` is executed, it is built and executed in static graph mode. The static graph caches the build result. Therefore, when `func` is called for the second time, the modification of `value` does not take effect.

6. If a function with the `ms_function` modifier contains operators (such as `MatMul` and `Add`) that do not require parameter training, these operators can be directly called in the modified function. If the modified function contains operators (such as `Conv2D` and `BatchNorm` operators) that require parameter training, these operators must be instantiated outside the modified function. The following uses sample code to describe the two scenarios.

    Scenario 1: Directly call an operator (`mindspore.ops.Add` in the example) that does not require parameter training in the modified function. The sample code is as follows:

    ```python
    import numpy as np
    import mindspore as ms
    import mindspore.ops as ops
    from mindspore import ms_function

    ms.set_context(mode=ms.PYNATIVE_MODE)

    add = ops.Add()

    @ms_function
    def add_fn(x, y):
        res = add(x, y)
        return res

    x = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
    y = ms.Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))
    z = add_fn(x, y)

    print("x:", x.asnumpy(), "\ny:", y.asnumpy(), "\nz:", z.asnumpy())
    ```

    ```text
        x: [1. 2. 3.]
        y: [4. 5. 6.]
        z: [5. 7. 9.]
    ```

    Scenario 2: The operator (`mindspore.nn.Conv2d` in the example) that requires parameter training must be instantiated outside the modified function. The sample code is as follows:

    ```python
    import numpy as np
    import mindspore.nn as nn
    import mindspore as ms
    from mindspore import ms_function

    ms.set_context(mode=ms.PYNATIVE_MODE)

    # Instantiate the conv_obj operator in the conv_fn function.
    conv_obj = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=0)
    conv_obj.init_parameters_data()

    @ms_function
    def conv_fn(x):
        res = conv_obj(x)
        return res

    input_data = np.random.randn(1, 3, 3, 3).astype(np.float32)
    z = conv_fn(ms.Tensor(input_data))
    print(z.asnumpy())
    ```

    ```text
        [[[[ 0.00829158 -0.02994147]
        [-0.09116832 -0.00181637]]

        [[-0.00519348 -0.02172063]
        [-0.04015012 -0.02083161]]

        [[ 0.00608188 -0.01443425]
        [-0.01468289  0.01200477]]

        [[ 0.00845292  0.00044869]
        [-0.00361492  0.01993337]]]]
    ```
