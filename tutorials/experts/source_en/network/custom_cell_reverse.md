# Customizing Reverse Propagation Function of Cell

<a href="https://gitee.com/mindspore/docs/blob/r1.9/tutorials/experts/source_en/network/custom_cell_reverse.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png"></a>

When MindSpore is used to build a neural network, the `nn.Cell` class needs to be inherited. We might have the following problems when we construct networks:

1. There are operations or operators in Cell that are not derivable or for which reverse propagation rules are not yet defined.

2. When replacing certain forward calculation procedures of Cell, you need to customize the corresponding reverse propagation function.

Then we can use the function of customizing the backward propagation function of the Cell object. The format is as follows:

```python
def bprop(self, ..., out, dout):
    return ...
```

- Input parameters: Input parameters in the forward propagation plus `out` and `dout`. `out` indicates the computation result of the forward propagation, and `dout` indicates the gradient returned to the `nn.Cell` object.
- Return values: Gradient of each input in the forward propagation. The number of return values must be the same as the number of inputs in the forward propagation.

A complete simple example is as follows:

```python
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()

    def construct(self, x, y):
        out = self.matmul(x, y)
        return out

    def bprop(self, x, y, out, dout):
        dx = x + 1
        dy = y + 1
        return dx, dy


class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation(get_all=True)

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)


x = ms.Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=ms.float32)
y = ms.Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=ms.float32)
out = GradNet(Net())(x, y)
print(out)
```

```text
    (Tensor(shape=[2, 3], dtype=Float32, value=
    [[ 1.50000000e+00,  1.60000002e+00,  1.39999998e+00],
     [ 2.20000005e+00,  2.29999995e+00,  2.09999990e+00]]), Tensor(shape=[3, 3], dtype=Float32, value=
    [[ 1.00999999e+00,  1.29999995e+00,  2.09999990e+00],
     [ 1.10000002e+00,  1.20000005e+00,  2.29999995e+00],
     [ 3.09999990e+00,  2.20000005e+00,  4.30000019e+00]]))
```

This example customizes the gradient calculation process for the `MatMul` operation by defining `bprop` function of Cell, where `dx` is the derivative of the input `x`, `dy` is the derivative of the input `y`, `out` is the result of the `MatMul` calculation, and `dout` is the gradient passed back to `Net`.

## Application example

1. There are some operators which is non-differentiable or has not been defined the back propagation function in the Cell. For example, the operator `ReLU6` has not been defined its second-order back propagation rule, which can be defined by customizing the `bprop` function of Cell. The code is as follow:

   ```python
   import mindspore.nn as nn
   from mindspore import Tensor
   from mindspore import dtype as mstype
   import mindspore.ops as ops


   class ReluNet(nn.Cell):
       def __init__(self):
           super(ReluNet, self).__init__()
           self.relu = ops.ReLU()

       def construct(self, x):
           return self.relu(x)


   class Net(nn.Cell):
       def __init__(self):
           super(Net, self).__init__()
           self.relu6 = ops.ReLU6()
           self.relu = ReluNet()

       def construct(self, x):
           return self.relu6(x)

       def bprop(self, x, out, dout):
           dx = self.relu(x)
           return (dx, )


   x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
   net = Net()
   out = ops.grad(ops.grad(net))(x)
   print(out)
   ```

   ```text
   [[1. 1. 1.]
    [1. 1. 1.]]
   ```

   The above code defines the first-order back propagation rule by customizing the `bprop` function of `Net` and gets the second-order back propagation rule by the back propagation rule of `self.relu` in the `bprop`.

2. We need the customized back propagation function when we want to replace some forward calculate process of the Cell. For example, there is following code in the network SNN:

   ```python
   class relusigmoid(nn.Cell):
       def __init__(self):
           super().__init__()
           self.sigmoid = ops.Sigmoid()
           self.greater = ops.Greater()

       def construct(self, x):
           spike = self.greater(x, 0)
           return spike.astype(mindspore.float32)

       def bprop(self, x, out, dout):
           sgax = self.sigmoid(x * 5.0)
           grad_x = dout * (1 - sgax) * sgax * 5.0
           return (grad_x,)

   class IFNode(nn.Cell):
       def __init__(self, v_threshold=1.0, fire=True, surrogate_function=relusigmoid()):
           super().__init__()
           self.v_threshold = v_threshold
           self.fire = fire
           self.surrogate_function = surrogate_function

       def construct(self, x, v):
           v = v + x
           if self.fire:
               spike = self.surrogate_function(v - self.v_threshold) * self.v_threshold
               v -= spike
               return spike, v
           return v, v
   ```

   The above code replaces the origin sigmoid activation function in the sub-network `IFNode` with a customized activation function relusigmoid, and then we should customize the new back propagation function for the new activation function.

## Constraints

- If the number of return values of the `bprop` function is 1, the return value must be written in the tuple format, that is, `return (dx,)`.
- In graph mode, the `bprop` function needs to be converted into a graph IR. Therefore, the static graph syntax must be complied with. For details, see [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/r1.9/note/static_graph_syntax_support.html).
- Only support returning the gradient of the forward propagation input, not the gradient of the `Parameter`.
- The use of `Parameter` is not supported in `bprop`.
