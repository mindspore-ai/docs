# Custom Neural Network Layers

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/model_train/custom_program/network_custom.md)

Normally, the neural network layer interface and function interface provided by MindSpore can meet the model construction requirements, but since the AI field is constantly updating, it is possible to encounter new network structures without built-in modules. At this point, we can customize the neural network layer through the function interface provided by MindSpore, Primitive operator, and can use the `Cell.bprop` method to customize the reverse. The following are the details of each of the three customization methods.

## Constructing Neural Network Layers by Using the Function Interface

MindSpore provides a large number of basic function interfaces, which can be used to construct complex Tensor operations, encapsulated as neural network layers. The following is an example of `Threshold` with the following equation:

$$
y =\begin{cases}
   x, &\text{ if } x > \text{threshold} \\
   \text{value}, &\text{ otherwise }
   \end{cases}
$$

It can be seen that `Threshold` determines whether the value of the Tensor is greater than the `threshold` value, keeps the value whose judgment result is `True`, and replaces the value whose judgment result is `False`. Therefore, the corresponding implementation is as follows:

```python
class Threshold(nn.Cell):
    def __init__(self, threshold, value):
        super().__init__()
        self.threshold = threshold
        self.value = value

    def construct(self, inputs):
        cond = ops.gt(inputs, self.threshold)
        value = ops.fill(inputs.dtype, inputs.shape, self.value)
        return ops.select(cond, inputs, value)
```

Here `ops.gt`, `ops.fill`, and `ops.select` are used to implement judgment and replacement respectively. The following custom `Threshold` layer is implemented:

```python
m = Threshold(0.1, 20)
inputs = mindspore.Tensor([0.1, 0.2, 0.3], mindspore.float32)
m(inputs)
```

```text
Tensor(shape=[3], dtype=Float32, value= [ 2.00000000e+01,  2.00000003e-01,  3.00000012e-01])
```

It can be seen that `inputs[0] = threshold`, so it is replaced with `20`.

## Custom Cell Reverse

In special scenarios, we not only need to customize the forward logic of the neural network layer, but also want to manually control the computation of its reverse, which we can define through the `Cell.bprop` interface. The function will be used in scenarios such as new neural network structure design and backward propagation speed optimization. In the following, we take `Dropout2d` as an example to introduce custom Cell reverse.

```python
class Dropout2d(nn.Cell):
    def __init__(self, keep_prob):
        super().__init__()
        self.keep_prob = keep_prob
        self.dropout2d = ops.Dropout2D(keep_prob)

    def construct(self, x):
        return self.dropout2d(x)

    def bprop(self, x, out, dout):
        _, mask = out
        dy, _ = dout
        if self.keep_prob != 0:
            dy = dy * (1 / self.keep_prob)
        dy = mask.astype(mindspore.float32) * dy
        return (dy.astype(x.dtype), )

dropout_2d = Dropout2d(0.8)
dropout_2d.bprop_debug = True
```

The `bprop` method has three separate input parameters:

- *x*: Forward input. When there are multiple forward inputs, the same number of inputs are required.
- *out*: Forward input.
- *dout*: When backward propagation is performed, the current Cell executes the previous reverse result.

Generally we need to calculate the reverse result according to the reverse derivative formula based on the forward output and the reverse result of the front layer, and return it. The reverse calculation of `Dropout2d` requires masking the reverse result of the front layer based on the `mask` matrix of the forward output, and then scaling according to `keep_prob`. The final implementation can get the correct calculation result.

When customizing the reverse direction of a Cell, it supports extended writing in PyNative mode and can differentiate the weights inside the Cell. The specific columns are as follows:

```python
class NetWithParam(nn.Cell):
    def __init__(self):
        super(NetWithParam, self).__init__()
        self.w = Parameter(Tensor(np.array([2.0], dtype=np.float32)), name='weight')
        self.internal_params = [self.w]

    def construct(self, x):
        output = self.w * x
        return output

    def bprop(self, *args):
        return (self.w * args[-1],), {self.w: args[0] * args[-1]}
```

`bprop` method supports *args as an input parameter, and the last data in the args array, `args[-1]` is the gradient returned to the cell. Set the weight of differentiation through `self.internal_params`, and return a tuple and a dictionary in the `bprop` function. Return the tuple corresponding to the input gradient, as well as the dictionary corresponding to the gradient with key as the weight and value as the weight.
