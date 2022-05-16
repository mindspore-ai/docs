# Applying PyNative Mode

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/debug/pynative.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

In PyNative mode, MindSpore supports the execution of single operators, ordinary functions and networks, as well as the operation of individual gradients. Below we will introduce the use of these operations and considerations in detail through sample code.

## Executing Operations

First, we import the dependencies and set the run mode to PyNative mode:

```python
import numpy as np
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor, set_context, PYNATIVE_MODE, dtype as mstype

set_context(mode=PYNATIVE_MODE)
```

### Executing Single Operators

The following is example code of executing Add operator [mindspore.ops.Add](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Add.html#mindspore.ops.Add):

```python
add = ops.Add()
x = Tensor(np.array([1, 2]).astype(np.float32))
y = Tensor(np.array([3, 5]).astype(np.float32))
z = add(x, y)
print("x:", x.asnumpy(), "\ny:", y.asnumpy(), "\nz:", z.asnumpy())
```

### Executing Functions

Execute the custom function `add_func` and the sample code is as follows:

```python
add = ops.Add()

def add_func(x, y):
    z = add(x, y)
    z = add(z, x)
    return z

x = Tensor(np.array([1, 2]).astype(np.float32))
y = Tensor(np.array([3, 5]).astype(np.float32))
z = add_func(x, y)
print("x:", x.asnumpy(), "\ny:", y.asnumpy(), "\nz:", z.asnumpy())
```

### Executing Network

Execute a custom network `Net` to define the network structure in the construst, and the sample code is as follows:

```python
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = ops.Mul()

    def construct(self, x, y):
        return self.mul(x, y)

net = Net()
x = Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
y = Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))
z = net(x, y)

print("x:", x.asnumpy(), "\ny:", y.asnumpy(), "\nz:", z.asnumpy())
```

## Customizing **bprop** Function

Users can customize backpropagation (calculation) function of the nn.Cell object, thus control the process of the nn.Cell object gradient calculation, locating gradient problems.

Custom bprop functions are used by: adding a user-defined bprop function to the defined nn. Cell object. The training process uses user-defined bprop functions to generate reverse graphs.

The sample code is as follows:

```python
set_context(mode=PYNATIVE_MODE)

class Net(nn.Cell):
    def construct(self, x, y):
        z = x * y
        z = z * y
        return z

    def bprop(self, x, y, out, dout):
        x_dout = x + y
        y_dout = x * y
        return x_dout, y_dout

grad_all = ops.GradOperation(get_all=True)
output = grad_all(Net())(Tensor(1, mstype.float32), Tensor(2, mstype.float32))
print(output)
```

## Synchronous Execution

In PyNative mode, in order to improve performance, the operator uses asynchronous execution on the device, so when the operator executes incorrectly, the error message may not be displayed until the program is executed until the end. In response to this situation, MindSpore added a pynative_synchronize setting to control whether asynchronous execution is used on the operator device.

In PyNative mode, the operator defaults to asynchronous execution, and you can control whether the execution is asynchronous by setting the content. When operator execution fails, it is convenient to see the location of the code where the error occurred through the calling stack. The sample code is as follows:

```python
from mindspore import dtype as mstype

# Synchronize operator execution by setting the pynative_synchronize
set_context(mode=PYNATIVE_MODE, pynative_synchronize=True)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.get_next = ops.GetNext([mstype.float32], [(1, 1)], 1, "test")

    def construct(self, x1,):
        x = self.get_next()
        x = x + x1
        return x

set_context()
x1 = np.random.randn(1, 1).astype(np.float32)
net = Net()
output = net(Tensor(x1))
print(output.asnumpy())
```

Output: At this time, the operator is synchronous execution, and when the operator executes incorrectly, you can see the complete call stack and find the wrong line of code.

```text
Traceback (most recent call last):
  File "test.py", line 24, in <module>
    output = net(Tensor(x1))
  File ".../mindspore/nn/cell.py", line 602, in __call__
    raise err
  File ".../mindspore/nn/cell.py", line 599, in __call__
    output = self._run_construct(cast_inputs, kwargs)
  File ".../mindspore/nn/cell.py", line 429, in _run_construct
    output = self.construct(*cast_inputs, **kwargs)
  File "test.py", line 17, in construct
    x = self.get_next()
  File ".../mindspore/ops/primitive.py", line 294, in __call__
    return _run_op(self, self.name, args)
  File ".../mindspore/common/api.py", line 90, in wrapper
    results = fn(*arg, **kwargs)
  File ".../mindspore/ops/primitive.py", line 754, in _run_op
    output = real_run_op(obj, op_name, args)
RuntimeError: mindspore/ccsrc/plugin/device/gpu/kernel/data/dataset_iterator_kernel.cc:139 Launch] For 'GetNext', gpu Queue(test) Open Failed: 2
```

