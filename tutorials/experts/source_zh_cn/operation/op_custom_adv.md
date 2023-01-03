# 自定义算子进阶用法

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/tutorials/experts/source_zh_cn/operation/op_custom_adv.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 算子信息注册

算子信息主要描述了算子实现函数所支持的输入输出类型、输入输出数据格式、属性和target（平台信息），它是后端做算子选择和映射时的依据。它通过[CustomRegOp](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.CustomRegOp.html#mindspore-ops-customregop)接口定义，通过[custom_info_register](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.custom_info_register.html#mindspore-ops-custom-info-register)装饰器或者[Custom](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.Custom.html#mindspore-ops-custom)原语构造函数中的`reg_info`参数，实现算子信息与算子实现函数的绑定，并最终注册到MindSpore C++侧的算子信息库。`reg_info`参数优先级高于`custom_info_register`装饰器。

算子信息中的target的值可以为"Ascend"或"GPU"或"CPU"，描述的是算子实现函数在当前target上所支持的输入输出类型、输入输出数据格式和属性等信息，对于同一个算子实现函数，其在不同target上支持的输入输出类型可能不一致，所以通过target进行区分。算子信息在同一target下只会被注册一次。

> - 算子信息中定义输入输出信息的个数和顺序、算子实现函数中的输入输出信息的个数和顺序，两者要完全一致。
> - 对于akg类型的自定义算子，若算子存在属性输入，则必须注册算子信息，算子信息中的属性名称与算子实现函数中使用的属性名称要一致；对于tbe类型的自定义算子，当前必须注册算子信息；对于aot类型的自定义算子，由于算子实现函数需要预先编译成动态库，所以无法通过装饰器方式绑定算子信息，只能通过`reg_info`参数传入算子信息。
> - 若自定义算子只支持特定的输入输出数据类型或数据格式，则需要注册算子信息，以便在后端做算子选择时进行数据类型和数据格式的检查。对于不提供算子信息的情况，则在后端做算子选择和映射的时候，将会从当前算子的输入中推导信息。

## 定义算子反向传播函数

如果算子要支持自动微分，需要定义其反向传播函数（bprop），然后将bprop函数传入`Custom`原语构造函数的`bprop`参数。你需要在bprop中描述利用正向输入、正向输出和输出梯度得到输入梯度的反向计算逻辑。反向计算逻辑可以使用内置算子或自定义Custom算子。

定义算子反向传播函数时需注意以下几点：

- bprop函数的入参顺序约定为正向的输入、正向的输出、输出梯度。若算子为多输出算子，正向输出和输出梯度将以元组的形式提供。
- bprop函数的返回值形式约定为输入梯度组成的元组，元组中元素的顺序与正向输入参数顺序一致。即使只有一个输入梯度，返回值也要求是元组的形式。

下面test_grad.py为例，展示反向传播函数的用法：

```python
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
import mindspore.ops as ops

ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")

# 自定义算子正向实现
def square(x):
    y = output_tensor(x.shape, x.dtype)
    for i0 in range(x.shape[0]):
        y[i0] = y[i0] * y[i0]
    return y

# 自定义算子反向实现
def square_grad(x, dout):
    dx = output_tensor(x.shape, x.dtype)
    for i0 in range(x.shape[0]):
        dx[i0] = 2.0 * x[i0]
    for i0 in range(x.shape[0]):
        dx[i0] = dx[i0] * dout[i0]
    return dx

# 反向传播函数
def bprop():
    op = ops.Custom(square_grad, lambda x, _: x, lambda x, _: x, func_type="akg")

    def custom_bprop(x, out, dout):
        dx = op(x, dout)
        return (dx,)

    return custom_bprop

class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        # 定义akg类型的自定义算子，并提供反向传播函数
        self.op = ops.Custom(square, lambda x: x, lambda x: x, bprop=bprop(), func_type="akg")

    def construct(self, x):
        return self.op(x)

if __name__ == "__main__":
    x = np.array([1.0, 4.0, 9.0]).astype(np.float32)
    dx = ms.grad(Net())(ms.Tensor(x))
    print(dx)
```

其中：

- 反向传播函数中使用是的akg类型的自定义算子，算子定义与使用需要分开，即自定义算子在`custom_bprop`函数外面定义，在`custom_bprop`函数内部使用。

执行用例：

```bash
python test_grad.py
```

执行结果：

```text
[ 2.  8. 18.]
```

> 更多示例可参考MindSpore源码中[tests/st/ops/graph_kernel/custom](https://gitee.com/mindspore/mindspore/tree/r2.0.0-alpha/tests/st/ops/graph_kernel/custom)下的用例。
