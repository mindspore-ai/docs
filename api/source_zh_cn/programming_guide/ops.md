# ops模块

<!-- TOC -->

- [ops模块](#ops模块)
    - [mindspore.ops.operations](#mindsporeopsoperations)
    - [mindspore.ops.functional](#mindsporeopsfunctional)
    - [mindspore.ops.composite](#mindsporeopscomposite)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/api/source_zh_cn/programming_guide/ops.md" target="_blank"><img src="../_static/logo_source.png"></a>

MindSpore的ops模块主要存放算子相关接口，同时包含算子的校验和正反向关联的逻辑。

ops主要包括operations、functional和composite，可通过ops直接获取到这三类算子。  
- operations提供单个的Primtive算子。一个算子对应一个原语，是最小的执行对象，需要实例化之后使用。
- composite提供一些预定义的组合算子，以及复杂的涉及图变换的算子，如`GradOperation`。
- functional提供operations和composite实例化后的对象，简化算子的调用流程。

## mindspore.ops.operations

operations提供了所有的Primitive算子接口，是开放给用户的最低阶算子接口。算子支持情况可查询[算子支持列表](https://www.mindspore.cn/docs/zh-CN/master/operator_list.html#mindspore-ops-operations)。

Primitive算子也称为算子原语，它直接封装了底层的Ascend、GPU、AICPU、CPU等多种算子的具体实现，为用户提供基础算子能力。

Primitive算子接口是构建高阶接口、自动微分、网络模型等能力的基础。

代码样例如下：
```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops.operations as P

input_x = mindspore.Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
input_y = 3.0
pow = P.Pow()
output = pow(input_x, input_y)
print("output =", output)
```

输出如下：
```
output = [ 1.  8. 64.]
```

## mindspore.ops.functional

为了简化没有属性的算子的调用流程，MindSpore提供了一些算子的functional版本。入参要求参考原算子的输入输出要求。算子支持情况可以查询[算子支持列表](https://www.mindspore.cn/docs/zh-CN/master/operator_list.html#mindspore-ops-functional)。

例如`P.Pow`算子，我们提供了functional版本的`F.tensor_pow`算子。

使用functional的代码样例如下：

```python
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.ops import functional as F

input_x = mindspore.Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
input_y = 3.0
output = F.tensor_pow(input_x, input_y)
print("output =", output)
```

输出如下：
```
output = [ 1.  8. 64.]
```

## mindspore.ops.composite

composite提供了一些算子的组合，包括clip_by_value和random相关的一些算子，以及涉及图变换的函数（`GradOperation`、`HyperMap`和`Map`等）。

算子的组合可以直接像一般函数一样使用，例如使用`normal`生成一个随机分布：
```python
from mindspore.common import dtype as mstype
from mindspore.ops import composite as C
from mindspore import Tensor

mean = Tensor(1.0, mstype.float32)
stddev = Tensor(1.0, mstype.float32)
output = C.normal((2, 3), mean, stddev, seed=5)
print("ouput =", output)
```
输出如下：
```
output = [[2.4911082  0.7941146  1.3117087]
 [0.30582333  1.772938  1.525996]]
```

> 以上代码运行于MindSpore的GPU版本。

针对涉及图变换的函数，用户可以使用`MultitypeFuncGraph`定义一组重载的函数，根据不同类型，走到不同实现。

代码样例如下：
```python
import numpy as np
from mindspore.ops.composite import MultitypeFuncGraph
from mindspore import Tensor
from mindspore.ops import functional as F

add = MultitypeFuncGraph('add')
@add.register("Number", "Number")
def add_scalar(x, y):
    return F.scalar_add(x, y)

@add.register("Tensor", "Tensor")
def add_tensor(x, y):
    return F.tensor_add(x, y)

tensor1 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
tensor2 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
print('tensor', add(tensor1, tensor2))
print('scalar', add(1, 2))
```
输出如下：
```
tensor [[2.4, 4.2] 
 [4.4, 6.4]]
scalar 3
```

此外，高阶函数`GradOperation`提供了根据输入的函数，求这个函数对应的梯度函数的方式，详细可以参阅[API文档](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.ops.composite.html#mindspore.ops.composite.GradOperation)。