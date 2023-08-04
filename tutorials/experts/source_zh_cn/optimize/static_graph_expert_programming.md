# 静态图高级编程技巧

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/optimize/static_graph_expert_programming.md)

本篇章介绍一些常用的静态图优化的高级编程技巧，这些技巧能够有效地提高静态图的编译效率以及执行效率，并使程序运行地更加稳定。有关静态图编译的基础介绍，请见[使用静态图加速](https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/advanced/accelerate_with_static_graph.md)。

## 如何优化编译性能

### 使用lazy_inline装饰器

文档待补充

### 使用HyperMap

使用场景： 使用HyperMap替换for循环来优化编译性能。

`HyperMap`是一个特殊的类，类对象构造时需要传入映射函数f，调用对象时需要传入f的n个参数序列，更多使用方法见：[HyperMap](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.HyperMap.html)。映射函数f必须是`MultitypeFuncGraph`类型, 可参考[MultitypeFuncGraph](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.MultitypeFuncGraph.html)。在使用for循环批量处理列表元素时，可以通过`HyperMap`等价语义替换来优化网络编译性能。例如：

```python
import time
from mindspore.ops import MultitypeFuncGraph, HyperMap
from mindspore import ops, Tensor
import mindspore as ms

add = MultitypeFuncGraph('add')
@add.register("Tensor", "Tensor")
def add_Tensor(x, y):
    return ops.add(x, y)

add_map = HyperMap(add)
list1 = [Tensor(i) for i in range(200)]
list2 = [Tensor(i) for i in range(200)]
@ms.jit
def hyper_map_net():
    output = add_map(list1, list2)
    return output

start_time = time.time()
output = hyper_map_net()
end_time = time.time()
print("hyper map cost time:", end_time - start_time)

@ms.jit
def for_loop_net():
    out = []
    for i in range(200):
        out.append(i+i)
    return out

start_time = time.time()
for_loop_net()
end_time = time.time()
print("for loop cost time:", end_time - start_time)
```

结果如下（实际耗时与硬件环境有关，以下数据仅供参考）：

```text
hyper map cost time: 0.1894233226776123
for loop cost time: 1.2634551525115967
```

### 使用编译缓存

使用场景：在进行训练或者推理时，如果某个网络结构未作任何变更，通过使用编译缓存来缩短编译时间。

编译缓存的本质是存储了网络模型的编译中间过程文件，当网络模型不变时，生产的编译中间过程文件也是一样的，因此可以复用上一次编程产生的中间过程文件，详细使用方法可参考[设置context](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_context.html?highlight=enable_compile_cache)中的`enable_compile_cache`相关内容。

一个通过使能编译缓存来优化编译性能的代码样例如下：

```python
import time
from mindspore import set_context
from mindspore import dtype
import mindspore as ms

@ms.jit
def func(input_x, input_y):
    output = input_x
    for _ in range(200):
        output = input_x + input_x * input_y + output
    return output

set_context(enable_compile_cache=False)
x = ms.Tensor([1], dtype.float32)
y = ms.Tensor([2], dtype.float32)
start_time = time.time()
out = func(x, y)
end_time = time.time()
print("Disable comile_cache cost time:", end_time - start_time)
```

上述测试样例是关闭编译缓存状态，执行上述测试样例2次，第1次耗时和第2次耗时如下（实际耗时与硬件环境有关，以下数据仅供参考）：

```text
Disable comile_cache cost time: 0.5485098361968994
Disable comile_cache cost time: 0.4614279270172119
```

可以看到，关闭编译缓存时，第1次执行样例与第2次执行样例耗时基本接近。

```python
import time
from mindspore import set_context
from mindspore import dtype
import mindspore as ms

@ms.jit
def func(input_x, input_y):
    output = input_x
    for _ in range(200):
        output = input_x + input_x * input_y + output
    return output

set_context(enable_compile_cache=True, compile_cache_path="my_compile_cache")
x = ms.Tensor([1], dtype.float32)
y = ms.Tensor([2], dtype.float32)
start_time = time.time()
out = func(x, y)
end_time = time.time()
print("Enable comile_cache cost time:", end_time - start_time)
```

上述测试样例是开启编译缓存状态，执行上述测试样例2次，第1次耗时和第2次耗时如下（实际耗时与硬件环境有关，以下数据仅供参考）：

```text
Enable comile_cache cost time: 0.6357541084289551
Enable comile_cache cost time: 0.09379792213439941
```

可以看到，开启编译缓存时，第2次执行样例耗时只有第一次执行耗时的1/7左右。

## 如何优化执行性能

### 使用jit_class

使用场景：定义jit_class代替自定义Class，提高执行性能。

其他内容待补充。

### 使用select算子

使用场景：`Select`算子来替代if控制流语句，减少静态图子图生成，提高执行性能（也可以提高编译性能）。

编写网络时，会经常使用到if语句，如果if语句的条件是变量条件，每个if语句都会产生额外的子图，if语句的使用可参考：[if语句](https://mindspore.cn/tutorials/experts/zh-CN/master/network/control_flow.html#if语句)。在静态图模式下，子图数量越多，编译耗时越久，因此部分场景可以通过`Select`算子等价替换if语句来优化编译性能。

需要注意的是，使用`Select`算子替换if语句会影响网络的运行性能。一方面，`Select`算子会同时执行true分支和false分支，而if语句只执行其一个分支，因此使用if运行耗时相比使用`Select`算子耗时减少；另一方面，`Select`算子性能优于if语句产生的控制流算子，使用if运行耗时相比使用`Select`算子运行耗时增加。综合上述两种因素，最终运行性能变化情况需要结合实际情况判断。一般来讲，当分支中算子数量较少，建议使用`Select`算子；当分支中算子数量较多，建议使用if语句。

一个使用`Select`算子替代if语句来优化编译性能的代码样例如下：

```python
import time
from mindspore import ops
import mindspore as ms

@ms.jit
def if_net(x, y):
    out = 0
    for _ in range(100):
        if x < y:
            x = x - y
        else:
            x = x + y
        out = out + x
    return out

start_time = time.time()
out = if_net(ms.Tensor([0]), ms.Tensor([1]))
end_time = time.time()
print("if net cost time:", end_time - start_time)

@ms.jit
def select_net(x, y):
    out = x
    for _ in range(100):
        cond = x < y
        x = ops.select(cond, x - y, x + y)
        out = out + x
    return out

start_time = time.time()
out = select_net(ms.Tensor([0]), ms.Tensor([1]))
end_time = time.time()
print("select net cost time:", end_time - start_time)
```

上述代码的运行结果如下（实际耗时与硬件环境有关，以下数据仅供参考）：

```text
if net cost time: 1.1603329181671143
select net cost time: 0.483151912689209
```

### 使用Vmap进行批处理

使用场景：在处理无依赖关系的批量数据且相关的算子支持Vmap功能时，可以使用Vmap替代for循环处理批量数据来优化执行性能（也可以提高编译性能）。

MindSpore已支持Vmap特性，Vmap的详细介绍可参考[自动向量化Vmap](https://www.mindspore.cn/tutorials/experts/zh-CN/master/vmap/vmap.html)。

一个使用Vmap替换for循环处理批量数据来优化编译性能的代码样例如下：

```python
import numpy as np
import time
from mindspore import ops, vmap
import mindspore as ms

def hswish_func(x):
    return ops.HSwish()(x)

@ms.jit
def manually_batched(xs):
    output = []
    for i in range(xs.shape[0]):
        output.append(hswish_func(xs[i]))
    return ops.stack(output)

shape = (100, 2)
prop = 100
x_np = (np.random.randn(*shape) * prop).astype(np.float32)
x = ms.Tensor(x_np)
x = ops.sub(x, 0)

start_time = time.time()
output_vmap = vmap(hswish_func, in_axes=(0,))(x)
end_time = time.time()
print("Vmap cost time:", end_time - start_time)

start_time = time.time()
output_manually = manually_batched(x)
end_time = time.time()
print("for loop cost time:", end_time - start_time)
```

代码的运行结果如下（实际耗时与硬件环境有关，以下数据仅供参考）：

```text
Vmap cost time: 0.05766916275024414
for loop cost time: 1.9284062385559082
```

上述样例中，相当于需要批量处理100组Tensor数据，可以看到使用Vmap处理的性能超过使用for循环处理性能的30倍。

## 依赖控制保证执行序

如果函数的运行结果依赖或影响外部状态，我们认为该函数具有副作用，比如函数会改变外部全局变量、函数的结果依赖全局变量的值。如果算子会改变输入参数的值或者算子的输出依赖全局参数的值，我们认为这是带副作用的算子。

根据内存属性和IO状态，将副作用划分为内存副作用和IO副作用。当前内存副作用主要有Assign、优化器算子等等，IO副作用主要有Print算子。详细可以查看算子定义，内存副作用算子在定义中有side_effect_mem属性，IO副作用算子在定义中有side_effect_io属性。

Depend用于处理依赖项操作。在大多数情况下，如果操作符有IO副作用或内存副作用，则将根据用户的语义执行它们，不需要另外使用Depend算子来保证执行顺序。在某些情况下，如果两个运算符A和B没有顺序依赖关系，并且A必须在B之前执行，我们建议使用Depend指定它们的执行顺序。使用方法如下：

```python
a = A(x)
b = B(y)
```

在插入Depend算子后，如下：

```python
a = A(x)
y = Depend(y, a)
b = B(y)
```

值得说明的是，用于浮点数溢出状态检测的一组特殊算子它们存在隐含副作用，但又不属于IO副作用或内存副作用。此外，使用时还有严格的顺序要求，即：在使用NPUClearFloatStatus算子前需要保证NPUAllocFloatStatus已经执行，使用NPUGetFloatStatus算子前需要保证NPUClearFloatStatus已经执行。因为这些算子使用较少，目前的方案是保持它们的定义为无副作用形式，以Depend确保执行顺序。如下：

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, set_context, Tensor
from mindspore import dtype as mstype

set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.alloc_status = ops.NPUAllocFloatStatus()
        self.get_status = ops.NPUGetFloatStatus()
        self.clear_status = ops.NPUClearFloatStatus()

    def construct(self, x):
        init = self.alloc_status()
        clear_status = self.clear_status(init)
        x = ops.Depend()(x, clear_status)
        res = ops.sub(x, ops.neg(x))
        init = ops.Depend()(init, res)
        get_status = self.get_status(init)
        res = ops.Depend()(res, get_status)
        return res

value = 5
data = np.full((2, 3), value, dtype=np.float16)
x = Tensor(data, dtype=mstype.float16)
net = Net()
res = net(x)
print(res)
```

运行以上脚本，可以得到：

```text
[[10. 10. 10.]
 [10. 10. 10.]]
```
