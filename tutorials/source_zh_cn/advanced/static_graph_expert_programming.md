# 静态图高级编程技巧

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/advanced/static_graph_expert_programming.md)

本篇章介绍一些常用的静态图优化的高级编程技巧，这些技巧能够有效地提高静态图的编译效率以及执行效率，并使程序运行地更加稳定。有关静态图编译的基础介绍，请见[使用静态图加速](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/accelerate_with_static_graph.html)。

## 如何优化编译性能

### 使用lazy_inline装饰器

神经网络模型的编译过程往往采用默认inline的方式，把层级的代码表达最终展开成一张扁平的计算图，一方面寻求最大的编译优化机会，另一方面也可以简化自动微分以及执行的逻辑。inline后形成的计算图包含了所有的计算节点，可以在更大的范围内进行优化，比如常量折叠、节点融合、并行分析等，也可以更好地实现内存分配，减少内存申请和性能开销。虽然inline优化对于运行期性能提升帮助非常大，但过度inline也带来了编译期的负担。例如随着计算图节点数量膨胀，执行pass的耗时也在急剧增长。

为了减轻inline对编译性能带来的损耗，对于重复调用相同计算单元的场景（典型的场景是在for循环中调用同一个Cell类的不同实例），我们提供了Lazy Inline机制来减少编译时间。

#### 大模型pipeline并行场景

在大模型场景中，编译耗时问题尤为突出，一是大模型的模型结构层次深，节点数多；二是大模型在训练时，由于启用pipeline并行，导致模型规模和节点数进一步加大，如果原来图的规模是O，那开启pipeline并行，单节点图的规模变为(O/X)*Y，其中X为pipeline的stage数量，Y为micro batch的数量。以盘古13B网络为例，计算图中计算节点数量达到13.5万个，单次编译时长可接近3小时。

我们观察到类似盘古的大模型网络结构，是由多层layer组成的，在开启pipeline并行时，各个micro batch的layer层结构是完全一样的。当开启pipeline并行时，`PipelineCell`使用for循环的方式来多次调用相同结构的layer，代码如下所示：

```python
from mindspore import nn

class PipelineCell(nn.Cell):
    def __init__(self, network, micro_size):
        ...
        self.network = network
        self.micro_size = micro_size
        ...

    def construct(self, ...):
        ...
        for i in range(self.micro_size):
            output = self.network(...)
        ...
```

如果我们把循环体看作被频繁调用的子图，通过把它标记为Lazy Inline，告知编译器推迟inline处理，那么就可以在编译的大部分阶段大幅度减少计算图节点数量，从而获得性能收益。例如上面的代码，可以保留`network`实例的子图结构，不inline或者不提前inline。对此，我们提供了`@lazy_inline`装饰器来实现延迟inline。

以Pangu_alpha网络为例，`PipelineCell`函数体中处理的`network`为`PanGUAlphaWithLoss`类的实例，为实现延迟inline，我们需要对`PanGUAlphaWithLoss`类的`__init__`函数加上`@lazy_inline`装饰器，以标记`PanGUAlphaWithLoss`类的子图结构需要被保留下来，不做inline或者延迟inline。如下所示：

```python
from mindspore import nn
from mindspore.common import lazy_inline

class PanGUAlphaWithLoss(nn.Cell):
    @lazy_inline
    def __init__(self, ...):
        ...

    def construct(self, ...):
```

> 完整代码可以参考：[Pangu_alpha](https://gitee.com/mindspore/models/tree/master/official/nlp/Pangu_alpha)

还是以盘古13B网络为例，应用Lazy Inline方案后，计算图编译规模从13万+节点下降到2万+个节点，编译时间从3个小时下降到20分钟。

#### 更加泛化的一般场景

`@lazy_inline`是`Cell::__init__`的装饰器，它会以`__init__`的所有参数生成Cell的`cell_init_args`属性值，`cell_init_args`值相同表明Cell类名和初始化参数值是一样的。而对于相同Cell类的实例，它们的weights还可能是不一样的，因此对于用`construct(self, x)`定义的网络结构，在实际编译时我们可以转换为`construct(x, self.cell_init_args, self.trainable_parameters())`。对于同一个Cell类的不同实例，如果`cell_init_args`是相同的，那么这两个实例可以复用同一个网络结构，如下所示：

```python
def construct(self, x)
    reuse_construct(x, self.trainable_parameters())
```

引入可复用计算图后，具有相同`cell_init_args`的Cell实例只需编译解析一次。所以对于更加泛化的调用同一个Cell类的不同实例的场景，只要`cell_init_args`是相同的，我们都可以加上`@lazy_inline`装饰器来加速编译。例如GPT网络：

```python
from mindspore import nn
from mindspore.common import lazy_inline

class Block(nn.Cell):
    @lazy_inline
    def __init__(self, config):
        ...

    def construct(self, x, attention_mask, layer_past):
        ...

class GPT_Model(nn.Cell):
    def __init__(self, config):
        ...
        for i in range(config.num_layers):
            self.blocks.append(Block(config))
            ...
        self.num_layers = config.num_layers

    def construct(self, input_ids, input_mask, layer_past):
        ...
        present_layer = ()
        for i in range(self.num_layers):
            hidden_states, present = self.blocks[i](...)
            present_layer = present_layer + (present,)
        ...
```

> 完整代码可以参考：[GPT](https://gitee.com/mindspore/models/tree/master/official/nlp/GPT)

GPT的网络结构由多层`Block`类的不同实例构成，这些`Block`的初始化参数都是同一个`config`，所以加上`@lazy_inline`装饰器后，这些`Block`实例都可以复用同一个网络结构，而且在大部分的编译阶段都不进行inline，从而可以大幅度减少编译时间。

#### 使用步骤

1. 如上面的例子，在网络脚本中，往需要延迟inline和复用子图结构的Cell类的`__init__`函数加上`@lazy_inline`装饰器。
2. 执行训练脚本前，需要设置环境变量MS_DEV_CELL_REUSE的值为1或者2来开启Lazy Inline功能，两个级别的含义为：
   - MS_DEV_CELL_REUSE=1，表示开启Lazy Inline功能，但是后端只有执行序级别的inline。目前在Ascend 910B上只支持该级别。
   - MS_DEV_CELL_REUSE=2，表示开启Lazy Inline功能，后端在执行序优化和内存复用前做Inline，具有更好的内存优化效果。

#### 使用限制

1. Cell 是以Cell的类名和`__init__`参数值生成Cell实例标识的，这是基于`__init__`的参数确定Cell 的所有属性，以及`construct`构图开始时的Cell属性和`__init__`执行完的属性一致为假设前提，因此Cell与构图有关的属性，在`__init__`执行完后不能进行更改。例如：

   ```python
   from mindspore import nn
   from mindspore.common import lazy_inline

   class Block(nn.Cell):
       @lazy_inline
       def __init__(self, ...):
           self.x = 0
           ...

       def construct(self, ...):
           if self.x == 0:
               ...
           else:
               ...
           ...

   class Model(nn.Cell):
       def __init__(self, ...):
           ...
           self.num_layers = 10
           for i in range(self.num_layers):
               self.blocks.append(Block(...)) # 此处Block进行初始化
               ...
           self.blocks[0].x = 1               # 此处在Block初始化后修改Block的属性，会导致该Block无法复用同一份子图

       def construct(self, ...):
           ...
           for i in range(self.num_layers):
               res = self.blocks[i](...)
           ...
   ```

   如上代码所示，网络Model中的某个`Block`实例，它的属性`x`在该实例初始化后被修改了，那么这个`Block`实例就无法准确复用同一个子图结构了。

2. 一个Cell类的网络结构包含多个Cell_X类的实例，同时每个Cell_X类的网络结构又包含多个Cell_Y的实例的场景，如果往Cell_X和Cell_Y类的`__init__`函数上都加上`@lazy_inline`，那么只有最外层的Cell_X实例的网络结构被编译成可复用的计算图且被延迟inline，内层的Cell_Y实例的计算图还是会被inline。例如：

   ```python
   from mindspore import nn
   from mindspore.common import lazy_inline

   class InnerBlock(nn.Cell):
       @lazy_inline             # InnerBlock不会被延迟inline
       def __init__(self, ...):
           ...

       def construct(self, ...):
           ...

   class OuterBlock(nn.Cell):
       @lazy_inline             # OuterBlock将会被延迟inline
       def __init__(self, ...):
           ...
           self.num_layers = 10
           for i in range(self.num_layers):
               self.blocks.append(InnerBlock(...))

       def construct(self, ...):
           ...
           for i in range(self.num_layers):
               res = self.blocks[i](...)
           ...

   class Model(nn.Cell):
       def __init__(self, ...):
           ...
           self.num_layers = 10
           for i in range(self.num_layers):
               self.blocks.append(OuterBlock(...))

       def construct(self, ...):
           ...
           for i in range(self.num_layers):
               res = self.blocks[i](...)
           ...
   ```

   后续有计划支持这种多层级的Lazy Inline机制。

### 使用HyperMap

使用场景：使用HyperMap替换for循环来优化编译性能。

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

使用场景：在进行训练或者推理时，如果编译依赖的文件未作任何变更，通过使用编译缓存来缩短编译时间。

编译缓存的本质是存储了网络模型的编译中间过程文件，当网络模型不变时，生产的编译中间过程文件也是一样的，因此可以复用上一次编程产生的中间过程文件。

通过设置context中的[enable_compile_cache](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_context.html?highlight=enable_compile_cache)或环境变量[MS_COMPILER_CACHE_ENABLE](https://www.mindspore.cn/docs/zh-CN/master/note/env_var_list.html?highlight=MS_COMPILER_CACHE_ENABLE)，可以指定是否保存和加载编译缓存，前者优先级更高。

通过设置context中的[compile_cache_path](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_context.html?highlight=compile_cache_path)或环境变量[MS_COMPILER_CACHE_PATH](https://www.mindspore.cn/docs/zh-CN/master/note/env_var_list.html?highlight=MS_COMPILER_CACHE_PATH)，可以指定MindSpore编译缓存目录，用于存储图和算子编译过程生成的缓存文件，前者优先级更高。

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

可以看到，关闭编译缓存时，第2次执行样例比第1次耗时少一些，这是因为算子编译缓存是默认开启的，第2次执行样例能够利用前一次的算子编译缓存。

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

使用场景：使用`@jit_class`装饰器修饰自定义类，提高执行性能。jit_class应用于静态图模式，在动态图模式下，`@jit_class`会被忽略，不影响动态图模式的执行逻辑。

#### jit_class的介绍

用户在网络脚本中定义一个类时，可以写成继承于`Cell`的类、自定义类、`@jit_class`修饰的类，它们的用法和区别如下：

- 继承于Cell的类

  Cell是MindSpore中神经网络的基本构成单元，模型或者神经网络层应当继承该类。静态图模式下，使用`Cell`类并且在`construct`函数中编写执行代码，此时`construct`函数的代码会被编译成静态计算图。

- 自定义类

  定义自定义类后，可以对类进行实例化、调用类对象的属性和方法，请参考[自定义类的使用](https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html#支持自定义类的使用)。相比于`Cell`的类定义，自定义类更贴近用户调用Python类的使用习惯。自定义类在静态图模式下的实现方式与`Cell`不同，例如，调用自定义类对象的函数方法时，其函数方法中的代码不会被编译成静态计算图，而是通过Python解释器进行解释执行。

- `@jit_class`修饰的类

  为了兼顾用户的Python使用习惯和静态图编译带来的性能优势，提供了`@jit_class`装饰器。给自定义类修饰`@jit_class`装饰器后，该类的函数代码会被编译成静态计算图，基于图优化、静态图整图下沉等技术，编译器可以针对计算图进行全局的优化，从而获得较好的执行性能。

在静态图模式下，通过使用`@jit_class`修饰自定义类，用户可以创建、调用该类的实例，并且可以获取其属性和方法。

#### jit_class装饰器的使用

jit_class装饰器仅支持修饰自定义类，不支持修饰继承于`Cell`的类。

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class InnerNet:
    value = ms.Tensor(np.array([1, 2, 3]))

class Net(nn.Cell):
    def construct(self):
        return InnerNet().value

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
out = net()
print(out)
```

运行结果如下：

```text
[1 2 3]
```

如果jit_class修饰继承于`Cell`的类，将会报错。

```python
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class Net(nn.Cell):
    def construct(self, x):
        return x

ms.set_context(mode=ms.GRAPH_MODE)
x = ms.Tensor(1)
net = Net()
net(x)
```

报错信息如下：

```text
TypeError: Decorator jit_class is used for user-defined classes and cannot be used for nn.Cell: Net<>.
```

jit_class支持自定义类嵌套使用、自定义类与`Cell`嵌套使用的场景。需要注意的是，类继承时，如果父类使用了jit_class，子类也会具有jit_class的能力。

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class Inner:
    def __init__(self):
        self.value = ms.Tensor(np.array([1, 2, 3]))

@ms.jit_class
class InnerNet:
    def __init__(self):
        self.inner = Inner()

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.inner_net = InnerNet()

    def construct(self):
        out = self.inner_net.inner.value
        return out

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
out = net()
print(out)
```

运行结果如下：

```text
[1 2 3]
```

#### 获取类的属性和方法

支持通过类名或类实例调用属性和方法。

```python
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class InnerNet:
    def __init__(self, val):
        self.number = val

    def act(self, x, y):
        return self.number * (x + y)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.inner_net = InnerNet(2)

    def construct(self, x, y):
        return self.inner_net.number + self.inner_net.act(x, y)

ms.set_context(mode=ms.GRAPH_MODE)
x = ms.Tensor(2, dtype=ms.int32)
y = ms.Tensor(3, dtype=ms.int32)
net = Net()
out = net(x, y)
print(out)
```

运行结果如下：

```text
12
```

#### 创建类的实例

对于将会被编译成静态计算图的函数，如`Cell`的`construct`函数、`@jit`修饰的函数或前两者调用的子函数，如果需要在函数内创建`@jit_class`所修饰的类的实例，参数要求为常量。

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class InnerNet:
    def __init__(self, val):
        self.number = val + 3

class Net(nn.Cell):
    def construct(self):
        net = InnerNet(2)
        return net.number

ms.set_context(mode=ms.GRAPH_MODE)
net = Net()
out = net()
print(out)
```

运行结果如下：

```text
5
```

#### 调用类的实例

调用`@jit_class`所修饰的类的实例时，将会调用该类的`__call__`函数方法。

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class InnerNet:
    def __init__(self, number):
        self.number = number

    def __call__(self, x, y):
        return self.number * (x + y)

class Net(nn.Cell):
    def construct(self, x, y):
        net = InnerNet(2)
        out = net(x, y)
        return out

ms.set_context(mode=ms.GRAPH_MODE)
x = ms.Tensor(2, dtype=ms.int32)
y = ms.Tensor(3, dtype=ms.int32)
net = Net()
out = net(x, y)
print(out)
```

运行结果如下：

```text
10
```

如果该类没有定义`__call__`函数，将会报错提示。

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

@ms.jit_class
class InnerNet:
    def __init__(self, number):
        self.number = number

class Net(nn.Cell):
    def construct(self, x, y):
        net = InnerNet(2)
        out = net(x, y)
        return out

ms.set_context(mode=ms.GRAPH_MODE)
x = ms.Tensor(2, dtype=ms.int32)
y = ms.Tensor(3, dtype=ms.int32)
net = Net()
out = net(x, y)
print(out)
```

报错信息如下：

```text
RumtimeError: MsClassObject: 'InnerNet' has no __call__ function, please check the code.
```

### 使用select算子

使用场景：`Select`算子来替代if控制流语句，减少静态图子图生成，提高执行性能（也可以提高编译性能）。

编写网络时，会经常使用到if语句，如果if语句的条件是变量条件，每个if语句都会产生额外的子图。在静态图模式下，子图数量越多，编译耗时越久，因此部分场景可以通过`Select`算子等价替换if语句来优化编译性能。

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

## 优化冗余显存拷贝操作

在函数式编程中，通过参数和返回值之外的渠道和外界存在数据交换的函数，被称为非纯函数，被认为是存在副作用的。在MindSpore框架内部，针对副作用的问题会插入Load算子，该算子属于虚拟算子，不需要在后端执行，不占用显存，仅用于表示需要读取全局变量的值。在图模式下，需要编译完整个图之后才将图中的各个算子下发到后端执行，使用Load算子多次读取全局变量，而不是多次使用真实算子多次保存全局变量的值，这样可以减少显存的消耗。

但是，全局变量的值可能是变化的，如果没有真实算子保存值，某些场景下会存在精度问题。针对这种情况，MindSpore框架内部会插入真实算子，占用一定的显存来保存全局变量的值，从而避免出现精度问题。

我们提供了MS_DEV_SIDE_EFFECT_LOAD_ELIM开关来优化显存占用的程度，即设置export MS_DEV_SIDE_EFFECT_LOAD_ELIM=0/1/2/3。

- 当将MS_DEV_SIDE_EFFECT_LOAD_ELIM设置为0时，表示对框架内部的Load算子都插入真实算子，即占用显存最多，保证网络精度没有问题。
- 当将MS_DEV_SIDE_EFFECT_LOAD_ELIM设置为1或者没有设置值时（即默认模式），表示对框架内部的Load算子可能出现精度问题的场景保守地插入真实算子，保证网络精度没有问题。
- 当将MS_DEV_SIDE_EFFECT_LOAD_ELIM设置为2，在损耗一定编译性能的前提下，尽量少地插入真实算子，优化显存较多，且保证网络精度没有问题。
- 当将MS_DEV_SIDE_EFFECT_LOAD_ELIM设置为3，不插入真实算子，不保证网络的精度，显存消耗最少。