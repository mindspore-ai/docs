# Tensor View 机制

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/feature/atomgit/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/features/view.md)

## 概述

View操作是指创建一个新的Tensor，该Tensor与原始Tensor共享相同的数据存储(data storage)，但具有不同的形状或排列方式。换句话说，view操作不会复制数据，而是通过不同的视角来持有现有的数据。

核心特点：

- **内存共享**：View操作创建的新Tensor与原始Tensor共享底层数据存储。
- **零拷贝**：不进行数据复制，避免内存分配开销。
- **形状变换**：可以改变Tensor的形状而不改变数据内容。

## Tensor View方法

### Tensor.view()

`Tensor.view()`方法用于快速、高效地改变Tensor的形状。

```python
import mindspore
from mindspore import nn, context

context.set_context(mode=context.GRAPH_MODE, jit_level='O0')

class ViewNet(nn.Cell):
    def construct(self, x):
        # 使用view改变形状
        y = x.view(-1)  # 展平为1D Tensor
        # 改为其他形状
        z = x.view(3, 2)  # 改为3x2 Tensor
        return y, z

# 创建原始Tensor
x = mindspore.tensor([[1, 2, 3], [4, 5, 6]])
net = ViewNet()
y, z = net(x)

print("原始Tensor:", x)
print("原始形状:", x.shape)
print("展平后:", y)
print("新形状:", y.shape)
print("改后:", z)
print("新形状:", z.shape)
```

运行结果如下：

```text
原始Tensor: [[1 2 3]
 [4 5 6]]
原始形状: (2, 3)
展平后: [1 2 3 4 5 6]
新形状: (6,)
改后: [[1 2]
 [3 4]
 [5 6]]
新形状: (3, 2)
```

### Tensor.view_as()

`Tensor.view_as()`方法将一个Tensor的形状调整为与另一个目标Tensor相同的形状。

```python
import mindspore
from mindspore import nn, context

context.set_context(mode=context.GRAPH_MODE, jit_level='O0')

class ViewAsNet(nn.Cell):
    def construct(self, x, target):
        # 将x改为与target相同的形状
        return x.view_as(target)

x = mindspore.tensor([[1, 2, 3], [4, 5, 6]])
target = mindspore.tensor([[0, 0], [0, 0], [0, 0]])

net = ViewAsNet()
y = net(x, target)

print("修改结果:", y)
print("目标形状:", target.shape)
print("结果形状:", y.shape)
```

运行结果如下：

```text
修改结果: [[1 2]
 [3 4]
 [5 6]]
目标形状: (3, 2)
结果形状: (3, 2)
```

## 注意事项

### Tensor连续性

view操作要求Tensor在内存中是连续的。如果输入Tensor不连续，或`view`的输出需要传递给不支持视图（view）输入的算子，则必须先调用`contiguous()`方法确保Tensor连续。

```python
import mindspore

x = mindspore.tensor([[1, 2, 3], [4, 5, 6]])
y = mindspore.mint.transpose(x, 0, 1)  # 创建非连续Tensor

# 检查连续性
print("y是否连续:", y.is_contiguous())

# 使用contiguous()确保连续性
z = y.contiguous().view(-1)
print("z是否连续:", z.is_contiguous())
```

运行结果如下：

```text
y是否连续: False
z是否连续: True
```

### 元素总数一致

view操作要求新形状的元素总数与原始Tensor相同。

```python
import mindspore

x = mindspore.tensor([1, 2, 3, 4, 5, 6])
print("原始Tensor元素数:", x.numel())

# 正确：6 = 2 * 3
y = x.view(2, 3)
print("改为2x3:", y)

# 错误：6 ≠ 2 * 4
try:
    z = x.view(2, 4)
except ValueError as e:
    print("形状不匹配错误:", e)
```

运行结果如下：

```text
原始Tensor元素数: 6
改为2x3: [[1 2 3]
 [4 5 6]]
形状不匹配错误: ValueError: The accumulate of x_shape must be equal to out_shape, but got x_shape: [const vector]{6}, and out_shape: [const vector]{2, 4}
```

## view与reshape区别

- **view操作**：
    - **严格要求连续性**: View操作要求Tensor在内存中必须是连续的。
    - **失败机制**: 如果Tensor不连续，view操作会抛出错误。
    - **解决方案**: 需要先调用`contiguous()`方法。

- **reshape操作**：
    - **灵活处理**: `reshape`操作更灵活，不要求Tensor必须连续。
    - **自动处理**: 如果Tensor不连续，`reshape`会自动创建新拷贝。
    - **始终成功**: 只要形状匹配，`reshape`操作总是能成功。

`reshape`与`view`的一个主要区别在于`reshape`可以处理非连续的Tensor，它会隐式地创建一个新的连续Tensor。而`view`则要求Tensor必须是连续的。

```python
import mindspore
import numpy as np
from mindspore import nn, context

context.set_context(mode=context.GRAPH_MODE, jit_level='O0')

class ReshapeNet(nn.Cell):
    def construct(self, x):
        return x.reshape(3, 4)

class ViewNet(nn.Cell):
    def construct(self, x):
        return x.view(3, 4)

# 创建一个非连续的Tensor
a = mindspore.tensor(np.arange(12).reshape(3, 4))
b = a.transpose(1, 0) # b现在是非连续的
print("b is contiguous:", b.is_contiguous())

# reshape可以成功执行
reshape_net = ReshapeNet()
c = reshape_net(b)
print("reshape success, c shape:", c.shape)

# view会失败
try:
    view_net = ViewNet()
    d = view_net(b)
except RuntimeError as e:
    print("view failed:", e)
```

运行结果如下：

```text
b is contiguous: False
reshape success, c shape: (3, 4)
view failed: The tensor is not contiguous. You can call .contiguous() to get a contiguous tensor.
```

## View与Inplace特性对比

### View特性

- **共享数据**: View算子创建的新Tensor与原始Tensor共享底层数据，不会复制数据。
- **新Tensor对象**: View操作会返回一个新的Tensor对象，但这个新对象指向的是原始数据内存。
- **属性独立**: 新Tensor可以有独立的形状（shape）、步长（stride）和数据类型（dtype）。
- **双向影响**: 修改View Tensor的数据会影响原始Tensor，反之亦然。

### Inplace特性

- **原地修改**: Inplace算子直接在原始Tensor的内存上进行计算并修改数据，不会创建新的Tensor对象。
- **节省内存**: 因为不创建新的Tensor来存储结果，所以可以节省内存。
- **命名约定**: 在MindSpore中，Inplace算子通常以结尾的下划线 `_` 来标识，例如 `add_`。

### 主要区别

| 特性 | View | Inplace |
| :--- | :--- | :--- |
| **返回值** | 返回一个新的Tensor对象 | 返回修改后的原始Tensor对象 |
| **对象数量** | 创建一个新的Tensor对象，与原始对象共享数据 | 不创建新对象，在原始对象上修改 |
| **核心目的** | 高效地以不同“视角”访问数据 | 节省内存，在原数据上直接计算和更新 |

关于更多view inplace特性的用法，请参考下面的文档：
参考[view inplace](https://atomgit.com/mindspore/docs/blob/master/tutorials/source_zh_cn/compile/static_graph.md#view%E5%92%8Cin-place%E5%8A%9F%E8%83%BD)
