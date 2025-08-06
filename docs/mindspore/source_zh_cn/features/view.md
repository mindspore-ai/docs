# Tensor View 机制

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.0/docs/mindspore/source_zh_cn/features/view.md)

View操作是指创建一个新的张量，该张量与原始张量共享相同的数据存储(data storage)，但具有不同的形状或排列方式。换句话说，view操作不会复制数据，而是通过不同的视角来持有现有的数据。

核心特点：

- 内存共享：View操作创建的新张量与原始张量共享底层数据存储。
- 零拷贝：不进行数据复制，避免内存分配开销。
- 形状变换：可以改变张量的形状而不改变数据内容。

Tensor.view()方法：

```python

import mindspore

# 创建原始张量
x = mindspore.tensor([[1, 2, 3], [4, 5, 6]])
print("原始张量:", x)
print("原始形状:", x.shape)

# 使用view改变形状
y = x.view(-1)  # 展平为1D张量
print("展平后:", y)
print("新形状:", y.shape)

# 改为其他形状
z = x.view(3, 2)  # 改为3x2张量
print("改后:", z)
print("新形状:", z.shape)
```

Tensor.view_as()方法：

```python
import mindspore

x = mindspore.tensor([[1, 2, 3], [4, 5, 6]])
target = mindspore.tensor([[0, 0], [0, 0], [0, 0]])

# 将x改为与target相同的形状
y = x.view_as(target)
print("修改结果:", y)
print("目标形状:", target.shape)
print("结果形状:", y.shape)
```

需要注意的是：

1. view操作要求张量在内存中是连续的，如果不连续，需要先调用contiguous()方法

    ```python
    import mindspore
    from mindspore import ops

    x = mindspore.tensor([[1, 2, 3], [4, 5, 6]])
    y = mindspore.mint.transpose(x, (1, 0))  # 创建非连续张量

    # 检查连续性
    print("y是否连续:", y.is_contiguous())

    # 使用contiguous()确保连续性
    z = y.contiguous().view(-1)
    print("z是否连续:", z.is_contiguous())
    ```

2. view操作要求新形状的元素总数与原始张量相同

    ```python
    import mindspore

    x = mindspore.tensor([1, 2, 3, 4, 5, 6])
    print("原始张量元素数:", x.numel())

    # 正确：6 = 2 * 3
    y = x.view(2, 3)
    print("改为2x3:", y)

    # 错误：6 ≠ 2 * 4
    try:
        z = x.view(2, 4)
    except RuntimeError as e:
        print("形状不匹配错误:", e)
    ```

view与reshape区别：

- view操作：

    - 严格要求连续性: View操作要求张量在内存中必须是连续的。
    - 失败机制: 如果张量不连续，view操作会抛出错误。
    - 解决方案: 需要先调用contiguous()方法。

- reshape操作：

    - 灵活处理: reshape操作更灵活，不要求张量必须连续。
    - 自动处理: 如果张量不连续，reshape会自动创建新拷贝。
    - 始终成功: 只要形状匹配，reshape操作总是能成功。