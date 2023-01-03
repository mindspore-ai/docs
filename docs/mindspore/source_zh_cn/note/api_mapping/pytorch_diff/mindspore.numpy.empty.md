# 比较与torch.empty的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/mindspore.numpy.empty.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.empty

```text
torch.empty(
    *size,
    *,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False
    memory_format=torch.contiguous_format
) -> Tensor
```

更多内容详见[torch.empty](https://pytorch.org/docs/1.8.1/generated/torch.empty.html)。

## mindspore.numpy.empty

```text
mindspore.numpy.empty(shape, dtype=mstype.float32) -> Tensor
```

更多内容详见[mindspore.numpy.empty](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/numpy/mindspore.numpy.empty.html)。

## 差异对比

PyTorch：返回一个未初始化张量，张量的形状由size定义。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，但dtype参数默认值不同。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | size | shape |功能一致，参数名不同 |
| | 参数2 | out           | -         | 不涉及 |
|  | 参数3 | dtype         | dtype     | 功能一致，默认值不同 |
| | 参数4 | layout | - | 不涉及 |
| | 参数5 | device | - | 不涉及 |
| | 参数6 | requires_grad | - | MindSpore无此参数，默认支持反向求导 |
| | 参数7 | pin_memory | - | 不涉及 |
| | 参数8 | memory_format | - | 不涉及 |

### 代码示例1

> 对于参数dtype，PyTorch默认值为None，输出类型为torch.float32，MindSpore默认值为mstype.float32

```python
# PyTorch
import torch

torch_output = torch.empty(2, 3)
print(list(torch_output.shape))
# [2, 3]

# MindSpore
import mindspore

ms_output = mindspore.numpy.empty((2, 3))
print(ms_output.shape)
# [2, 3]
```
