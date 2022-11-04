# 比较与torch.BartlettWindow的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/BartlettWindow.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.bartlett_window

```text
torch.blackman_window(
    window_length,
    periodic=True,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False
) -> Tensor
```

更多内容详见 [torch.bartlett_window](https://pytorch.org/docs/1.8.1/generated/torch.bartlett_window.html)。

## mindspore.ops.BartlettWindow

```text
mindspore.ops.BartlettWindow(
    window_length,
    periodic=True,
    dtype=mstype.float32
) -> Tensor
```

更多内容详见 [mindspore.ops.bartlett_window](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.blackman_window.html)。

## 差异对比

PyTorch：返回size与window_length相同的巴特兰窗，periodic参数确定返回窗口是否会删除对称窗口的最后一个重复值。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，精度稍有差异。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 输入 | 单输入 |window_length | window_length | - |
|参数 | 参数1 | periodic | periodic | - |
|  | 参数2 | dtype        | dtype | - |
| | 参数3 | layout | - | 功能一致，MindSpore无此参数 |
| | 参数4 | device | - | 功能一致，MindSpore无此参数 |
| | 参数5 | requires_grad | - | 功能一致，MindSpore无此参数 |

### 代码示例1

```python
# PyTorch
import torch

torch_output = torch.blackman_window(5, periodic=True)
print(torch_output.numpy())
#[0.         0.4        0.8        0.79999995 0.39999998]

# MindSpore
import mindspore

window_length = Tensor(5, mindspore.int32)
bartlett_window = ops.BartlettWindow(periodic=True)
ms_output = bartlett_window(window_length)
print(ms_output.asnumpy())
#[0.  0.4 0.8 0.8 0.4]
```
