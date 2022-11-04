# 比较与torch.blackman_window的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/BlackmanWindow.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.blackman_window

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

更多内容详见 [torch.blackman_window](https://pytorch.org/docs/1.8.1/generated/torch.blackman_window.html)。

## mindspore.ops.BlackmanWindow

```text
mindspore.ops.BlackmanWindow(
    window_length,
    periodic=True,
    dtype=mstype.float32
) -> Tensor
```

更多内容详见 [mindspore.ops.blackman_window](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.blackman_window.html)。

## 差异对比

PyTorch：返回size与window_length相同的布莱克曼窗，periodic参数确定返回窗口是否会删除对称窗口的最后一个重复值。

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

torch_output = torch.blackman_window(12, periodic=True)
print(torch_output.numpy())
#[-2.9802322e-08  4.0212840e-02  2.0077014e-01  5.0978714e-01
#  8.4922981e-01  1.0000000e+00  8.4922981e-01  5.0978720e-01
#  2.0077008e-01  4.0212780e-02]

# MindSpore
import mindspore

window_length = Tensor(10, mindspore.int32)
blackman_window = ops.BlackmanWindow(periodic=True)
ms_output = blackman_window(window_length)
print(ms_output.asnumpy())
#[0.         0.04021286 0.20077014 0.50978714 0.8492299  1.
# 0.8492299  0.50978714 0.20077014 0.04021286]
```
