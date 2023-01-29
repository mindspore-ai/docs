# 比较与torch.blackman_window的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/blackman_window.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## torch.blackman_window

```text
torch.blackman_window(
    window_length,
    periodic=True,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False
) -> Tensor
```

更多内容详见[torch.blackman_window](https://pytorch.org/docs/1.8.1/generated/torch.blackman_window.html)。

## mindspore.ops.blackman_window

```text
mindspore.ops.blackman_window(
    window_length,
    periodic=True,
    *,
    dtype=mstype.float32
) -> Tensor
```

更多内容详见[mindspore.ops.blackman_window](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.blackman_window.html)。

## 差异对比

PyTorch：返回size与window_length相同的布莱克曼窗，periodic参数确定返回窗口是否会删除对称窗口的最后一个重复值。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，精度稍有差异。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数 | 参数1 |window_length | window_length | - |
| | 参数2 | periodic | periodic | - |
|  | 参数3 | dtype        | dtype | - |
| | 参数4 | layout | - | 不涉及 |
| | 参数5 | device | - | 不涉及 |
| | 参数6 | requires_grad | - | MindSpore无此参数，默认支持反向求导 |

### 代码示例1

```python
# PyTorch
import torch

torch_output = torch.blackman_window(12, periodic=True)
print(torch_output.numpy())
# [-2.9802322e-08 2.6987284e-02 1.3000000e-01 3.4000000e-01
#  6.3000000e-01 8.9301264e-01 1.0000000e+00 8.9301258e-01
#  6.2999994e-01 3.3999997e-01 1.3000003e-01 2.6987225e-02]

# MindSpore
import mindspore
from mindspore import Tensor

window_length = Tensor(12, mindspore.int32)
ms_output = mindspore.ops.blackman_window(window_length, periodic=True)
print(ms_output.asnumpy())
# [-1.3877788e-17 2.6987297e-02 1.3000000e-01 3.4000000e-01
#  6.3000000e-01 8.9301270e-01 1.0000000e+00 8.9301270e-01
#  6.3000000e-01 3.4000000e-01 1.3000000e-01 2.6987297e-02]
```
