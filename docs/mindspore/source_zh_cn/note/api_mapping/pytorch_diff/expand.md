# 比较与torch.Tensor.expand的差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/expand.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.Tensor.expand

```text
torch.Tensor.expand(*sizes) -> Tensor
```

更多内容详见[torch.Tensor.expand](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.expand)。

## mindspore.Tensor.expand

```text
mindspore.Tensor.expand(size) -> Tensor
```

更多内容详见[mindspore.Tensor.expand](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/Tensor/mindspore.Tensor.expand.html)。

## 差异对比

MindSpore此API功能与PyTorch一致。

PyTorch：参数 `sizes` 的数据类型是 ``int`` 或 ``torch.Size`` 。

MindSpore：参数 `size` 的数据类型是 ``Tensor`` 。

| 分类  | 子类  |PyTorch    | MindSpore | 差异  |
| :-:   | :-:   | :-:       | :-:       |:-:    |
|参数   | 参数1 | sizes     | size      | PyTorch支持的数据类型是 ``int`` 或 ``torch.Size`` ，MindSpore支持的数据类型是 ``Tensor`` |

### 代码示例

```python
# PyTorch
import torch
import numpy as np
x = torch.tensor(np.array([[1], [2], [3]]), dtype=torch.float32)
size = (3, 4)
y = x.expand(size)
print(y)
# tensor([[1., 1., 1., 1.],
#         [2., 2., 2., 2.],
#         [3., 3., 3., 3.]])

# MindSpore
import mindspore as ms
import numpy as np
x = ms.Tensor(np.array([[1], [2], [3]]), dtype=ms.float32)
size = ms.Tensor(np.array([3,4]), dtype=ms.int32)
y = x.expand(size)
print(y)
# [[1. 1. 1. 1.]
#  [2. 2. 2. 2.]
#  [3. 3. 3. 3.]]
```
