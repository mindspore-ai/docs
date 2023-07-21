# 比较与torch.Tensor.expand的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/expand.md)

## torch.Tensor.expand

```text
torch.Tensor.expand(*sizes) -> Tensor
```

更多内容详见[torch.Tensor.expand](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.expand)。

## mindspore.Tensor.broadcast_to

```text
mindspore.Tensor.broadcast_to(shape) -> Tensor
```

更多内容详见[mindspore.Tensor.broadcast_to](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/Tensor/mindspore.Tensor.broadcast_to.html)。

## 差异对比

MindSpore此API功能与PyTorch一致，参数支持的数据类型有差异。

PyTorch：`sizes` 为广播后的目标shape，其类型可以为 ``torch.Size`` 或者为由 ``int`` 构成的序列。

MindSpore：`shape` 为广播后的目标shape，其类型可以为 ``tuple[int]`` 。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                    |
| ---- | ----- | ------- | --------- | --------------------------------------- |
| 参数 | 参数1 | *sizes | shape | 二者参数名不同，均表示广播后的目标shape。 `sizes` 的类型可以为 ``torch.Size`` 或者为由 ``int`` 构成的序列，`shape` 的类型可以为 ``tuple[int]`` 。|

### 代码示例

```python
# PyTorch
import torch

x = torch.tensor([1, 2, 3])
output = x.expand(3, 3)
print(output)
print(value)
# tensor([[1, 2, 3],
#         [1, 2, 3],
#         [1, 2, 3]])

# MindSpore
import mindspore
from mindspore import Tensor

shape = (3, 3)
x = Tensor(np.array([1, 2, 3]))
output = x.broadcast_to(shape)
print(output)
# [[1 2 3]
#  [1 2 3]
#  [1 2 3]]
```
