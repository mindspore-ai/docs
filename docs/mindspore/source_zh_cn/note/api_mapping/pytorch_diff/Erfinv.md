# 比较与torch.erfinv的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Erfinv.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.erfinv

```text
torch.erfinv(input) -> Tensor
```

更多内容详见 [torch.erfinv](https://pytorch.org/docs/1.8.1/generated/torch.erfinv.html)。

## mindspore.ops.Erfinv

```text
class mindspore.ops.Erfinv()(input_x) -> Tensor
```

更多内容详见 [mindspore.ops.Erfinv](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Erfinv.html)。

## 差异对比

PyTorch：计算输入Tensor的逆误差函数。逆误差函数在范围(-1,1)，公式为：erfinv（erf（x））= x。

MindSpore：MindSpore此API实现功能与PyTorch一致。差异在于MindSpore需要先创建一个实例对象，再进行调用，且调用时参数名为`input_x`，和pytorch仅参数名不同。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| :-: | :-: | :-: | :-: |:-:|
|参数 | 参数1 | input | input_x（实例化后） |功能一致，MindSpore实例化前无此参数，实例化后仅参数名不同 |

### 代码示例

> 两API实现功能一致，MindSpore在使用时需要先实例化后再进行调用。

```python
# PyTorch
import torch
from torch import tensor

input = tensor([[0, 0.5, -1],[1, -0.5, 2]], dtype=torch.float32)
out = torch.erfinv(input).numpy()
print(out)
# [[ 0.          0.47693628        -inf]
#  [        inf -0.47693628         nan]]

# MindSpore
import mindspore
from mindspore import Tensor

input_x = Tensor([[0, 0.5, -1],[1, -0.5, 2]], mindspore.float32)
erfinv = mindspore.ops.Erfinv()
output = erfinv(input_x)
print(output)
# [[ 0.          0.47693628        -inf]
#  [        inf -0.47693628         nan]]
```
