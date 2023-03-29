# 比较与torch.randint_like的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/randint_like.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>

## torch.randint_like

```text
torch.randint_like(input, low=0, high, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format)
```

更多内容详见[torch.randint_like](https://pytorch.org/docs/1.8.1/generated/torch.randint_like.html#torch.randint_like)。

## mindspore.ops.randint_like

```text
mindspore.ops.randint_like(input, low, high, *, dtype=None, seed=None)
```

更多内容详见[mindspore.ops.randint_like](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.randint_like.html#mindspore.ops.randint_like)。

## 差异对比

PyTorch：`low` 为可选输入，默认值为0。

MindSpore：`low` 为必选输入，无默认值。

| 分类  | 子类  | PyTorch       | MindSpore | 差异                                   |
|-----|-----|---------------|-----------|--------------------------------------|
| 参数  | 参数1 | input         | input     | 无差异                                  |
| 参数  | 参数2 | low           | low       | PyTorch中 `low` 有默认值0，MindSpore不存在默认值 |
| 参数  | 参数3 | high          | high      | 无差异                                  |
| 参数  | 参数4 | dtype         | dtype     | 无差异                                  |
| 参数  | 参数5 | layout        | -         | 通用差异                                 |
| 参数  | 参数6 | device        | -         | 通用差异                                 |
| 参数  | 参数7 | requires_grad | -         | 通用差异                                 |
| 参数  | 参数8 | memory_format | -         | 通用差异                                 |
| 参数  | 参数9 | -             | seed      | 通用差异                                 |

### 代码示例

```python
# PyTorch
import torch

# PyTorch 无需传入low的值，相当于MindSpore中low=0。
x = torch.tensor([[2, 3], [1, 2]], dtype=torch.float32)
y = torch.randint_like(x, 10)
print(y.shape)
# (2, 2)

# MindSpore
import mindspore

# MindSpore 必须将torch中low的默认值值（此处为0），作为输入传入。
x = mindspore.Tensor([[2, 3], [1, 2]], mindspore.float32)
x = mindspore.ops.randint_like(x, 0, 10, )
print(x.shape)
# (2, 2)
```