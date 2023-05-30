# 比较与torch.randint的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/randint.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>

## torch.randint

```text
torch.randint(low=0, high, size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

更多内容详见[torch.randint](https://pytorch.org/docs/1.8.1/generated/torch.randint.html#torch.randint)。

## mindspore.ops.randint

```text
mindspore.ops.randint(low, high, size, seed=None, *, dtype=None)
```

更多内容详见[mindspore.ops.randint](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.randint.html#mindspore.ops.randint)。

## 差异对比

PyTorch：`low` 为可选输入，默认值为0。

MindSpore：`low` 为必选输入，无默认值。

| 分类  | 子类   | PyTorch       | MindSpore | 差异                                   |
|-----|------|---------------|-----------|--------------------------------------|
| 参数  | 参数1  | low           | low       | PyTorch中 `low` 有默认值0，MindSpore不存在默认值 |
|   | 参数2  | high          | high      | 无差异                                  |
|   | 参数3  | size          | size      | 无差异                                  |
|   | 参数4  | generator     | -         | 通用差异                                 |
|   | 参数5  | out           | -         | 通用差异                                 |
|   | 参数6  | dtype         | dtype     | 无差异                                  |
|   | 参数7  | layout        | -         | 通用差异                                 |
|   | 参数8  | device        | -         | 通用差异                                 |
|   | 参数9  | requires_grad | -         | 通用差异                                 |
|   | 参数10 | -             | seed      | 通用差异                                 |

### 代码示例

```python
# PyTorch
import torch

# PyTorch 无需传入low的值，相当于MindSpore中low=0。
x = torch.randint(10, (3, 3))
print(tuple(x.shape))
# (3, 3)

# MindSpore
import mindspore

# MindSpore 必须将torch中low的默认值（此处为0），作为输入传入。
x = mindspore.ops.randint(0, 10, (3, 3))
print(x.shape)
# (3, 3)
```
