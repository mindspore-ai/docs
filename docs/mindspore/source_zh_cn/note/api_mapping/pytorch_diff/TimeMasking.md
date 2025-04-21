# 比较与torchaudio.transforms.TimeMasking的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/TimeMasking.md)

## torchaudio.transforms.TimeMasking

```python
class torchaudio.transforms.TimeMasking(time_mask_param: int, iid_masks: bool = False)
```

更多内容详见[torchaudio.transforms.TimeMasking](https://pytorch.org/audio/0.8.0/transforms.html#torchaudio.transforms.TimeMasking.html)。

## mindspore.dataset.audio.TimeMasking

```python
class mindspore.dataset.audio.TimeMasking(iid_masks=False, time_mask_param=0, mask_start=0, mask_value=0.0)
```

更多内容详见[mindspore.dataset.audio.TimeMasking](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset_audio/mindspore.dataset.audio.TimeMasking.html#mindspore.dataset.audio.TimeMasking)。

## 差异对比

PyTorch：给音频波形施加时域掩码。

MindSpore：给音频波形施加时域掩码。不支持变化的`mask_value`取值。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | time_mask_param      | time_mask_param     | - |
|     | 参数2 | iid_masks   | iid_masks   | - |
|     | 参数3 | -   | mask_start   | 添加掩码的起始位置 |
|     | 参数4 | -   | mask_value   | 指定填充掩码值，MindSpore计算时无法再更改 |

## 代码示例

```python
import numpy as np

fake_wav = np.array([[[0.17274511, 0.85174704, 0.07162686, -0.45436913],
                      [-1.0271876, 0.33526883, 1.7413973, 0.12313101]]]).astype(np.float32)

# PyTorch
import torch
import torchaudio.transforms as T
torch.manual_seed(1)

transformer = T.TimeMasking(time_mask_param=2, iid_masks=True)
torch_result = transformer(torch.from_numpy(fake_wav), mask_value=0.0)
print(torch_result)
# Out: tensor([[[ 0.0000,  0.8517,  0.0716, -0.4544],
#               [ 0.0000,  0.3353,  1.7414,  0.1231]]])

# MindSpore
import mindspore as ms
import mindspore.dataset.audio as audio
ms.dataset.config.set_seed(2)

transformer = audio.TimeMasking(time_mask_param=2, iid_masks=True, mask_start=0, mask_value=0.0)
ms_result = transformer(fake_wav)
print(ms_result)
# Out: [[[ 0.          0.85174704  0.07162686 -0.45436913]
#        [ 0.          0.33526883  1.7413973   0.12313101]]]
```
