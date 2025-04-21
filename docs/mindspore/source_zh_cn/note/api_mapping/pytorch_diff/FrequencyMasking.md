# 比较与torchaudio.transforms.FrequencyMasking的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/FrequencyMasking.md)

## torchaudio.transforms.FrequencyMasking

```python
class torchaudio.transforms.FrequencyMasking(freq_mask_param: int, iid_masks: bool = False)
```

更多内容详见[torchaudio.transforms.FrequencyMasking](https://pytorch.org/audio/0.8.0/transforms.html#torchaudio.transforms.FrequencyMasking.html)。

## mindspore.dataset.audio.FrequencyMasking

```python
class mindspore.dataset.audio.FrequencyMasking(iid_masks=False, freq_mask_param=0, mask_start=0, mask_value=0.0)
```

更多内容详见[mindspore.dataset.audio.FrequencyMasking](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset_audio/mindspore.dataset.audio.FrequencyMasking.html#mindspore.dataset.audio.FrequencyMasking)。

## 差异对比

PyTorch：给音频波形施加频域掩码。

MindSpore：给音频波形施加频域掩码。不支持变化的`mask_value`取值。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | freq_mask_param     | freq_mask_param    | - |
|     | 参数2 | iid_masks   | iid_masks   | - |
|     | 参数3 | -   | mask_start   | 添加掩码的起始位置 |
|     | 参数4 | -   | mask_value   | 指定填充掩码值，MindSpore计算时无法再更改 |

## 代码示例

```python
import numpy as np

fake_specgram = np.array([[[0.17274511, 0.85174704, 0.07162686, -0.45436913],
                           [-1.0271876, 0.33526883, 1.7413973, 0.12313101]]]).astype(np.float32)

# PyTorch
import torch
import torchaudio.transforms as T
torch.manual_seed(1)

transformer = T.FrequencyMasking(freq_mask_param=2, iid_masks=True)
torch_result = transformer(torch.from_numpy(fake_specgram), mask_value=0.0)
print(torch_result)
# Out: tensor([[[ 0.0000,  0.0000,  0.0000,  0.0000],
#               [-1.0272,  0.3353,  1.7414,  0.1231]]])

# MindSpore
import mindspore as ms
import mindspore.dataset.audio as audio

ms.dataset.config.set_seed(2)
transformer = audio.FrequencyMasking(freq_mask_param=2, iid_masks=True, mask_start=0, mask_value=0.0)
ms_result = transformer(fake_specgram)
print(ms_result)
# Out: [[[ 0.          0.          0.          0.        ]
#        [-1.0271876   0.33526883  1.7413973   0.12313101]]]
```
