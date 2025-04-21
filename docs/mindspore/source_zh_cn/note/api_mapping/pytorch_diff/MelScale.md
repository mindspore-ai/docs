# 比较与torchaudio.transforms.MelScale的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/MelScale.md)

## torchaudio.transforms.MelScale

```python
class torchaudio.transforms.MelScale(n_mels: int = 128, sample_rate: int = 16000, f_min: float = 0.0, f_max: Optional[float] = None,
                                     n_stft: Optional[int] = None, norm: Optional[str] = None)
```

更多内容详见[torchaudio.transforms.MelScale](https://pytorch.org/audio/0.8.0/transforms.html#torchaudio.transforms.MelScale.html)。

## mindspore.dataset.audio.MelScale

```python
class mindspore.dataset.audio.MelScale(n_mels=128, sample_rate=16000, f_min=0.0, f_max=None,
                                       n_stft=201, norm=NormType.NONE, mel_type=MelType.HTK)
```

更多内容详见[mindspore.dataset.audio.MelScale](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset_audio/mindspore.dataset.audio.MelScale.html#mindspore.dataset.audio.MelScale)。

## 差异对比

PyTorch：将普通STFT转换为梅尔尺度的STFT。

MindSpore：将普通STFT转换为梅尔尺度的STFT，支持指定梅尔频谱的尺度。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | n_mels     | n_mels     | - |
|     | 参数2 | sample_rate     | sample_rate     | - |
|     | 参数4 | f_min  | f_min    | - |
|     | 参数5 | f_max   | f_max     | - |
|     | 参数6 | n_stft    | n_stft     | - |
|     | 参数10 | norm   | norm     | - |
|     | 参数11 | -   | mel_type      | 要使用的Mel尺度 |

## 代码示例

```python
import numpy as np

fake_input = np.array([[1., 1.],
                       [0., 0.],
                       [1., 1.],
                       [1., 1.]]).astype(np.float32)

# PyTorch
import torch
import torchaudio.transforms as T

transformer = T.MelScale(n_stft=4, n_mels=2)
torch_result = transformer(torch.from_numpy(fake_input))
print(torch_result)
# Out: tensor([[0.0000, 0.0000],
#              [0.5394, 0.5394]])

# MindSpore
import mindspore.dataset.audio as audio

transformer = audio.MelScale(n_stft=4, n_mels=2)
ms_result = transformer(fake_input)
print(ms_result)
# Out: [[0.         0.        ]
#       [0.53936154 0.53936154]]
```
