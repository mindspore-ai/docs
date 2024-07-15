# 比较与torchaudio.transforms.MelSpectrogram的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/MelSpectrogram.md)

## torchaudio.transforms.MelSpectrogram

```python
class torchaudio.transforms.MelSpectrogram(sample_rate: int = 16000, n_fft: int = 400, win_length: Optional[int] = None,
                                           hop_length: Optional[int] = None, f_min: float = 0.0, f_max: Optional[float] = None,
                                           pad: int = 0, n_mels: int = 128, window_fn: Callable[[...], torch.Tensor] = <built-in method hann_window of type object>,
                                           power: Optional[float] = 2.0, normalized: bool = False, wkwargs: Optional[dict] = None,
                                           center: bool = True, pad_mode: str = 'reflect', onesided: bool = True, norm: Optional[str] = None)
```

更多内容详见[torchaudio.transforms.MelSpectrogram](https://pytorch.org/audio/0.8.0/transforms.html#torchaudio.transforms.MelSpectrogram.html)。

## mindspore.dataset.audio.MelSpectrogram

```python
class mindspore.dataset.audio.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=None,
                                             hop_length=None, f_min=0.0, f_max=None,
                                             pad=0, n_mels=128, window=WindowType.HANN, power=2.0, normalized=False,
                                             center=True, pad_mode=BorderType.REFLECT, onesided=True, norm=NormType.NONE, mel_scale=MelType.HTK)
```

更多内容详见[mindspore.dataset.audio.MelSpectrogram](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset_audio/mindspore.dataset.audio.MelSpectrogram.html#mindspore.dataset.audio.MelSpectrogram)。

## 差异对比

PyTorch：计算原始音频信号的梅尔频谱。支持自定义窗函数或对窗函数传入不同的配置参数。支持对STFT结果进行幅值规范化。

MindSpore：计算原始音频信号的梅尔频谱。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | sample_rate     | sample_rate      | - |
|     | 参数2 | win_length     | win_length    | - |
|     | 参数3 | hop_length   | hop_length    | - |
|     | 参数4 | n_fft   | n_fft    | - |
|     | 参数5 | f_min    | f_min   | - |
|     | 参数6 | f_max   | f_max    | - |
|     | 参数7 | pad   | pad  | - |
|     | 参数8 | n_mels    | n_mels     | - |
|     | 参数9 | window_fn    | window      | MindSpore仅支持5种窗函数 |
|     | 参数10 | power    | power     | - |
|     | 参数11 | normalized   | normalized     | - |
|     | 参数12 | wkwargs    | -     | 自定义窗函数的入参，MindSpore不支持 |
|     | 参数13 | center   | center     | - |
|     | 参数14 | pad_mode    | pad_mode     | - |
|     | 参数15 | onesided    | onesided     | - |
|     | 参数16 | norm    | norm     | - |
|     | 参数17 | -    | mel_scale      | 要使用的Mel尺度 |

## 代码示例

```python
import numpy as np

fake_input = np.array([[[1, 1, 2, 2, 3, 3, 4]]]).astype(np.float32)

# PyTorch
import torch
import torchaudio.transforms as T

transformer = T.MelSpectrogram(sample_rate=16000, n_fft=4, win_length=2, hop_length=4, window_fn=torch.hann_window)
torch_result = transformer(torch.from_numpy(fake_input))
print(torch_result)
# Out: tensor([[[[0.0000, 0.0000],
#                ...
#                [0.5235, 4.7117],
#                [0.4765, 4.2883],
#                ...
#                [0.0000, 0.0000]]]])

# MindSpore
import mindspore.dataset.audio as audio

transformer = audio.MelSpectrogram(sample_rate=16000, n_fft=4, win_length=2, hop_length=4, window=audio.WindowType.HANN)
ms_result = transformer(fake_input)
print(ms_result)
# Out: [[[[0.         0.        ]
#         ...
#         [0.52353615 4.7118254 ]
#         [0.47646385 4.2881746 ]
#         ...
#         [0.         0.        ]]]]
```
