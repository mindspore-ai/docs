# 比较与torchaudio.transforms.Spectrogram的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Spectrogram.md)

## torchaudio.transforms.Spectrogram

```python
class torchaudio.transforms.Spectrogram(n_fft: int = 400, win_length: Optional[int] = None, hop_length: Optional[int] = None,
                                        pad: int = 0, window_fn: Callable[[...], torch.Tensor] = <built-in method hann_window of type object>,
                                        power: Optional[float] = 2.0, normalized: bool = False, wkwargs: Optional[dict] = None,
                                        center: bool = True, pad_mode: str = 'reflect', onesided: bool = True)
```

更多内容详见[torchaudio.transforms.Spectrogram](https://pytorch.org/audio/0.8.0/transforms.html#torchaudio.transforms.Spectrogram.html)。

## mindspore.dataset.audio.Spectrogram

```python
class mindspore.dataset.audio.Spectrogram(n_fft=400, win_length=None, hop_length=None,
                                          pad=0, window=WindowType.HANN,
                                          power=2.0, normalized=False,
                                          center=True, pad_mode=BorderType.REFLECT, onesided=True)
```

更多内容详见[mindspore.dataset.audio.Spectrogram](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset_audio/mindspore.dataset.audio.Spectrogram.html#mindspore.dataset.audio.Spectrogram)。

## 差异对比

PyTorch：从音频信号创建其频谱。支持自定义窗函数或对窗函数传入不同的配置参数。支持对STFT结果进行幅值规范化。

MindSpore：从音频信号创建其频谱。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | n_fft      | n_fft      | - |
|     | 参数2 | win_length     | win_length    | - |
|     | 参数3 | hop_length   | hop_length    | - |
|     | 参数4 | pad   | pad    | - |
|     | 参数5 | window_fn    | window    | MindSpore仅支持5种窗函数 |
|     | 参数6 | power  | power    | - |
|     | 参数7 | normalized  | normalized   | - |
|     | 参数8 | wkwargs   | -     | 自定义窗函数的入参，MindSpore不支持 |
|     | 参数9 | center    | center     | - |
|     | 参数10 | pad_mode    | pad_mode     | - |
|     | 参数11 | onesided   | onesided     | - |

## 代码示例

```python
import numpy as np

fake_input = np.array([[[1, 1, 2, 2, 3, 3, 4]]]).astype(np.float32)

# PyTorch
import torch
import torchaudio.transforms as T

transformer = T.Spectrogram(n_fft=8, window_fn=torch.hamming_window)
torch_result = transformer(torch.from_numpy(fake_input))
print(torch_result)
# Out: tensor([[[[3.5874e+01, 1.3237e+02],
#                [1.8943e+00, 3.2839e+01],
#                [8.4640e-01, 2.1553e-01],
#                [2.0643e-02, 2.4623e-01],
#                [6.5697e-01, 1.2876e+00]]]])

# MindSpore
import mindspore.dataset.audio as audio

transformer = audio.Spectrogram(n_fft=8, window=audio.WindowType.HAMMING)
ms_result = transformer(fake_input)
print(ms_result)
# Out: [[[[3.5873653e+01 1.3237122e+02]
#         [1.8942689e+00 3.2838711e+01]
#         [8.4640014e-01 2.1552797e-01]
#         [2.0642618e-02 2.4623220e-01]
#         [6.5697211e-01 1.2876146e+00]]]]
```
