# 比较与torchaudio.transforms.SpectralCentroid的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/SpectralCentroid.md)

## torchaudio.transforms.SpectralCentroid

```python
class torchaudio.transforms.SpectralCentroid(sample_rate: int, n_fft: int = 400, win_length: Optional[int] = None,
                                             hop_length: Optional[int] = None, pad: int = 0,
                                             window_fn: Callable[[...], torch.Tensor] = <built-in method hann_window of type object>,
                                             wkwargs: Optional[dict] = None)
```

更多内容详见[torchaudio.transforms.SpectralCentroid](https://pytorch.org/audio/0.8.0/transforms.html#torchaudio.transforms.SpectralCentroid.html)。

## mindspore.dataset.audio.SpectralCentroid

```python
class mindspore.dataset.audio.SpectralCentroid(sample_rate, n_fft=400, win_length=None, hop_length=None,
                                               pad=0, window=WindowType.HANN)
```

更多内容详见[mindspore.dataset.audio.SpectralCentroid](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset_audio/mindspore.dataset.audio.SpectralCentroid.html#mindspore.dataset.audio.SpectralCentroid)。

## 差异对比

PyTorch：计算每个通道沿时间轴的频谱中心。支持自定义窗函数或对窗函数传入不同的配置参数。支持对STFT结果进行幅值规范化。

MindSpore：计算每个通道沿时间轴的频谱中心。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | sample_rate      | sample_rate      | - |
|     | 参数2 | n_fft    |  n_fft  | - |
|     | 参数3 | win_length  | win_length    | - |
|     | 参数4 | hop_length  | hop_length    | - |
|     | 参数5 | pad    | pad   |  |
|     | 参数6 | window_fn   | window     | MindSpore仅支持5种窗函数 |
|     | 参数7 | wkwargs  | -    | 自定义窗函数的入参，MindSpore不支持 |

## 代码示例

```python
import numpy as np

fake_input = np.array([[[1, 1, 2, 2, 3, 3, 4]]]).astype(np.float32)

# PyTorch
import torch
import torchaudio.transforms as T

transformer = T.SpectralCentroid(sample_rate=44100, n_fft=8, window_fn=torch.hann_window)
torch_result = transformer(torch.from_numpy(fake_input))
print(torch_result)
# Out: tensor([[[4436.1182, 3768.7986]]])

# MindSpore
import mindspore.dataset.audio as audio

transformer = audio.SpectralCentroid(sample_rate=44100, n_fft=8, window=audio.WindowType.HANN)
ms_result = transformer(fake_input)
print(ms_result)
# Out: [[[[4436.117  3768.7979]]]]
```
