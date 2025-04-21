# 比较与torchaudio.transforms.Resample的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Resample.md)

## torchaudio.transforms.Resample

```python
class torchaudio.transforms.Resample(orig_freq: int = 16000, new_freq: int = 16000, resampling_method: str = 'sinc_interpolation')
```

更多内容详见[torchaudio.transforms.Resample](https://pytorch.org/audio/0.8.0/transforms.html#torchaudio.transforms.Resample.html)。

## mindspore.dataset.audio.Resample

```python
class mindspore.dataset.audio.Resample(orig_freq=16000, new_freq=16000, resample_method=ResampleMethod.SINC_INTERPOLATION,
                                       lowpass_filter_width=6, rolloff=0.99, beta=None)
```

更多内容详见[mindspore.dataset.audio.Resample](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset_audio/mindspore.dataset.audio.Resample.html#mindspore.dataset.audio.Resample)。

## 差异对比

PyTorch：将信号从一个频率重采样至另一个频率。

MindSpore：将信号从一个频率重采样至另一个频率。支持额外的信号滤波处理。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | orig_freq     | orig_freq    | - |
|     | 参数2 | new_freq    | new_freq   | - |
|     | 参数3 | resampling_method    | resample_method     | - |
|     | 参数4 | -   | lowpass_filter_width    | 滤波器的带宽 |
|     | 参数5 | -   | rolloff     | 滤波器的滚降频率 |
|     | 参数6 | -   | beta     | Kaiser窗的形状参数 |

## 代码示例

```python
import numpy as np

fake_input = np.array([[[[-0.2197528, 0.3821656]]],
                      [[[0.57418776, 0.46741104]]],
                      [[[0.76986176, -0.5793846]]]]).astype(np.float32)

# PyTorch
import torch
import torchaudio.transforms as T

transformer = T.Resample(orig_freq=16000, new_freq=24000)
torch_result = transformer(torch.from_numpy(fake_input))
print(torch_result)
# Out: tensor([[[[-0.2140,  0.2226,  0.3510]]],
#              [[[ 0.5728,  0.6145,  0.2789]]],
#              [[[ 0.7568, -0.1601, -0.6101]]]])

# MindSpore
import mindspore.dataset.audio as audio

transformer = audio.Resample(orig_freq=16000, new_freq=24000)
ms_result = transformer(fake_input)
print(ms_result)
# Out: [[[[-0.21398525  0.22255361  0.35099414]]]
#       [[[ 0.5728122   0.614469    0.2788692 ]]]
#       [[[ 0.75675076 -0.16008556 -0.61005235]]]]
```
