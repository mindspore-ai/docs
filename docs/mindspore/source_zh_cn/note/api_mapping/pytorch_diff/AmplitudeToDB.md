# 比较与torchaudio.transforms.AmplitudeToDB的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/AmplitudeToDB.md)

## torchaudio.transforms.AmplitudeToDB

```python
class torchaudio.transforms.AmplitudeToDB(stype: str = 'power', top_db: Optional[float] = None)
```

更多内容详见[torchaudio.transforms.AmplitudeToDB](https://pytorch.org/audio/0.8.0/transforms.html#torchaudio.transforms.AmplitudeToDB.html)。

## mindspore.dataset.audio.AmplitudeToDB

```python
class mindspore.dataset.audio.AmplitudeToDB(stype=ScaleType.POWER, ref_value=1.0, amin=1e-10, top_db=80.0)
```

更多内容详见[mindspore.dataset.audio.AmplitudeToDB](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset_audio/mindspore.dataset.audio.AmplitudeToDB.html#mindspore.dataset.audio.AmplitudeToDB)。

## 差异对比

PyTorch：将输入音频从振幅/功率标度转换为分贝标度。

MindSpore：将输入音频从振幅/功率标度转换为分贝标度，支持指定波形取值下界和分贝系数计算参考值。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | stype    | stype    | - |
|     | 参数2 | top_db   | top_db   | 默认值不同 |
|     | 参数3 | -   | ref_value   | 分贝系数计算参考值，PyTorch固定使用1e-10 |
|     | 参数4 | -   | amin   | 波形取值下界，PyTorch固定使用1.0 |

## 代码示例

```python
import numpy as np

fake_input = np.array([[[[-0.2197528, 0.3821656]]],
                       [[[0.57418776, 0.46741104]]],
                       [[[0.76986176, -0.5793846]]]]).astype(np.float32)

# PyTorch
import torch
import torchaudio.transforms as T

transformer = T.AmplitudeToDB(stype="power", top_db=80.)
torch_result = transformer(torch.from_numpy(fake_input))
print(torch_result)
# Out: tensor([[[[-84.1775,  -4.1775]]],
#        [[[ -2.4095,  -3.3030]]],
#        [[[ -1.1359, -81.1359]]]])

# MindSpore
import mindspore.dataset.audio as audio

transformer = audio.AmplitudeToDB(stype=audio.ScaleType.POWER, top_db=80., ref_value=1.0, amin=1e-10)
ms_result = transformer(fake_input)
print(ms_result)
# Out: [[[[-84.17748    -4.177484 ]]]
#      [[[ -2.4094608  -3.3030105]]]
#      [[[ -1.1358725 -81.13587  ]]]]
```
