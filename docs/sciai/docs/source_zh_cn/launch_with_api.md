# 调用API启动模型

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/docs/sciai/docs/source_zh_cn/launch_with_api.md)&nbsp;&nbsp;

MindSpore SciAI为用户提供了高阶API接口`AutoModel`。借助`AutoModel`，用户可以通过一行代码完成模型的实例化。

用户可以通过`AutoModel`的接口进行模型参数更新，并启动训练或评估。

## 使用AutoModel获取模型

用户可以使用`AutoModel.from_pretrained`接口获取已支持的网络模型。

这里使用Conservatice Physics-Informed Neural Networks (CPINNs) 作为教学案例。CPINNs模型相关代码请参考[链接](https://gitee.com/mindspore/mindscience/SciAI/sciai/model/cpinns)。

更多关于该模型的信息，请参考[论文](https://www.sciencedirect.com/science/article/abs/pii/S0045782520302127)。

```python
from sciai.model import AutoModel

# 获取cpinns网络模型
model = AutoModel.from_pretrained("cpinns")
```

## 使用AutoModel训练、微调模型

用户可以使用`AutoModel.train`实现模型的训练，并且在执行训练之前，
使用`AutoModel.update_config`调整训练参数，或是加载`.ckpt`文件实现模型微调。
接口`AutoModel.update_config`所接受的可选参数依赖于模型类型，

```python
from sciai.model import AutoModel

# 获取cpinns网络模型
model = AutoModel.from_pretrained("cpinns")
# （可选）加载参数ckpt文件，使用已有参数进行模型初始化
model.update_config(load_ckpt=True, load_ckpt_path="./checkpoints/your_file.ckpt", epochs=500)
# 使用默认参数训练网络，生成的图片、数据与日志将保存至用户的执行目录中
model.train()
```

## 使用AutoModel评估模型

用户可以使用`AutoModel.evaluate`评估训练结果。

该接口将默认加载SciAI模型库中提供的`.ckpt`文件用于评估，用户也可以调用`model.update_config`接口自定义加载的文件。

```python
from sciai.model import AutoModel

# 获取cpinns网络模型
model = AutoModel.from_pretrained("cpinns")
# （可选）自定义加载ckpt文件
model.update_config(load_ckpt=True, load_ckpt_path="./checkpoints/your_file.ckpt")
# 评估网络模型
model.evaluate()
```
