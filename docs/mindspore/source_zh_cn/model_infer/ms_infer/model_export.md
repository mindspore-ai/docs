# 模型导出

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindspore/source_zh_cn/model_infer/ms_infer/model_export.md)

MindSpore提供了训练和推理统一的中间表示（Intermediate Representation, IR）。可使用export接口直接将模型保存为MindIR（当前仅支持严格图模式）。

```python
import mindspore as ms
import numpy as np
from mindspore import Tensor

# Define the network structure of LeNet5. Refer to
# https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindspore/code/lenet.py
net = LeNet5()
input_tensor = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32))
ms.export(net, input_tensor, file_name='lenet', file_format='MINDIR')

```

详细接口参考链接：[mindspore.export](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/mindspore/mindspore.export.html?highlight=export#mindspore.export)。

模型导出结果提供给MindSpore Lite使用，使用方式可参考[MindSpore Lite推理](../lite_infer/overview.md)。
