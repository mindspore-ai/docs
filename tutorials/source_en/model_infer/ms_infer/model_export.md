# Model Export

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/model_infer/ms_infer/model_export.md)

MindSpore provides a unified intermediate representation (IR) for training and inference. You can use the export API to directly save a model as MindIR. (Currently, only graph mode is supported.)

```python
import mindspore as ms
import numpy as np
from mindspore import Tensor

# Define the network structure of LeNet5. Refer to
# https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
net = LeNet5()
input_tensor = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32))
ms.export(net, input_tensor, file_name='lenet', file_format='MINDIR')

```

For details about the API, see [mindspore.export](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.export.html?highlight=export#mindspore.export).

The model export result is provided for MindSpore Lite. For details about how to use the result, see [Lite Inference Overview](../lite_infer/overview.md).
