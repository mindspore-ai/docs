# Model Performance Profiler

[![](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/model_infer/ms_infer/profiling.md)

MindSpore provides the profiler API to collect neural network performance data. Currently, it supports the analysis of data related to AI Core operators, AI CPU operators, host CPU operators, memory, device communication, clusters, and more.

Example:

```python
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.train import Model

input_data = Tensor(np.random.randint(0, 255, [1, 1, 32, 32]), mindspore.float32)
# Define the network structure of LeNet5. Refer to
# https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
# Init Profiler
# Note that the Profiler should be initialized before model.predict
with mindspore.profiler.profile() as prof:
    model = Model(LeNet5())
    result = model.predict(input_data)
    # Profiler end
    prof.step()

```

The performance profiling method for inference is basically the same as that for training. After collecting the performance data, you can analyze the performance by referring to [Performance Profiling](https://www.mindspore.cn/docs/en/master/model_train/optimize/profiler.html). Inference focuses on operator performance analysis, computation workload performance analysis, and timeline analysis.

For details about the API, see [mindspore.profiler.profile](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.profiler.profile.html).
