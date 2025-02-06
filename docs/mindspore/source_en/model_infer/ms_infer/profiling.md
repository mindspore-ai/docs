# Model Performance Profiler

[![](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.5.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.5.0/docs/mindspore/source_en/model_infer/ms_infer/profiling.md)

MindSpore provides the profiler API to collect neural network performance data. Currently, it supports the analysis of data related to AI Core operators, AI CPU operators, host CPU operators, memory, device communication, clusters, and more.

Example:

```python
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.train import Model
from mindspore import Profiler

input_data = Tensor(np.random.randint(0, 255, [1, 1, 32, 32]), mindspore.float32)
# Define the network structure of LeNet5. Refer to
# https://gitee.com/mindspore/docs/blob/r2.5.0/docs/mindspore/code/lenet.py
# Init Profiler
# Note that the Profiler should be initialized before model.predict
profiler = Profiler()
model = Model(LeNet5())
result = model.predict(input_data)

# Profiler end
profiler.analyse()

```

The performance profiling method for inference is basically the same as that for training. After collecting the performance data, you can analyze the performance by referring to [Performance Profiling](https://www.mindspore.cn/docs/en/r2.5.0/model_train/optimize/profiler.html). Inference focuses on operator performance analysis, computation workload performance analysis, and timeline analysis.

For details about the API, see [mindspore.Profiler](https://www.mindspore.cn/docs/en/r2.5.0/api_python/mindspore/mindspore.Profiler.html?highlight=profiler#mindspore.Profiler).
