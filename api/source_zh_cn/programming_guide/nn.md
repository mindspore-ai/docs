# nn模块

<a href="https://gitee.com/mindspore/docs/blob/master/api/source_zh_cn/programming_guide/nn.md" target="_blank"><img src="../_static/logo_source.png"></a>

MindSpore的nn模块是Python实现的模型组件，是对低阶API的封装，主要包括各种模型层、损失函数、优化器等。

同时nn也提供了部分与Primitive算子同名的接口，主要作用是对Primitive算子进行进一步封装，为用户提供更友好的API。

代码样例如下：
```python
import numpy as np
from mindspore.common.tensor import Tensor
import mindspore.nn as nn
import mindspore

net = nn.PSNR()
img1 = Tensor(np.random.random((1,3,16,16)), mindspore.float32)
img2 = Tensor(np.random.random((1,3,16,16)), mindspore.float32)
output = net(img1, img2)
print("output = ", output)
```

输出如下：
```
output = [7.6338434]
```

各种模型层、损失函数、优化器等代码样例正在完善中。
