## 准备工作

在进行网络开发或网络迁移工作之前，首先需要安装MindSpore，并掌握机器学习的相关知识。用户可以选择购买《深度学习与MindSpore实践》一书来了解相关知识，或通过访问 [https://www.mindspore.cn](mindspore.cn) 了解MindSpore的用法。

### 安装MindSpore

MindSpore支持在Ascend、CPU、GPU环境安装并使用，支持EulerOS-arm、CentOS-arm、CentOS-x86、Ubuntu-arm，Ubuntu-x86操作系统，可访问[https://www.mindspore.cn/install](https://www.mindspore.cn/install)下载MindSpore安装包，并参考该网站指导安装依赖。

#### 使用pip安装

从官网下载MindSpore安装包成功后，得到mindspore_\{device\}-\{version\}-cp37-cp37m-linux_\{arch\}.whl，使用pip安装。

```python
pip install mindspore_{device}-{version}-cp37-cp37m-linux_{arch}.whl
```

若环境已安装旧版本MindSpore，当前需要更新MindSpore，在安装前请卸载旧版本MindSpore.

#### 使用源码安装

访问 [https://gitee.com/mindspore/mindspore](https://gitee.com/mindspore/mindspore)，使用`git clone https://gitee.com/mindspore/mindspore.git`下载MindSpore源码，build.sh提供了多个备选参数选择定制MindSpore服务，一般通过以下命令编译MindSpore.

```python
cd mindspore
bash build.sh -e cpu -j{thread_num} # cpu环境
bash build.sh -e ascend -j{thread_num} # Ascend环境
bash build.sh -e gpu -j{thread_num} # gpu环境
```

编译成功后，在output目录下会生成MindSpore安装包，然后使用 **pip安装** 或 **将当前目录添加到PYTHONPATH** 的方式使用源码编译的结果。

> 使用pip安装的优点在于能够快速上手，方便快捷
> 使用源码安装可以定制MindSpore服务，并可以切换到任意commit_id编译MindSpore.

#### 设置环境变量（仅用于Ascend环境）

```python
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/add-ons/:${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}                 # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                # Python library that TBE implementation depends on
```

#### MindSpore验证

若以下命令能正常执行成功并退出，说明安装成功。

对于CPU环境:

```python
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="CPU")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.tensor_add(x, y))
```

对于Ascend环境:

```python
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="Ascend")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.tensor_add(x, y))
```

对于GPU环境:

```python
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="GPU")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.tensor_add(x, y))
```

### 知识准备

#### MindSpore编程指南

用户可以通过参考[https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)初步了解如何使用mindspore进行训练，调试，调优；也可以通过参考[https://www.mindspore.cn/doc/programming_guide/zh-CN/master/index.html](https://www.mindspore.cn/doc/programming_guide/zh-CN/master/index.html)了解MindSpore的一些基本特性；也可以通过参考[https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)对MindSpore的具体接口做详细了解，以便于客户能够更好的使用。

#### ModelZoo和Hub

ModelZoo是MindSpore与社区共同提供的深度优化的模型集市，给开发者提供了深度优化的模型，以便于生态中的小伙伴可以方便地基于ModelZoo中的模型进行个性化开发。当前已经覆盖了机器视觉、自然语言处理、语音等多个领域的主流模型。

MindSpore Hub是存放MindSpore官方或者第三方开发者提供的预训练模型的平台。它向应用开发者提供了简单易用的模型加载和微调APIs，使得用户可以基于预训练模型进行推理或者微调，并部署到自己的应用中。用户也可以将自己训练好的模型按照指定的步骤发布到MindSpore Hub中，以供其他用户进行下载和使用。

> **ModelZoo** [https://gitee.com/mindspore/mindspore/tree/master/model_zoo](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)
>
> **Hub** [https://www.mindspore.cn/resources/hub](https://www.mindspore.cn/resources/hub)

#### 云上训练

ModelArts是华为云提供的面向AI开发者的一站式开发平台，集成了昇腾AI处理器资源池，用户可以在该平台下体验MindSpore。

> 云上使用MindSpore [https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/use_on_the_cloud.html](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/use_on_the_cloud.html)
>
> AI开发平台ModelArts [https://support.huaweicloud.com/wtsnew-modelarts/index.html](https://support.huaweicloud.com/wtsnew-modelarts/index.html)
