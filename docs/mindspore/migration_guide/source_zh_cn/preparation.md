# 准备工作

<!-- TOC -->

- [准备工作](#准备工作)
    - [概述](#概述)
    - [安装MindSpore](#安装mindspore)
        - [使用pip安装](#使用pip安装)
        - [使用源码安装](#使用源码安装)
        - [设置环境变量（仅用于Ascend环境）](#设置环境变量仅用于ascend环境)
        - [MindSpore验证](#mindspore验证)
    - [知识准备](#知识准备)
        - [MindSpore编程指南](#mindspore编程指南)
        - [ModelZoo和Hub](#modelzoo和hub)
        - [云上训练](#云上训练)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/preparation.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

在进行网络开发或网络迁移工作之前，首先需要安装MindSpore，并掌握机器学习的相关知识。用户可以选择购买《深度学习与MindSpore实践》一书来了解相关知识，通过访问[MindSpore官网](https://www.mindspore.cn)了解MindSpore的用法。

## 安装MindSpore

MindSpore支持在Ascend、CPU、GPU环境安装并使用，支持EulerOS-arm、CentOS-arm、CentOS-x86、Ubuntu-arm、Ubuntu-x86、Windows-X86操作系统，可访问[MindSpore安装页面](https://www.mindspore.cn/install)下载MindSpore安装包，并参考该网站指导完成安装。

### 使用pip安装

从官网下载MindSpore安装包后，得到`mindspore_{device}-{version}-{python_version}-linux_{arch}.whl`文件，请使用pip安装。

```bash
pip install mindspore_{device}-{version}-{python_version}-linux_{arch}.whl
```

- `{python_version}`表示用户的Python版本，Python版本为3.7.5时，`{python_version}`应写为`cp37-cp37m`。Python版本为3.9.0时，则写为`cp39-cp39`。

若环境已安装旧版本MindSpore，当前需要更新MindSpore，请在安装前卸载旧版本。

### 使用源码安装

访问[MindSpore代码仓](https://gitee.com/mindspore/mindspore)，使用`git clone https://gitee.com/mindspore/mindspore.git`下载MindSpore源码，源码根目录下的`build.sh`文件提供了多个备选参数，用于选择定制MindSpore服务，一般通过以下命令编译MindSpore。

```bash
cd mindspore
bash build.sh -e cpu -j{thread_num} # cpu环境
bash build.sh -e ascend -j{thread_num} # Ascend环境
bash build.sh -e gpu -j{thread_num} # gpu环境
```

编译成功后，在`output`目录下会生成MindSpore安装包，然后使用 **pip安装** 或 **将当前目录添加到PYTHONPATH** 的方式使用源码编译的结果。

> 使用pip安装的优点在于能够快速上手，方便快捷。
>
> 使用源码安装可以定制MindSpore服务，并可以切换到任意commit_id编译并使用MindSpore。

### 设置环境变量（仅用于Ascend环境）

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}                 # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                # Python library that TBE implementation depends on
```

### MindSpore验证

若以下命令能正常执行成功并退出，说明安装成功。

对于CPU环境：

```python
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="CPU")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))
```

对于Ascend环境：

```python
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="Ascend")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))
```

对于GPU环境：

```python
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="GPU")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))
```

## 知识准备

### MindSpore编程指南

用户可以通过参考[MindSpore教程](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/index.html)了解如何使用MindSpore进行训练、调试、调优、推理；也可以通过参考[MindSpore编程指南](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/index.html)了解MindSpore的基本组成和常用编程方法；也可以通过参考[MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)详细了解MindSpore各接口的相关信息，以便于用户能够更好地使用。

### ModelZoo和Hub

[ModelZoo](https://gitee.com/mindspore/models/tree/master)是MindSpore与社区共同提供的深度优化的模型集市，向开发者提供了深度优化的模型，以便于生态中的小伙伴可以方便地基于ModelZoo中的模型进行个性化开发。当前已经覆盖了机器视觉、自然语言处理、语音、推荐系统等多个领域的主流模型。

[mindspore Hub](https://www.mindspore.cn/resources/hub)是存放MindSpore官方或者第三方开发者提供的预训练模型的平台。它向应用开发者提供了简单易用的模型加载和微调API，使得用户可以基于预训练模型进行推理或者微调，并部署到自己的应用中。用户也可以将自己训练好的模型按照指定的步骤[发布模型](https://www.mindspore.cn/hub/docs/zh-CN/master/publish_model.html)到MindSpore Hub中，供其他用户下载和使用。

### 云上训练

ModelArts是华为云提供的面向AI开发者的一站式开发平台，集成了昇腾AI处理器资源池，用户可以在该平台下体验MindSpore。相关文档可参考[云上使用MindSpore](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/use_on_the_cloud.html)和[AI开发平台ModelArts](https://support.huaweicloud.com/wtsnew-modelarts/index.html)。
