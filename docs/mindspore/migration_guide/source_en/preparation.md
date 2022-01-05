# Preparation

Translator: [Misaka19998](https://gitee.com/Misaka19998/docs/tree/master)

<!-- TOC -->

- [Preparation](#preparation)
    - [Overview](#overview)
    - [Installing MindSpore](#installing-mindspore)
        - [Installing by pip](#installing-by-pip)
        - [Installing by Source Code](#installing-by-source-code)
        - [Configuring Environment Variables (only for Ascend)](#configuring-environment-variables-(only-for-ascend))
        - [MindSpore Verification](#mindspore-verification)
    - [Knowledge Preparation](#knowledge-preparation)
        - [MindSpore Programming Guide](#mindspore-programming-guide)
        - [ModelZoo and Hub](#modelzoo-and-hub)
        - [Training on the Cloud](#training-on-the-cloud)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/preparation.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

Before developing or migrating networks, you need to install MindSpore and learn machine learning knowledge. Users have a choice to buy *Introduction to Deep learning with MindSpore* to learn related knowledge and visit [MindSpore Official website](https://www.mindspore.cn/en) to know how to use MindSpore.

## Installing MindSpore

MindSpore supports installing and running in EulerOS-arm, CentOS-arm, CentOS-x86, Ubuntu-arm, Ubuntu-x86 or Windows-x86 system with Ascend, CPU or GPU environments. You can download packages form [MindSpore Installationl](https://www.mindspore.cn/install/en), and install MindSpore by following the tutorials.

### Installing by pip

After downloading the package `mindspore_{device}-{version}-{python_version}-{os}_{arch}.whl` from the official website, please install it by pip.

```bash
pip install mindspore_{device}-{version}-{python_version}-{os}_{arch}.whl
```

- `{python_version}` spcecifies the python version for which MindSpore is built. If you wish to use Python3.7.5,`{python_version}` should be `cp37-cp37m`. If Python3.9.0 is used, it should be `cp39-cp39`.
- `{os` spcecifies the operation system, such as `linux` or `win`.
- `{arch}` spcecifies the system architecture, such as `x86_64` or `amd64`.

If an old version of MindSpore exists in the environment, please uninstall it before updating a new version of MindSpore.

### Installing by Source Code

You can visit [Repository of Mindspore](https://gitee.com/mindspore/mindspore) and download the source code by `git clone https://gitee.com/mindspore/mindspore.git`. A file  `build.sh` in root directory provides several optional parameters, to choose and customize the MindSpore service. The following code is for compiling MindSpore.

```bash
cd mindspore
bash build.sh -e cpu -j{thread_num} # cpu
bash build.sh -e ascend -j{thread_num} # ascend
bash build.sh -e gpu -j{thread_num} # gpu
```

After successfully compilation, MindSpore install package will be created in `output` directory. Then you can **install it by pip** or **add current directory to PYTHONPATH** to use this package.

> Installing by pip is fast and convenient to start.
>
> Installing by source code can customize MindSpore service and change to any commit_id to compile and run MindSpore.

### Configuring Environment Variables (only for Ascend)

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

### Mindspore Verification

MindSpore is installed successfully if you can run the following code and exit properly.

For CPU:

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

For Ascend:

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

For GPU:

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

## Knowledge Preparation

### MindSpore Programming Guide

Users can read [MindSpore Tutorial](https://www.mindspore.cn/docs/programming_guide/en/master/index.html) to learn how to train, debug, optimize and infer by MindSpore, and read [MindSpore Programming Guide](https://www.mindspore.cn/docs/programming_guide/en/master/index.html) to know the fundamental parts and programming methods of MindSpore. Users can also see detailed MindSpore interfaces by referring to [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html).

### ModelZoo and Hub

[ModelZoo](https://gitee.com/mindspore/models/tree/master) is a model market of MindSpore and community, which provides deeply-optimized models to developers. In order that the users of MindSpore will have individual development conveniently based on models in ModelZoo. Currently, there are major models in several fields, like computer vision, natural language processing, audio and recommender systems.

[mindspore Hub](https://www.mindspore.cn/resources/hub/en) is a platform to save pretrained model of official MindSpore or third party developers. It provides some simple and useful APIs for developers to load and finetune models, so that users can infer or tune models based on pretrained models and deploy models to their applications. Users is able to follow some steps to [publish model](https://www.mindspore.cn/hub/docs/en/master/publish_model.html) to MindSpore Hub,for other developers to download and use.

### Training on the Cloud

ModelArts is a one-stop development platform for AI developers, which contains Ascend resource pool. Users can experience MindSpore in this platform and read related document [MindSpore use_on_the_cloud](https://www.mindspore.cn/docs/programming_guide/en/master/use_on_the_cloud.html) and [AI Platform ModelArts](https://support.huaweicloud.com/intl/en-us/wtsnew-modelarts/index.html).
