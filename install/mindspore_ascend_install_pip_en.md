# Installing MindSpore in Ascend 910 by pip

<!-- TOC -->

- [Installing MindSpore in Ascend 910 by pip](#installing-mindspore-in-ascend-910-by-pip)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Installing MindSpore](#installing-mindspore)
    - [Configuring Environment Variables](#configuring-environment-variables)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/install/mindspore_ascend_install_pip_en.md)

This document describes how to quickly install MindSpore in a Linux system with an Ascend 910 environment by pip.

## System Environment Information Confirmation

- Confirm that the 64-bit operating system is installed and the [glibc](https://www.gnu.org/software/libc/)>=2.17, where Ubuntu 18.04/CentOS 7.6/EulerOS 2.8/OpenEuler 20.03/KylinV10 SP1 are verified.
- Ensure that right version [GCC 7.3.0](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz) is installed.
- Confirm that [gmp 6.1.2](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz) is installed.
- Confirm that Python 3.7.5 is installed.
    - If you didn't install Python or you have installed other versions, please download the Python 3.7.5 64-bit from [Python](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) or [Huaweicloud](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz) to install.
- Confirm that the Ascend 910 AI processor software package ([Ascend Data Center Solution 21.0.2](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-data-center-solution-pid-251167910/software/252504581?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252309113%7C251167910)) are installed.
    - For the installation of software package,  please refer to the [Product Document](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-data-center-solution-pid-251167910).
    - The software packages include Driver and Firmware and CANN 5.0.2.
        - [Driver and Firmware A800-9000 1.0.11.SCP001 ARM platform](https://support.huawei.com/enterprise/zh/ascend-computing/a800-9000-pid-250702818/software/253882155?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C250702818)and[Driver and Firmware A800-9010 1.0.11.SCP001 x86 platform](https://support.huawei.com/enterprise/zh/ascend-computing/a800-9010-pid-250702809/software/253882161?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C250702809)
        - [CANN 5.0.2.1](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/253944991?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373)
    - Confirm that the current user has the right to access the installation path `/usr/local/Ascend`of Ascend 910 AI processor software package, If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located. For the specific configuration, please refer to the software package instruction document.
    - Install the .whl package provided in Ascend 910 AI processor software package. The .whl package is released with the software package. After software package is upgraded, reinstall the .whl package.

        ```bash
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-{version}-py3-none-any.whl
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-{version}-py3-none-any.whl
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-{version}-py3-none-any.whl
        ```

    - If the Ascend 910 AI processor software package is upgraded, the .whl package also needs to be reinstalled, first uninstall the original installation package, and then refer to the above command to reinstall.

        ```bash
        pip uninstall te topi hccl -y
        ```

## Installing MindSpore

It is recommended to refer to [Version List](https://www.mindspore.cn/versions/en) to perform SHA-256 integrity verification, and then execute the following command to install MindSpore after the verification is consistent.

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/ascend/{arch}/mindspore_ascend-{version}-cp37-cp37m-linux_{arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/r1.3/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.3/requirements.txt).
- `{version}` denotes the version of MindSpore. For example, when you are installing MindSpore 1.1.0, `{version}` should be 1.1.0.  
- `{arch}` denotes the system architecture. For example, the Linux system you are using is x86 architecture 64-bit, `{arch}` should be `x86_64`. If the system is ARM architecture 64-bit, then it should be `aarch64`.

## Configuring Environment Variables

**If Ascend 910 AI processor software is installed in a non-default path**, after MindSpore is installed, export runtime-related environment variables. `/usr/local/Ascend` in the following command `LOCAL_ASCEND=/usr/local/Ascend` denotes the installation path of the software package, please replace it as your actual installation path.

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}                 # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}
# Python library that TBE implementation depends on
```

## Installation Verification

i:

```bash
python -c "import mindspore;mindspore.run_check()"
```

- The outputs should be the same as:

```text
mindspore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

It means MindSpore has been installed successfully.

ii:

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

- The outputs should be the same as:

```text
[[[ 2.  2.  2.  2.],
 [ 2.  2.  2.  2.],
 [ 2.  2.  2.  2.]],

 [[ 2.  2.  2.  2.],
 [ 2.  2.  2.  2.],
 [ 2.  2.  2.  2.]],

 [[ 2.  2.  2.  2.],
 [ 2.  2.  2.  2.],
 [ 2.  2.  2.  2.]]]
```

It means MindSpore has been installed successfully.

## Version Update

Using the following command if you need to update the MindSpore version:

```bash
pip install --upgrade mindspore-ascend
```

