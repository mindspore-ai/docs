# Installing MindSpore in Ascend by Source Code

<!-- TOC -->

- [Installing MindSpore in Ascend by Source Code](#installing-mindspore-in-ascend-by-source-code)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Downloading Source Code from Code Repository](#downloading-source-code-from-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Configuring Environment Variables](#configuring-environment-variables)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)
    - [Installing MindInsight](#installing-mindinsight)
    - [Installing MindArmour](#installing-mindarmour)
    - [Installing MindSpore Hub](#installing-mindspore-hub)

<!-- /TOC -->

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/install/mindspore_ascend_install_source_en.md)

This document describes how to quickly install MindSpore in a Linux system with an Ascend 910 environment by source code.

## System Environment Information Confirmation

- Confirm that Ubuntu 18.04/CentOS 7.6/EulerOS 2.8 is installed with 64-bit operating system.
- Confirm that [GCC 7.3.0](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz) is installed.
- Confirm that [gmp 6.1.2](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz) is installed.
- Confirm that [Python 3.7.5](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) is installed.
- Confirm that [OpenSSL 1.1.1 or later](https://github.com/openssl/openssl.git) is installed.
    - Set system variable `export OPENSSL_ROOT_DIR="OpenSSL installation directory"` after installation.
- Confirm that [CMake 3.18.3 or later](https://cmake.org/download/) is installed.
    - Add the path where the executable file `cmake` stores to the environment variable PATH.
- Confirm that [patch 2.5 or later](http://ftp.gnu.org/gnu/patch/) is installed.
    - Add the path where the executable file `patch` stores to the environment variable PATH.
- Confirm that [wheel 0.32.0 or later](https://pypi.org/project/wheel/) is installed.
- Confirm that the Ascend 910 AI processor software package (Atlas Data Center Solution V100R020C10：[A800-9000 1.0.8 (aarch64)](https://support.huawei.com/enterprise/zh/ascend-computing/a800-9000-pid-250702818/software/252069004?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C250702818), [A800-9010 1.0.8 (x86_64)](https://support.huawei.com/enterprise/zh/ascend-computing/a800-9010-pid-250702809/software/252062130?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C250702809), [CANN V100R020C10](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/251174283?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373)) are installed.
- Confirm that the current user has the right to access the installation path `/usr/local/Ascend`of Ascend 910 AI processor software package, If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located. For the specific configuration, please refer to the software package instruction document.
    - Install the .whl package provided in Ascend 910 AI processor software package. The .whl package is released with the software package. After software package is upgraded, reinstall the .whl package.

        ```bash
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-{version}-py3-none-any.whl
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-{version}-py3-none-any.whl
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-{version}-py3-none-any.whl
        ```

- Confirm that the git tool is installed.

    If not, for Ubuntu users, use the following command to install it:

    ```bash
    apt-get install git
    ```

    If not, for EulerOS and CentOS users, use the following command to install it:

    ```bash
    yum install git
    ```

## Downloading Source Code from Code Repository

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.0
```

## Compiling MindSpore

Run the following command in the root directory of the source code to compile MindSpore:

```bash
bash build.sh -e ascend
```

Of which,

- In the `build.sh` script, the default number of compilation threads is 8. If the compiler performance is poor, compilation errors may occur. You can add -j{Number of threads} in to script to reduce the number of threads. For example, `bash build.sh -e ascend -j4`.

## Installing MindSpore

```bash
chmod +x build/package/mindspore_ascend-{version}-cp37-cp37m-linux_{arch}.whl
pip install build/package/mindspore_ascend-{version}-cp37-cp37m-linux_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.0/requirements.txt)). In other cases, you need to manually install dependency items.
- `{version}` denotes the version of MindSpore. For example, when you are downloading MindSpore 1.0.1, `{version}` should be 1.0.1.
- `{arch}` denotes the system architecture. For example, the Linux system you are using is x86 architecture 64-bit, `{arch}` should be `x86_64`. If the system is ARM architecture 64-bit, then it should be `aarch64`.

## Configuring Environment Variables

- **If Ascend 910 AI processor software is installed in a non-default path**, after MindSpore is installed, export runtime-related environment variables. `/usr/local/Ascend` in the following command `LOCAL_ASCEND=/usr/local/Ascend` denotes the installation path of the software package, please replace it as your actual installation path.

    ```bash
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
    export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}
    # Python library that TBE implementation depends on

    ```

## Installation Verification

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

Using the following command if you need update MindSpore version.

- Update Online

    ```bash
    pip install --upgrade mindspore-ascend
    ```

- Update after source code compilation

    After successfully executing the compile script `build.sh` in the root path of the source code, find the `whl` package in path `build/package`, use the following command to update your version.

    ```bash
    pip install --upgrade mindspore_ascend-{version}-cp37-cp37m-linux_{arch}.whl
    ```

## Installing MindInsight

If you need to analyze information such as model scalars, graphs, computation graphs and model traceback, you can install MindInsight.

For more details, please refer to [MindInsight](https://gitee.com/mindspore/mindinsight/blob/master/README.md).

## Installing MindArmour

If you need to conduct AI model security research or enhance the security of the model in you applications, you can install MindArmour.

For more details, please refer to [MindArmour](https://gitee.com/mindspore/mindarmour/blob/master/README.md).

## Installing MindSpore Hub

If you need to access and experience MindSpore pre-trained models quickly, you can install MindSpore Hub.

For more details, please refer to [MindSpore Hub](https://gitee.com/mindspore/hub/blob/master/README.md).
