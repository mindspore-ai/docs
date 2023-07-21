# Installing MindSpore in Ascend by pip

<!-- TOC -->

- [Installing MindSpore in Ascend by pip](#install-mindspore-in-ascend-by-pip)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Installing MindSpore](#installing-mindspore)
    - [Configuring Environment Variables](#configuring-environment-variables)
    - [Installation Verification](#insallation-verification)
    - [Version Update](#version-update)
    - [Installing MindInsight](#installing-mindinsight)
    - [Installing MindArmour](#installing-mindarmour)
    - [Installing MindSpore Hub](#installing-mindspore-hub)

<!-- /TOC -->

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/install/mindspore_ascend_install_pip_en.md)

This document describes how to quickly install MindSpore in a Linux system with an Ascend 910 environment by pip.

## System Environment Information Confirmation

- Confirm that Ubuntu 18.04/CentOS 7.6/EulerOS 2.8 is installed with 64-bit operating system.
- Confirm that [GCC 7.3.0](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz) is installed.
- Confirm that [gmp 6.1.2](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz) is installed.
- Confirm that Python 3.7.5 is installed.
    - If you didn't install Python or you have installed other versions, please download the Python 3.7.5 64-bit from [Python](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) or [Huaweicloud](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz) to install.
- Confirm that the Ascend 910 AI processor software package (Atlas Data Center Solution V100R020C10：[A800-9000 1.0.8 (aarch64)](https://support.huawei.com/enterprise/zh/ascend-computing/a800-9000-pid-250702818/software/252069004?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C250702818), [A800-9010 1.0.8 (x86_64)](https://support.huawei.com/enterprise/zh/ascend-computing/a800-9010-pid-250702809/software/252062130?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C250702809), [CANN V100R020C10](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/251174283?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373)) are installed.
- Confirm that the current user has the right to access the installation path `/usr/local/Ascend`of Ascend 910 AI processor software package, If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located. For the specific configuration, please refer to the software package instruction document.
    - Install the .whl package provided in Ascend 910 AI processor software package. The .whl package is released with the software package. After software package is upgraded, reinstall the .whl package.

        ```bash
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-{version}-py3-none-any.whl
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-{version}-py3-none-any.whl
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-{version}-py3-none-any.whl
        ```

## Installing MindSpore

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/ascend/{system}/mindspore_ascend-{version}-cp37-cp37m-linux_{arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.0/requirements.txt)). In other cases, you need to manually install dependency items.  
- `{version}` denotes the version of MindSpore. For example, when you are downloading MindSpore 1.0.1, `{version}` should be 1.0.1.  
- `{arch}` denotes the system architecture. For example, the Linux system you are using is x86 architecture 64-bit, `{arch}` should be `x86_64`. If the system is ARM architecture 64-bit, then it should be `aarch64`.  
- `{system}` denotes the system version. For example, if you are using EulerOS ARM architecture, `{system}` should be `euleros_aarch64`. Currently, the following systems are supported by Ascend: `euleros_aarch64`/`centos_aarch64`/`centos_x86`/`ubuntu_aarch64`/`ubuntu_x86`.

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

- After configuring the environment variables, execute the following Python script：

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

Using the following command if you need update MindSpore version:

```bash
pip install --upgrade mindspore-ascend
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
