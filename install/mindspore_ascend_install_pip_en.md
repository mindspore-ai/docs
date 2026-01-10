# Installing MindSpore in Ascend by pip

<!-- TOC -->

- [Installing MindSpore in Ascend by pip](#installing-mindspore-in-ascend-by-pip)
    - [Installing MindSpore and dependencies](#installing-mindspore-and-dependencies)
        - [Installing Python](#installing-python)
        - [Installing Ascend AI processor software package](#installing-ascend-ai-processor-software-package)
        - [Installing GCC](#installing-gcc)
        - [Installing MindSpore](#installing-mindspore)
    - [Configuring Environment Variables](#configuring-environment-variables)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.8.0/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/r2.8.0/install/mindspore_ascend_install_pip_en.md)

This document describes how to install MindSpore by pip on Linux in an Ascend environment.

## Installing MindSpore and dependencies

The following table lists the system environment and third-party dependencies required for installing MindSpore.

|Software|Version|Description|
|-|-|-|
|Debian series OS / openEuler series OS|Debianseries: Debian, Ubuntu, veLinux / openEuler serires: openEuler, CentOS, Kylin, BCLinux, UOS V20, AntOS, CTyunOS, CULinux, Tlinux, MTOS|Operating Systems compatible to MindSpore|
|[Python](#installing-python)|3.9-3.12|Python environment that MindSpore depends on|
|[Ascend AI processor software package](#installing-ascend-ai-processor-software-package)|CANN 8.5.0, CANN 8.3.RC1, CANN 8.2.RC1|Ascend platform AI computing library used by MindSpore|
|[GCC](#installing-gcc)|7.3.0|C++ compiler for compiling MindSpore|

The following describes how to install the third-party dependencies.

### Installing Python

[Python](https://www.python.org/) can be installed by Conda.

Install Miniconda:

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_25.7.0-2-Linux-$(arch).sh
bash Miniconda3-py39_25.7.0-2-Linux-$(arch).sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

After the installation is complete, you can set up Tsinghua source acceleration download for Conda, and see [here](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/).

Create a virtual environment, taking Python 3.9.11 as an example:

```bash
conda create -n mindspore_py39 python=3.9.11 -y
conda activate mindspore_py39
```

Run the following command to check the Python version.

```bash
python --version
```

If you are using ARM architecture, ensure that pip installed for current Python is 19.3 or later. If not, upgrade pip using the following command.

```bash
python -m pip install -U pip
```

### Installing Ascend AI processor software package

To install Ascend software package community edition, the recommended version is `8.5.0` in [CANN community edition](https://www.hiascend.com/developer/download/community/result?module=cann), then choose relevant driver and firmware packages in [firmware and driver](https://www.hiascend.com/hardware/firmware-drivers/community). For installation guide, please refer to [Installation guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/quickstart/instg_quick.html).

The default installation path of the installation package is `/usr/local/Ascend`. Ensure that the current user has the right to access the installation path of Ascend AI processor software package. If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located.

Install python packages required by Ascend AI processor software package. If packages `te topi hccl` have been installed before, you should uninstall these packages.

```bash
pip uninstall te topi hccl -y
pip install sympy protobuf attrs cloudpickle decorator ml-dtypes psutil scipy tornado jinja2
```

### Installing GCC

- On Ubuntu 18.04, run the following commands to install.

    ```bash
    sudo apt-get install gcc-7 -y
    ```

- On CentOS 7, run the following commands to install.

    ```bash
    sudo yum install centos-release-scl
    sudo yum install devtoolset-7
    ```

    After installation, run the following commands to switch to GCC 7.

    ```bash
    scl enable devtoolset-7 bash
    ```

- On EulerOS and openEuler, run the following commands to install.

    ```bash
    sudo yum install gcc -y
    ```

### Installing MindSpore

First, refer to [Version List](https://www.mindspore.cn/versions) to select the version of MindSpore you want to install, and perform SHA-256 integrity check. Taking version 2.7.1 as an example, execute the following commands.

```bash
export MS_VERSION=2.7.1
```

Then run the following command to install MindSpore.

```bash
pip install mindspore==${MS_VERSION} -i https://repo.mindspore.cn/pypi/simple --trusted-host repo.mindspore.cn --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple/
```

When the network is connected, dependencies of MindSpore are automatically downloaded during the .whl package installation. (For details about the dependency, see required_package in [setup.py](https://atomgit.com/mindspore/mindspore/blob/v2.8.0/setup.py)). In other cases, you need to install dependencies by yourself.

## Configuring Environment Variables

**If Ascend AI processor software is installed in a non-default path**, after MindSpore is installed, export runtime-related environment variables. `/usr/local/Ascend` in the following command `LOCAL_ASCEND=/usr/local/Ascend` denotes the installation path of the software package, and you need to replace it as the actual installation path of the software package.

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# environment variables
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# set environmet variables using script provided by CANN, swap "ascend-toolkit" with "nnae" if you are using CANN-nnae package instead
source ${LOCAL_ASCEND}/cann/set_env.sh
```

## Installation Verification

**Method 1:**

Execute the following command:

```bash
python -c "import mindspore;mindspore.set_device('Ascend');mindspore.run_check()"
```

The outputs should be the same as:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!
```

It means MindSpore has been installed successfully.

**Method 2:**

Execute the following command:

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_device("Ascend")
x = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))
```

The outputs should be the same as:

```text
[[[[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]]]
```

It means MindSpore has been installed successfully.

## Version Update

When upgrading from MindSpore 1.x to MindSpore 2.x, you need to manually uninstall the old version first:

```bash
pip uninstall mindspore-ascend
```

Then install MindSpore 2.x:

```bash
pip install mindspore=={version}
```

When upgrading from MindSpore 2.x:

```bash
pip install --upgrade mindspore=={version}
```

Of which,

- When updating to a release candidate (RC) version, set `{version}` to the RC version number, for example, 1.6.0rc1. When updating to a stable release, you can remove `=={version}`.
