# Installing MindSpore in Ascend by pip

<!-- TOC -->

- [Installing MindSpore in Ascend by pip](#installing-mindspore-in-ascend-by-pip)
    - [Automatic Installation](#automatic-installation)
    - [Manual Installation](#manual-installation)
        - [Installing Python](#installing-python)
        - [Installing Ascend AI processor software package](#installing-ascend-ai-processor-software-package)
        - [Installing GCC](#installing-gcc)
        - [Installing MindSpore](#installing-mindspore)
    - [Configuring Environment Variables](#configuring-environment-variables)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.4.0/install/mindspore_ascend_install_pip_en.md)

This document describes how to install MindSpore by pip on Linux in an Ascend environment.

- If you want to install MindSpore by pip on EulerOS 2.8 with the configured Ascend AI processor software package. you may use an [automatic installation script](https://gitee.com/mindspore/mindspore/raw/v2.4.0/scripts/install/euleros-ascend-pip.sh) for one-click installation. For details, see [Automatic Installation](#automatic-installation). The script will automatically install MindSpore and its required dependencies.

- If your system is Ubuntu 18.04, CentOS 7.6, openEuler 20.03, or KylinV10 SP1, or some dependencies, such as Python and GCC, have been installed, it is advised to manually install MindSpore by referring to [Manual Installation](#manual-installation).

## Automatic Installation

Before running the automatic installation script, you need to make sure that the Ascend AI processor software package is correctly installed on your system. If it is not installed, please refer to the section [Installing Ascend AI processor software package](#installing-ascend-ai-processor-software-package) to install it.

Run the following command to obtain and run the automatic installation script. The automatic installation script supports only MindSpore>=1.6.0 or later.

```bash
wget https://gitee.com/mindspore/mindspore/raw/v2.4.0/scripts/install/euleros-ascend-pip.sh
# install latest MindSpore and Python 3.9
# the default value of LOCAL_ASCEND is /usr/local/Ascend
PYTHON_VERSION=3.9 bash -i ./euleros-ascend-pip.sh
# to specify Python and MindSpore version, taking Python 3.9 and MindSpore 1.6.0 as examples
# and set LOCAL_ASCEND to /home/xxx/Ascend, use the following manners
# LOCAL_ASCEND=/home/xxx/Ascend PYTHON_VERSION=3.9 MINDSPORE_VERSION=1.6.0 bash -i ./euleros-ascend-pip.sh
```

This script performs the following operations:

- Install the dependencies required by MindSpore, such as GCC.
- Install Python3 and pip3 and set them as default.
- Install MindSpore Ascend by pip.
- Install Open MPI if OPENMPI is set to `on`.

After the script is executed, you need to reopen the terminal window to make the environment variables take effect.

The automatic installation script creates a virtual environment named `mindspore_pyXX` for MindSpore. Where `XX` is the Python version, such as Python 3.9, the virtual environment name is `mindspore_py39`. Run the following command to show all virtual environments.

```bash
conda env list
```

To activate the virtual environment, take Python 3.9 as an example, execute the following command.

```bash
conda activate mindspore_py39
```

Now you can jump to the [Configuring Environment Variables](#configuring-environment-variables) section to set the relevant environment variables.

For more usage, see the script header description.

## Manual Installation

The following table lists the system environment and third-party dependencies required for installing MindSpore.

|Software|Version|Description|
|-|-|-|
|Ubuntu 18.04/CentOS 7.6/EulerOS 2.8/openEuler 20.03/KylinV10 SP1|-|OS for running MindSpore|
|[Python](#installing-python)|3.9-3.11|Python environment that MindSpore depends on|
|[Ascend AI processor software package](#installing-ascend-ai-processor-software-package)|-|Ascend platform AI computing library used by MindSpore|
|[GCC](#installing-gcc)|7.3.0|C++ compiler for compiling MindSpore|

The following describes how to install the third-party dependencies.

### Installing Python

[Python](https://www.python.org/) can be installed by Conda.

Install Miniconda:

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-$(arch).sh
bash Miniconda3-py37_4.10.3-Linux-$(arch).sh -b
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

Ascend software package provides two distributions, commercial edition and community edition:

- Commercial edition needs approval from Ascend to download, release date is TBD.

- Community edition has no restrictions, release date is TBD.

The default installation path of the installation package is `/usr/local/Ascend`. Ensure that the current user has the right to access the installation path of Ascend AI processor software package. If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located.

Install the .whl packages provided in Ascend AI processor software package. If the .whl packages have been installed before, you should uninstall the .whl packages by running the following command.

```bash
pip uninstall te topi hccl -y
```

Run the following command to install the .whl packages if the Ascend AI package has been installed in default path. If the installation path is not the default path, you need to replace the path in the command with the installation path.

```bash
pip install sympy
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-*-py3-none-any.whl
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

First, refer to [Version List](https://www.mindspore.cn/versions) to select the version of MindSpore you want to install, and perform SHA-256 integrity check. Taking version 2.4.0 as an example, execute the following commands.

```bash
export MS_VERSION=2.4.0
```

Then run the following commands to install MindSpore according to the system architecture and Python version.

```bash
# x86_64 + Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/unified/x86_64/mindspore-${MS_VERSION/-/}-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# x86_64 + Python3.10
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/unified/x86_64/mindspore-${MS_VERSION/-/}-cp310-cp310-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# x86_64 + Python3.11
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/unified/x86_64/mindspore-${MS_VERSION/-/}-cp311-cp311-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64 + Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/unified/aarch64/mindspore-${MS_VERSION/-/}-cp39-cp39-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64 + Python3.10
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/unified/aarch64/mindspore-${MS_VERSION/-/}-cp310-cp310-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64 + Python3.11
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/unified/aarch64/mindspore-${MS_VERSION/-/}-cp311-cp311-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

When the network is connected, dependencies of MindSpore are automatically downloaded during the .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/v2.4.0/setup.py).) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/master/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/v2.4.0/requirements.txt).

## Configuring Environment Variables

**If Ascend AI processor software is installed in a non-default path**, after MindSpore is installed, export runtime-related environment variables. `/usr/local/Ascend` in the following command `LOCAL_ASCEND=/usr/local/Ascend` denotes the installation path of the software package, and you need to replace it as the actual installation path of the software package.

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# environment variables
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# set environmet variables using script provided by CANN, swap "ascend-toolkit" with "nnae" if you are using CANN-nnae package instead
source ${LOCAL_ASCEND}/ascend-toolkit/set_env.sh
```

## Installation Verification

i:

```bash
python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"
```

The outputs should be the same as:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!
```

It means MindSpore has been installed successfully.

ii:

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_context(device_target="Ascend")
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
