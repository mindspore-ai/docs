# Installing MindSpore Ascend by Conda

<!-- TOC -->

- [Installing MindSpore Ascend by Conda](#installing-mindspore-ascend-by-conda)
    - [Installing MindSpore and dependencies](#installing-mindspore-and-dependencies)
        - [Installing Ascend AI processor software package](#installing-ascend-ai-processor-software-package)
        - [Installing Conda](#installing-conda)
        - [Installing GCC](#installing-gcc)
        - [Creating and Accessing the Conda Virtual Environment](#creating-and-accessing-the-conda-virtual-environment)
        - [Installing MindSpore](#installing-mindspore)
    - [Configuring Environment Variables](#configuring-environment-variables)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend_install_conda_en.md)

[Conda](https://docs.conda.io/en/latest/) is an open-source, cross-platform, language-agnostic package manager and environment management system. It allows users to easily install different versions of binary software packages and any required libraries appropriate for their computing platform.

This document describes how to install MindSpore by Conda on Linux in an Ascend environment.

## Installing MindSpore and dependencies

The following table lists the system environment and third-party dependencies required for installing MindSpore.

|Software|Version|Description|
|-|-|-|
|Ubuntu 18.04 / CentOS 7.6 / EulerOS 2.8 / openEuler 20.03 / KylinV10 SP1|-|OS for running MindSpore|
|[Ascend AI processor software package](#installing-ascend-ai-processor-software-package)|-|Ascend platform AI computing library used by MindSpore|
|[Conda](#installing-conda)|Anaconda3 or Miniconda3|Python environment management tool|
|[GCC](#installing-gcc)|7.3.0|C++ compiler for compiling MindSpore|

The following describes how to install the third-party dependencies.

### Installing Ascend AI processor software package

Ascend software package provides two distributions, commercial edition and community edition:

- Commercial edition needs approval from Ascend to download, for detailed installation guide of the recommended version, please refer to [Ascend Training Solution 25.0.RC1](https://support.huawei.com/enterprise/zh/doc/EDOC1100472026)

- Community edition has no restrictions, the recommended version is `8.1.RC1.beta1` in [CANN community edition](https://www.hiascend.com/developer/download/community/result?module=cann), then choose relevant driver and firmware packages in [firmware and driver](https://www.hiascend.com/hardware/firmware-drivers/community). Please refer to the abovementioned commercial edition installation guide to choose which packages are to be installed and how to install them.

The default installation path of the installation package is `/usr/local/Ascend`. Ensure that the current user has the right to access the installation path `/usr/local/Ascend` of Ascend AI processor software package. If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located.

### Installing Conda

Run the following command to install Miniconda.

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-$(arch).sh
bash Miniconda3-py37_4.10.3-Linux-$(arch).sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

After the installation is complete, you can set up Tsinghua Source to accelerate the download for Conda, and refer to [Here](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/).

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

### Creating and Accessing the Conda Virtual Environment

Create a Conda virtual environment based on the Python version you want to use and activate the virtual environment.

If you want to use Python 3.9.11:

```bash
conda create -c conda-forge -n mindspore_py39 python=3.9.11 -y
conda activate mindspore_py39
```

If you wish to use another version of Python, just change the Python version in the above command. Python 3.9, Python 3.10 and Python 3.11 are currently supported.

Install the .whl package provided with the Ascend AI Processor software package in the virtual environment. The .whl package is released with the software package. After the software package is upgraded, you need to reinstall the .whl package.

```bash
pip install sympy
pip install "numpy>=1.20.0,<2.0.0"
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-{version}-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-{version}-py3-none-any.whl
```

If the software package of the Ascend AI processor is upgraded, the matching .whl package also needs to be reinstalled. Uninstall the original installation package and reinstall the .whl package by referring to the preceding commands.

```bash
pip uninstall te topi hccl -y
```

### Installing MindSpore

Ensure that you are in the Conda virtual environment and run the following command to install the latest MindSpore. To install other versions, please refer to the specified the version of [Version List](https://www.mindspore.cn/versions) after `conda install mindspore=`.

```bash
conda install mindspore -c mindspore -c conda-forge
```

When the network is connected, dependencies of MindSpore are automatically downloaded during the .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py)). In other cases, you need to install dependencies by yourself.

## Configuring Environment Variables

**If Ascend AI processor software is installed in a non-default path**, after MindSpore is installed, export Runtime-related environment variables. `/usr/local/Ascend` in the following command `LOCAL_ASCEND=/usr/local/Ascend` denotes the installation path of the software package, and you need to replace it as the actual installation path of the software package.

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# environment variables
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# set environmet variables using script provided by CANN, swap "ascend-toolkit" with "nnae" if you are using CANN-nnae package instead
source ${LOCAL_ASCEND}/ascend-toolkit/set_env.sh
```

## Installation Verification

**Method 1:**

Execute the following command:

```bash
python -c "import mindspore;mindspore.set_device('Ascend');mindspore.run_check()"
```

The output should be like:

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
conda remove mindspore-ascend
```

Then install MindSpore 2.x:

```bash
conda install mindspore -c mindspore -c conda-forge
```

When upgrading from MindSpore 2.x:

```bash
conda update mindspore -c mindspore -c conda-forge
```

Please reinstall .whl packages provided with the Ascend AI Processor software package.

First, uninstall old versions:

```bash
pip uninstall te topi hccl -y
```

Then, install again by running:

```bash
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-*-py3-none-any.whl
```
