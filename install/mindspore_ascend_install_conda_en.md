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

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/install/mindspore_ascend_install_conda_en.md)

[Conda](https://docs.conda.io/en/latest/) is an open-source, cross-platform, language-agnostic package manager and environment management system. It allows users to easily install different versions of binary software packages and any required libraries appropriate for their computing platform.

This document describes how to install MindSpore by Conda on Linux in an Ascend environment.

## Installing MindSpore and dependencies

The following table lists the system environment and third-party dependencies required for installing MindSpore.

|Software|Version|Description|
|-|-|-|
|Debian series OS / openEuler series OS|Debianseries: Debian, Ubuntu, veLinux / openEuler serires: openEuler, CentOS, Kylin, BCLinux, UOS V20, AntOS, CTyunOS, CULinux, Tlinux, MTOS|Operating Systems compatible to MindSpore|
|[Ascend AI processor software package](#installing-ascend-ai-processor-software-package)|CANN 9.0.0, CANN 8.5.0, CANN 8.3.RC1|Ascend platform AI computing library used by MindSpore|
|[Conda](#installing-conda)|Anaconda3 or Miniconda3|Python environment management tool|
|[GCC](#installing-gcc)|7.3.0|C++ compiler for compiling MindSpore|

The following describes how to install the third-party dependencies.

### Installing Ascend AI processor software package

Ascend CANN `9.0.0` will be released soon.

The default installation path of the installation package is `/usr/local/Ascend`. Ensure that the current user has the right to access the installation path `/usr/local/Ascend` of Ascend AI processor software package. If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located.

### Installing Conda

Run the following command to install Miniconda.

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_26.1.1-1-Linux-$(arch).sh
bash Miniconda3-py310_26.1.1-1-Linux-$(arch).sh
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

After the installation is complete, you can set up Tsinghua Source to accelerate the download for Conda, and refer to [Here](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/).

### Installing GCC

The following takes GCC 7 as example for how to install GCC on common operating systems:

- On Ubuntu, run the following commands to install.

    ```bash
    sudo apt-get install gcc-7 g++-7 -y
    ```

- On CentOS, run the following commands to install.

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
    sudo yum install gcc g++ -y
    ```

### Creating and Accessing the Conda Virtual Environment

Create a Conda virtual environment based on the Python version you want to use and activate the virtual environment.

If you want to use Python 3.10.20:

```bash
conda create -c conda-forge -n mindspore_py310 python=3.10.20 -y
conda activate mindspore_py310
```

If you wish to use another version of Python, just change the Python version in the above command. Python 3.9, Python 3.10, Python 3.11 and Python 3.12 are currently supported.

Install python packages required by Ascend AI processor software package. If packages `te topi hccl` have been installed before, you should uninstall these packages.

```bash
pip uninstall te topi hccl -y
pip install sympy protobuf attrs cloudpickle decorator ml-dtypes psutil scipy tornado jinja2
```

### Installing MindSpore

Ensure that you are in the Conda virtual environment and run the following command to install the latest MindSpore. To install other versions, please refer to the specified the version of [Version List](https://www.mindspore.cn/versions) after `conda install mindspore=`.

```bash
conda install mindspore -c mindspore -c conda-forge
```

When the network is connected, dependencies of MindSpore are automatically downloaded during the .whl package installation. (For details about the dependency, see required_package in [setup.py](https://atomgit.com/mindspore/mindspore/blob/master/setup.py)). In other cases, you need to install dependencies by yourself.

## Configuring Environment Variables

**If Ascend AI processor software is installed in a non-default path**, after MindSpore is installed, export Runtime-related environment variables. `/usr/local/Ascend` in the following command `LOCAL_ASCEND=/usr/local/Ascend` denotes the installation path of the software package, and you need to replace it as the actual installation path of the software package.

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

If packages `te topi hccl` have been installed in your conda environment before, you should uninstall these packages:

```bash
pip uninstall te topi hccl -y
```
