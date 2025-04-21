# Installing MindSpore CPU by Conda

<!-- TOC -->

- [Installing MindSpore CPU by Conda](#installing-mindspore-cpu-by-conda)
    - [Installing MindSpore and dependencies](#installing-mindspore-and-dependencies)
        - [Installing Conda](#installing-conda)
        - [Installing GCC](#installing-gcc)
        - [Creating and Accessing the Conda Virtual Environment](#creating-and-accessing-the-conda-virtual-environment)
        - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/install/mindspore_cpu_install_conda_en.md)

[Conda](https://docs.conda.io/en/latest/) is an open-source, cross-platform, language-agnostic package manager and environment management system. It allows users to easily install different versions of binary software packages and any required libraries appropriate for their computing platform.

This document describes how to install MindSpore by Conda on Linux in a CPU environment. The following takes Ubuntu 18.04 as an example to describe how to install MindSpore.

## Installing MindSpore and dependencies

The following table lists the system environment and third-party dependencies required to install MindSpore.

|Software|Version|Description|
|-|-|-|
|Ubuntu|18.04|OS for running MindSpore|
|[Conda](#installing-conda)|Anaconda3 or Miniconda3|Python environment management tool|
|[GCC](#installing-gcc)|7.3.0-9.4.0|C++ compiler for compiling MindSpore|

The following describes how to install the third-party dependencies.

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

After the installation is complete, you can set up Tsinghua source acceleration download for Conda, and see [here](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/).

### Installing GCC

Run the following commands to install GCC.

```bash
sudo apt-get install gcc-7 -y
```

To install a later version of GCC, run the following command to install GCC 8.

```bash
sudo apt-get install gcc-8 -y
```

Or install GCC 9.

```bash
sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-9 -y
```

### Creating and Accessing the Conda Virtual Environment

Create a Conda virtual environment based on the Python version you want and activate the virtual environment.

If you want to use Python 3.9.11, execute the following command:

```bash
conda create -c conda-forge -n mindspore_py39 python=3.9.11 -y
conda activate mindspore_py39
```

If you wish to use another version of Python, just change the Python version in the above command. Python 3.9, Python 3.10 and Python 3.11 are currently supported.

### Installing MindSpore

Ensure that you are in the Conda virtual environment and run the following command to install the latest MindSpore. To install other versions, please refer to the specified version number of [Version List](https://www.mindspore.cn/versions) after `conda install mindspore=`.

```bash
conda install mindspore -c mindspore -c conda-forge
```

When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/setup.py)). In other cases, you need to install dependency by yourself.

## Installation Verification

Execute the following command:

```bash
python -c "import mindspore;mindspore.set_device(device_target='CPU');mindspore.run_check()"
```

The outputs should be the same as:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed on platform [CPU] successfully!
```

It means MindSpore has been installed successfully.

## Version Update

When upgrading from MindSpore 1.x to MindSpore 2.x, you need to manually uninstall the old version first:

```bash
conda remove mindspore-cpu
```

Then install MindSpore 2.x:

```bash
conda install mindspore -c mindspore -c conda-forge
```

When upgrading from MindSpore 2.x:

```bash
conda update mindspore -c mindspore -c conda-forge
```
