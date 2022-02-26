# Installing MindSpore CPU by Conda

<!-- TOC -->

- [Installing MindSpore CPU by Conda](#installing-mindspore-cpu-by-conda)
    - [Installing Environment Dependencies](#installing-environment-dependencies)
    - [Creating and Accessing the Conda Virtual Environment](#creating-and-accessing-the-conda-virtual-environment)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_install_conda_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

[Conda](https://docs.conda.io/en/latest/) is an open-source, cross-platform, language-agnostic package manager and environment management system. It allows users to easily install different versions of binary software packages and any required libraries appropriate for their computing platform.

This document describes how to quickly install MindSpore by Conda in a Linux system with a CPU environment. The following takes Ubuntu 18.04 as an example to illustrate the steps to install MindSpore.

- If you want to install MindSpore by Conda on a fresh Ubuntu 18.04, you may use [automatic installation script](https://gitee.com/mindspore/mindspore/raw/master/scripts/install/ubuntu-cpu-conda.sh) for one-click installation. The automatic installation script will install MindSpore and its dependencies.

    The automatic installation script needs to replace the source list and install dependencies via APT, it will apply for root privileges during execution. Use the following commands to get the automatic installation script and execute.

    ```bash
    wget https://gitee.com/mindspore/mindspore/raw/master/scripts/install/ubuntu-cpu-conda.sh
    # install Python 3.7 and the latest MindSpore by default
    bash ./ubuntu-cpu-conda.sh
    # to specify Python and MindSpore version, e.g. Python 3.9 and MindSpore 1.5.0
    # PYTHON_VERSION=3.9 MINDSPORE_VERSION=1.5.0 bash ./ubuntu-cpu-conda.sh
    ```

    The script will:

    - Set the source list to huaweicloud source.
    - Install the dependencies required by MindSpore, such as GCC, gmp.
    - Install Conda and create a virtual environment for MindSpore.
    - Install MindSpore CPU by Conda.

    For more usage, please refer to the description at the head of the script.

- If your system has already installed some dependencies, such as Conda, GCC, etc., it is recommended to install manually by referring to the following installation steps.

## Installing Environment Dependencies

The following table lists the system environment and third-party dependencies required to install MindSpore.

|software|version|description|
|-|-|-|
|Ubuntu|18.04|operating system for compiling and running MindSpore|
|[Conda](#install-conda)|Anaconda3 or Miniconda3|Python environment management tool|
|[GCC](#install-gccgmp)|7.3.0|C++ compiler for compiling MindSpore|
|[gmp](#install-gccgmp)|6.1.2|multiple precision arithmetic library used by MindSpore|

The following are installation methods of third-party dependencies.

### Install Conda

Execute the following command to install Miniconda.

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

### Install GCC/gmp

You may install GCC and gmp by the following commands.

```bash
sudo apt-get install gcc-7 libgmp-dev -y
```

## Creating and Accessing the Conda Virtual Environment

Create a Conda virtual environment based on the Python version you want and activate the virtual environment.

If you want to use Python 3.7.5:

```bash
conda create -n mindspore_py37 python=3.7.5 -y
conda activate mindspore_py37
```

If you want to use Python 3.9.0:

```bash
conda create -n mindspore_py39 python=3.9.0 -y
conda activate mindspore_py39
```

## Installing MindSpore

Ensure that you are in the Conda virtual environment and run the following command to install the latest MindSpore. To install other versions, please refer to [Version List](https://www.mindspore.cn/versions) and specify the version after `mindspore-cpu=`.

```bash
conda install mindspore-cpu -c mindspore -c conda-forge
```

When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/master/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt).

## Installation Verification

```bash
python -c "import mindspore;mindspore.run_check()"
```

The outputs should be the same as:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

It means MindSpore has been installed successfully.

## Version Update

Using the following command if you need to update the MindSpore version:

```bash
conda update mindspore-cpu -c mindspore -c conda-forge
```
