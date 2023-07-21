# Installing MindSpore CPU by Conda

<!-- TOC -->

- [Installing MindSpore CPU by Conda](#installing-mindspore-cpu-by-conda)
    - [Automatic Installation](#automatic-installation)
    - [Manual Installation](#manual-installation)
        - [Installing Conda](#installing-conda)
        - [Installing GCC](#installing-gcc)
        - [Creating and Accessing the Conda Virtual Environment](#creating-and-accessing-the-conda-virtual-environment)
        - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.0/install/mindspore_cpu_install_conda_en.md)

[Conda](https://docs.conda.io/en/latest/) is an open-source, cross-platform, language-agnostic package manager and environment management system. It allows users to easily install different versions of binary software packages and any required libraries appropriate for their computing platform.

This document describes how to install MindSpore by Conda on Linux in a CPU environment. The following takes Ubuntu 18.04 as an example to describe how to install MindSpore.

- If you want to install MindSpore by Conda on a fresh Ubuntu 18.04, you may use an [automatic installation script](https://gitee.com/mindspore/mindspore/raw/r2.0/scripts/install/ubuntu-cpu-conda.sh) for one-click installation. For details, see [Automatic Installation](#automatic-installation). The script will automatically install MindSpore and its dependencies.

- If some dependencies, such as Conda and GCC, have been installed in your system, it is recommended to install manually by referring to the installation steps in the [Manual Installation](#manual-installation).

## Automatic Installation

The automatic installation script needs to replace the source list and install dependencies via APT, it will apply for root privileges during execution. Run the following command to obtain and run the automatic installation script. The automatic installation script supports only MindSpore>=1.6.0 or later.

```bash
wget https://gitee.com/mindspore/mindspore/raw/r2.0/scripts/install/ubuntu-cpu-conda.sh
# install Python 3.7 and the latest MindSpore by default
bash ./ubuntu-cpu-conda.sh
# to specify Python and MindSpore version, taking Python 3.9 and MindSpore 1.6.0 as examples, use the following manners
# PYTHON_VERSION=3.9 MINDSPORE_VERSION=1.6.0 bash ./ubuntu-cpu-conda.sh
```

This script performs the following operations:

- Change the software source configuration to a HUAWEI CLOUD source.
- Install the dependencies required by MindSpore, such as GCC.
- Install Conda and create a virtual environment for MindSpore.
- Install MindSpore CPU by Conda.

After the automatic installation script is executed, you need to reopen the terminal window to make the environment variables take effect. The automatic installation script creates a virtual environment named `mindspore_pyXX` for MindSpore. Where `XX` is the Python version, such as Python 3.7, the virtual environment name is `mindspore_py37`. Run the following command to show all virtual environments.

```bash
conda env list
```

To activate the virtual environment, take Python 3.7 as an example, execute the following command.

```bash
conda activate mindspore_py37
```

For more usage, see the script header description.

## Manual Installation

The following table lists the system environment and third-party dependencies required to install MindSpore.

|Software|Version|Description|
|-|-|-|
|Ubuntu|18.04|OS for running MindSpore|
|[Conda](#installing-conda)|Anaconda3 or Miniconda3|Python environment management tool|
|[GCC](#installing-gcc-and-gmp)|7.3.0~9.4.0|C++ compiler for compiling MindSpore|

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

If you want to use Python 3.7.5:

```bash
conda create -c conda-forge -n mindspore_py37 python=3.7.5 -y
conda activate mindspore_py37
```

If you wish to use another version of Python, just change the Python version in the above command. Python 3.7, Python 3.8 and Python 3.9 are currently supported.

### Installing MindSpore

Ensure that you are in the Conda virtual environment and run the following command to install the latest MindSpore. To install other versions, please refer to the specified version number of [Version List](https://www.mindspore.cn/versions) after `mindspore=`.

```bash
conda install mindspore -c mindspore -c conda-forge
```

When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/r2.0/setup.py).) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/r2.0/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r2.0/requirements.txt).

## Installation Verification

```bash
python -c "import mindspore;mindspore.run_check()"
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
