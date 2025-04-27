# Installing MindSpore in CPU by Source Code

<!-- TOC -->

- [Installing MindSpore in CPU by Source Code](#installing-mindspore-in-cpu-by-source-code)
    - [Installing dependencies](#installing-dependencies)
        - [Installing Python](#installing-python)
        - [Installing wheel setuptools PyYAML and Numpy](#installing-wheel-setuptools-pyyaml-and-numpy)
        - [Installing GCC git tclsh patch and NUMA](#installing-gcc-git-tclsh-patch-and-numa)
        - [Installing CMake](#installing-cmake)
        - [Installing LLVM-optional](#installing-llvm-optional)
    - [Downloading the Source Code from the Code Repository](#downloading-the-source-code-from-the-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/install/mindspore_cpu_install_source_en.md)

This document describes how to install MindSpore by compiling source code on Linux system in the CPU environment. The following takes Ubuntu 18.04 as an example to describe how to compile and install MindSpore.

## Installing dependencies

|Software|Version|Description|
|-|-|-|
|Ubuntu|18.04|OS for compiling and running MindSpore|
|[Python](#installing-python)|3.9-3.11|Python environment that MindSpore depends|
|[wheel](#installing-wheel-setuptools-pyyaml-and-numpy)|0.32.0 or later|Python packaging tool used by MindSpore|
|[setuptools](#installing-wheel-setuptools-pyyaml-and-numpy)|44.0 or later|Python package management tool used by MindSpore|
|[PyYAML](#installing-wheel-setuptools-pyyaml-and-numpy)|6.0-6.0.2|PyYAML module that operator compliation in MindSpore depends on|
|[Numpy](#installing-wheel-setuptools-pyyaml-and-numpy)|1.19.3-1.26.4|Numpy module that Numpy-related functions in MindSpore depends on|
|[GCC](#installing-gcc-git-tclsh-patch-and-numa)|7.3.0-9.4.0|C++ compiler for compiling MindSpore|
|[git](#installing-gcc-git-tclsh-patch-and-numa)|-|Source code management tools used by MindSpore|
|[CMake](#installing-cmake)|3.22.2 or later|Compilation tool that builds MindSpore|
|[tclsh](#installing-gcc-git-tclsh-patch-and-numa)|-|MindSpore SQLite compilation dependency|
|[patch](#installing-gcc-git-tclsh-patch-and-numa)|2.5 or later|Source code patching tool used by MindSpore|
|[NUMA](#installing-gcc-git-tclsh-patch-and-numa)|2.0.11 or later|Non-uniform memory access library used by MindSpore|
|[LLVM](#installing-llvm-optional)|12.0.1|Compiler framework used by MindSpore (optional, mandatory for graph kernel fusion and sparse computing)|

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

### Installing wheel setuptools PyYAML and Numpy

After installing Python, run the following command to install them.

```bash
pip install wheel
pip install -U setuptools
pip install pyyaml
pip install "numpy>=1.19.3,<=1.26.4"
```

The Numpy version used in the runtime environment must be no less than the Numpy version in the compilation environment to ensure the normal use of Numpy related capabilities in the framework.

### Installing GCC git tclsh patch and NUMA

Run the following commands to install GCC, git, tclsh, patch and NUMA.

```bash
sudo apt-get install gcc-7 git tcl patch libnuma-dev -y
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

### Installing CMake

Run the following command to install [CMake](https://cmake.org/).

```bash
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt-get install cmake -y
```

### Installing LLVM-optional

Run the following command to install [LLVM](https://llvm.org/).

```bash
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo add-apt-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-12 main"
sudo apt-get update
sudo apt-get install llvm-12-dev -y
```

## Downloading the Source Code from the Code Repository

```bash
git clone -b v2.6.0-rc1 https://gitee.com/mindspore/mindspore.git
```

## Compiling MindSpore

Go to the root directory of mindspore, then run the compilation script.

```bash
cd mindspore
bash build.sh -e cpu -j4 -S on
```

Where:

- If the compiler performance is good, add `-j{Number of threads}` to increase the number of threads. For example, `bash build.sh -e cpu -j12`.
- By default, the dependent source code is downloaded from GitHub. When -S is set to `on`, the source code is downloaded from the corresponding Gitee image.
- For details about how to use `build.sh`, see the script header description.

## Installing MindSpore

```bash
pip install output/mindspore-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

When the network is connected, dependencies of MindSpore are automatically downloaded during the .whl package installation. For details about dependencies, see required_package in the [setup.py](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/setup.py). In other cases, install the dependencies by yourself.

## Installation Verification

Execute the following command:

```bash
python -c "import mindspore;mindspore.set_device(device_target='CPU');mindspore.run_check()"
```

The output should be like:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed on platform [CPU] successfully!
```

It means MindSpore has been installed successfully.

## Version Update

After successfully executing the compile script `build.sh` in the root path of the source code, find the whl package in path `output`, and use the following command to update your version.

```bash
pip install --upgrade mindspore*.whl
```
