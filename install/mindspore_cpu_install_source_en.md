# Installing MindSpore in CPU by Source Code

<!-- TOC -->

- [Installing MindSpore in CPU by Source Code](#installing-mindspore-in-cpu-by-source-code)
    - [Environment Preparation](#environment-preparation)
    - [Downloading Source Code from Code Repository](#downloading-source-code-from-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_install_source_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

This document describes how to quickly install MindSpore by source code in a Linux system with a CPU environment. The following takes Ubuntu 18.04 as an example to illustrate the steps to compile and install MindSpore.

- If you want to configure an environment that can compile MindSpore on a fresh Ubuntu 18.04, you may use [automatic installation script](https://gitee.com/mindspore/mindspore/raw/master/scripts/install/ubuntu-cpu-source.sh) for one-click configuration. The automatic installation script will install the dependencies required to compile MindSpore.

    The automatic installation script needs to replace the source list and install dependencies via APT, so it needs root privileges to execute. Use the following commands to get the automatic installation script and execute.

    ```bash
    wget https://gitee.com/mindspore/mindspore/raw/master/scripts/install/ubuntu-cpu-source.sh
    # install Python 3.7 by default
    sudo bash ./ubuntu-cpu-source.sh
    # to specify Python version
    # sudo PYTHON_VERSION=3.9 bash ./ubuntu-cpu-source.sh
    ```

    The script will:

    - Set the source list to huaweicloud source.
    - Install the compilation dependencies required by MindSpore, such as GCC, CMake, etc.
    - Install Python3 and pip3 via APT and set them as default.

    For more usage, please refer to the description at the head of the script.

- If your system has already installed some dependencies, such as Python, GCC, etc., it is recommended to install manually by referring to the following installation steps.

## Environment Preparation

The following table lists the system environment and third-party dependencies required to compile and install MindSpore.

|software|version|description|
|-|-|-|
|Ubuntu|18.04|operating system for compiling and running MindSpore|
|[Python](#install-python)|3.7.5 or 3.9.0|Python interface for MindSpore|
|[wheel](#install-wheel-and-setuptools)|0.32.0 or later|Python packaging tool used by MindSpore|
|[setuptools](#install-wheel-and-setuptools)|44.0 or later|Python package management tool used by MindSpore|
|[GCC](#install-gccgitgmptclshpatchnuma)|7.3.0|C++ compiler for compiling MindSpore|
|[git](#install-gccgitgmptclshpatchnuma)|-|source code management tools used by MindSpore|
|[CMake](#install-cmake)|3.18.3 or later|build tools for MindSpore|
|[gmp](#install-gccgitgmptclshpatchnuma)|6.1.2|multiple precision arithmetic library used by MindSpore|
|[tclsh](#install-gccgitgmptclshpatchnuma)|-|sqlite compilation dependencies for MindSpore|
|[patch](#install-gccgitgmptclshpatchnuma)|2.5 or later|source code patching tool used by MindSpore|
|[NUMA](#install-gccgitgmptclshpatchnuma)|2.0.11 or later|non-uniform memory access library used by MindSpore|
|[LLVM](#install-llvm-optional)|12.0.1|compiler framework used by MindSpore (optional, required for graph kernel fusion)|

The following are installation methods of third-party dependencies.

### Install Python

[Python](https://www.python.org/) can be installed in several ways.

- Install Python with Conda.

    Install Miniconda:

    ```bash
    cd /tmp
    curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
    bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b
    cd -
    . ~/miniconda3/etc/profile.d/conda.sh
    conda init bash
    ```

    Create a Python 3.7.5 environment:

    ```bash
    conda create -n mindspore_py37 python=3.7.5 -y
    conda activate mindspore_py37
    ```

    Or create a Python 3.9.0 environment:

    ```bash
    conda create -n mindspore_py39 python=3.9.0 -y
    conda activate mindspore_py39
    ```

- Or install Python via APT with the following command.

    ```bash
    sudo apt-get update
    sudo apt-get install software-properties-common -y
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get install python3.7 python3.7-dev python3.7-distutils python3-pip -y
    # set new installed Python as default
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.7 100
    # install pip
    python -m pip install pip -i https://repo.huaweicloud.com/repository/pypi/simple
    sudo update-alternatives --install /usr/bin/pip pip ~/.local/bin/pip3.7 100
    pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
    ```

    To install Python 3.9, just replace `3.7` with `3.9` in the command.

You can check the Python version with the following command.

```bash
python --version
```

### Install wheel and setuptools

After installing Python, use the following command to install it.

```bash
pip install wheel
pip install -U setuptools
```

### Install GCC/git/gmp/tclsh/patch/NUMA

You may install GCC, git, gmp, tclsh, patch, NUMA by the following commands.

```bash
sudo apt-get install gcc-7 git libgmp-dev tcl patch libnuma-dev -y
```

### Install CMake

You may install [CMake](https://cmake.org/) by the following commands.

```bash
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt-get install cmake -y
```

### Install LLVM (optional)

You may install [LLVM](https://llvm.org/) by the following command.

```bash
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo add-apt-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-12 main"
sudo apt-get update
sudo apt-get install llvm-12-dev -y
```

## Downloading Source Code from Code Repository

```bash
git clone https://gitee.com/mindspore/mindspore.git
```

## Compiling MindSpore

Go to the root directory of mindspore, then execute the compile script.

```bash
cd mindspore
bash build.sh -e cpu -j4 -S on
```

Of which,

- If the compiler performance is strong, you can add -j{Number of threads} in to script to increase the number of threads. For example, `bash build.sh -e cpu -j12`.
- By default, the dependent source code is downloaded from github. You may set -S option to `on` to download from the corresponding gitee mirror.
- For more usage of `build.sh`, please refer to the description at the head of the script.

## Installing MindSpore

```bash
pip install output/mindspore-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/master/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt).

## Installation Verification

```bash
python -c "import mindspore;mindspore.run_check()"
```

The output should be like:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

It means MindSpore has been installed successfully.

## Version Update

Using the following command if you need to update the MindSpore version:

- Update online

    ```bash
    pip install --upgrade mindspore
    ```

- Update after source code compilation

    After successfully executing the compile script `build.sh` in the root path of the source code, find the whl package in path `output`, use the following command to update your version.

    ```bash
    pip install --upgrade mindspore-*.whl
    ```
