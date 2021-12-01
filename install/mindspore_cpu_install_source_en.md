# Installing MindSpore in CPU by Source Code

<!-- TOC -->

- [Installing MindSpore in CPU by Source Code](#installing-mindspore-in-cpu-by-source-code)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Downloading Source Code from Code Repository](#downloading-source-code-from-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/install/mindspore_cpu_install_source_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

This document describes how to quickly install MindSpore by source code in a Linux system with a CPU environment.

## System Environment Information Confirmation

- Ensure that the 64-bit operating system is installed, where Ubuntu 18.04 are verified.

- Ensure that [GCC 7.3.0](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz) is installed.

- Ensure that [gmp 6.1.2](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz) is installed.

- Ensure that Python 3.7.5 or 3.9.0 is installed. If not installed, download and install Python from:

    - Python 3.7.5 (64-bit): [Python official website](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) or [HUAWEI CLOUD](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz).
    - Python 3.9.0 (64-bit): [Python official website](https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz) or [HUAWEI CLOUD](https://mirrors.huaweicloud.com/python/3.9.0/Python-3.9.0.tgz).

- Ensure that [CMake 3.18.3 or later](https://cmake.org/download/) is installed.
    - Add the path where the executable file `cmake` stores to the environment variable PATH.

- Ensure that [wheel 0.32.0 or later](https://pypi.org/project/wheel/) is installed.

- Ensure that [patch 2.5 or later](http://ftp.gnu.org/gnu/patch/) is installed.
    - Add the path where the executable file `patch` stores to the environment variable PATH.

- Ensure that [NUMA 2.0.11 or later](https://github.com/numactl/numactl) is installed.
     If not, use the following command to install it:

    ```bash
    apt-get install libnuma-dev
    ```

- Ensure that the git is installed, otherwise execute the following command to install git:

    ```bash
    apt-get install git # for linux distributions using apt, e.g. ubuntu
    yum install git     # for linux distributions using yum, e.g. centos
    ```

## Downloading Source Code from Code Repository

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.5
```

## Compiling MindSpore

Run the following command in the root directory of the source code to compile MindSpore:

```bash
bash build.sh -e cpu -j4
```

Of which,

- If the compiler performance is strong, you can add -j{Number of threads} in to script to increase the number of threads. For example, `bash build.sh -e cpu -j12`.

## Installing MindSpore

```bash
pip install output/mindspore-{version}-{python_version}-linux_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/r1.5/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/r1.5/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.5/requirements.txt).
- `{version}` specifies the MindSpore version number. For example, when installing MindSpore 1.5.0-rc1, set `{version}` to 1.5.0rc1.
- `{arch}` denotes the system architecture. For example, the Linux system you are using is x86 architecture 64-bit, `{arch}` should be `x86_64`. If the system is ARM architecture 64-bit, then it should be `aarch64`.
- `{python_version}` spcecifies the python version for which MindSpore is built. If you wish to use Python3.7.5,`{python_version}` should be `cp37-cp37m`. If Python3.9.0 is used, it should be `cp39-cp39`.

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

- Update online

    ```bash
    pip install --upgrade mindspore
    ```

- Update after source code compilation

    After successfully executing the compile script `build.sh` in the root path of the source code, find the whl package in path `output`, use the following command to update your version.

    ```bash
    pip install --upgrade mindspore-{version}-{python_version}-linux_{arch}.whl
    ```
