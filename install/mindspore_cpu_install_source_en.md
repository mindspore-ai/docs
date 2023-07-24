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

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/install/mindspore_cpu_install_source_en.md)

This document describes how to quickly install MindSpore by source code in a Linux system with a CPU environment.

## System Environment Information Confirmation

- Confirm that the 64-bit operating system is installed, where Ubuntu 18.04 are verified.
- Confirm that [GCC 7.3.0](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz) is installed.
- Confirm that [gmp 6.1.2](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz) is installed.
- Confirm that [Python 3.7.5](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) is installed.
- Confirm that [CMake 3.18.3 or later](https://cmake.org/download/) is installed.
    - Add the path where the executable file `cmake` stores to the environment variable PATH.
- Confirm that [wheel 0.32.0 or later](https://pypi.org/project/wheel/) is installed.
- Confirm that [patch 2.5 or later](http://ftp.gnu.org/gnu/patch/) is installed.
    - Add the path where the executable file `patch` stores to the environment variable PATH.
- Confirm that [NUMA 2.0.11 or later](https://github.com/numactl/numactl) is installed.
     If not, use the following command to install it:

    ```bash
    apt-get install libnuma-dev
    ```

- Confirm that the git tool is installed.  
    If not, use the following command to install it:

    ```bash
    apt-get install git
    ```

## Downloading Source Code from Code Repository

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.3
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
pip install output/mindspore-{version}-cp37-cp37m-linux_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/r1.3/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.3/requirements.txt).
- `{version}` denotes the version of MindSpore. For example, when you are installing MindSpore 1.1.0, `{version}` should be 1.1.0.
- `{arch}` denotes the system architecture. For example, the Linux system you are using is x86 architecture 64-bit, `{arch}` should be `x86_64`. If the system is ARM architecture 64-bit, then it should be `aarch64`.

## Installation Verification

```bash
python -c "import mindspore;mindspore.run_check()"
```

The outputs should be the same as:

```text
mindspore version: __version__
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
    pip install --upgrade mindspore-{version}-cp37-cp37m-linux_{arch}.whl
    ```
