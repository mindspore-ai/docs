# Installing MindSpore in CPU by Source Code

<!-- TOC -->

- [Installing MindSpore in CPU by Source Code](#installing-mindspore-in-cpu-by-source-code)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Downloading Source Code from Code Repository](#downloading-source-code-from-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)
    - [Installing MindArmour](#installing-mindarmour)
    - [Installing MindSpore Hub](#installing-mindspore-hub)

<!-- /TOC -->

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/install/mindspore_cpu_install_source_en.md)

This document describes how to quickly install MindSpore by source code in a Linux system with a CPU environment.

## System Environment Information Confirmation

- Confirm that Ubuntu 18.04 is installed with 64-bit operating system.
- Confirm that [GCC 7.3.0](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz) is installed.
- Confirm that [Python 3.7.5](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) is installed.
- Confirm that [OpenSSL 1.1.1 or later](https://github.com/openssl/openssl.git) is installed.
    - Set system variable `export OPENSSL_ROOT_DIR="OpenSSL installation directory"` after installation.
- Confirm that [CMake 3.18.3 or later](https://cmake.org/download/) is installed.
    - Add the path where the executable file `cmake` stores to the environment variable PATH.
- Confirm that [wheel 0.32.0 or later](https://pypi.org/project/wheel/) is installed.
- Confirm that [patch 2.5 or later](http://ftp.gnu.org/gnu/patch/) is installed.
    - Add the path where the executable file `patch` stores to the environment variable PATH.
- Confirm that the git tool is installed.  
    If not, use the following command to install it:

    ```bash
    apt-get install git
    ```

## Downloading Source Code from Code Repository

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.0
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
chmod +x build/package/mindspore-{version}-cp37-cp37m-linux_{arch}.whl
pip install build/package/mindspore-{version}-cp37-cp37m-linux_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.0/requirements.txt))，In other cases, you need to manually install dependency items.
- `{version}` denotes the version of MindSpore. For example, when you are downloading MindSpore 1.0.1, `{version}` should be 1.0.1.
- `{arch}` denotes the system architecture. For example, the Linux system you are using is x86 architecture 64-bit, `{arch}` should be `x86_64`. If the system is ARM architecture 64-bit, then it should be `aarch64`.

## Installation Verification

```bash
python -c 'import mindspore;print(mindspore.__version__)'
```

If the MindSpore version number is output, it means that MindSpore is installed successfully, and if the output is `No module named'mindspore'`, it means that the installation was not successful.

## Version Update

Using the following command if you need update MindSpore version:

- Update online

    ```bash
    pip install --upgrade mindspore
    ```

- Update after source code compilation

    After successfully executing the compile script `build.sh` in the root path of the source code, find the `whl` package in path `build/package`, use the following command to update your version.

    ```bash
    pip install --upgrade mindspore-{version}-cp37-cp37m-linux_{arch}.whl
    ```

## Installing MindArmour

If you need to conduct AI model security research or enhance the security of the model in you applications, you can install MindArmour.

For more details, please refer to [MindArmour](https://gitee.com/mindspore/mindarmour/blob/master/README.md).

## Installing MindSpore Hub

If you need to access and experience MindSpore pre-trained models quickly, you can install MindSpore Hub.

For more details, please refer to [MindSpore Hub](https://gitee.com/mindspore/hub/blob/master/README.md).
