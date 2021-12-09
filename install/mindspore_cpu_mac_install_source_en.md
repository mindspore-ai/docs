# Installing MindSpore in CPU by Source Code (macOS)

<!-- TOC -->

- [Installing MindSpore in CPU by Source Code (macOS)](#installing-mindspore-in-cpu-by-source-code-macos)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Downloading Source Code from Code Repository](#downloading-source-code-from-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_mac_install_source_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

This document describes how to quickly install MindSpore by compiling source code in a macOS system with a CPU environment.

## System Environment Information Confirmation

- Ensure macOS is installed, currently support both X86 and ARM64 architecture with 10.15(Catalina) and 11.3(Big Sur) verified.

- Ensure that [Xcode](https://developer.apple.com/xcode/) is installed, where Xcode 10.2 and Xcode 12.5 are verified.

- Ensure that Command Line Tools for Xcode is installed.
     If not, use the following command to install it:

    ```bash
    xcode-select --install
    sudo xcode-select -switch /Library/Developer/CommandLineTools
    ```

- Ensure that Python 3.9 is installed. If not installed, download and install Python from:

    - Python (64-bit)：[Python official website](https://www.python.org/downloads/macos/) or [HUAWEI CLOUD](https://repo.huaweicloud.com/python/).

- Ensure that [wheel 0.32.0 or later](https://pypi.org/project/wheel/) is installed.

- Ensure that [CMake 3.18.3 or later](https://cmake.org/download/) is installed.
    - Add the path where the executable file `cmake` stores to the environment variable PATH.

- Ensure that [patch 2.5 or later](http://ftp.gnu.org/gnu/patch/) is installed.
    - Add the path where the executable file `patch` stores to the environment variable PATH.

## Downloading Source Code from Code Repository

```bash
git clone https://gitee.com/mindspore/mindspore.git
```

## Compiling MindSpore

Run the following command in the root directory of the source code to compile MindSpore:

```bash
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export SDKROOT="`xcrun --show-sdk-path`"
bash build.sh -e cpu -S on -j4
```

Of which,

- If the compiler performance is good, you can add -j{Number of threads} in to script to increase the number of threads. For example, `bash build.sh -e cpu -S on -j12`.

## Installing MindSpore

```bash
pip install output/mindspore-{version}-{python_version}-macosx_{platform_version}_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

If there is any compile error when installing scipy package, please use following command to install scipy package first, then install mindspore package as normal.

```bash
pip install --pre -i https://pypi.anaconda.org/scipy-wheels-nightly/simple scipy
```

Of which,

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/master/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt).
- `{version}` specifies the MindSpore version number. For example, when installing MindSpore 1.6.0, set `{version}` to 1.6.0.
- `{platform_version}` specifies the macOS version number. For example, when installing macOS 10.15, set `{platform_version}` to `10_15`.
- `{arch}` denotes the system architecture. For example, the macOS you are using is x86 architecture 64-bit, `{arch}` should be `x86_64`. If the system is ARM architecture 64-bit, then it should be `arm64`.
- `{python_version}` spcecifies the python version for which MindSpore is built. If Python3.9.0 is used, it should be `cp39-cp39`.

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
    pip install --upgrade mindspore-{version}-{python_version}-macosx_{platform_version}_{arch}.whl
    ```
