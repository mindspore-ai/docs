# Installing MindSpore in CPU by Source Code-macOS

<!-- TOC -->

- [Installing MindSpore in CPU by Source Code-macOS](#installing-mindspore-in-cpu-by-source-code-macos)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Installing Python](#installing-python)
    - [Downloading Source Code from Code Repository](#downloading-source-code-from-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/install/mindspore_cpu_mac_install_source_en.md)

## System Environment Information Confirmation

- Ensure that [Xcode](https://xcodereleases.com/) (>=12.4 and <= 13.0) is installed, where 12.4(X86) and 13.0(m1) are verified.

- Ensure that `Command Line Tools for Xcode` is installed. If not, use `sudo xcode-select --install` command to install Command Line Tools.

- Ensure that [CMake 3.22.2 and the later version](https://cmake.org/download/) is installed. Use `brew install cmake` if it's not installed.

- Ensure that [patch 2.5](https://ftp.gnu.org/gnu/patch/) is installed. Use `brew install patch` if it's not installed.

- Ensure that [wheel 0.32.0 or later](https://pypi.org/project/wheel/) is installed. Use `pip install wheel` if it's not installed.

- Ensure that [PyYAML](https://pypi.org/project/pyyaml/) (>=6.0 and <= 6.0.2) is installed. Use `pip install pyyaml` if it's not installed.

- Ensure that [autoconf](https://ftp.gnu.org/gnu/autoconf/) is installed. Use `brew install autoconf` if it's not installed.

- Ensure that [Numpy](https://pypi.org/project/numpy/) (>=1.19.3 and <= 1.26.4) is installed. Use `pip install numpy` if it's not installed.

## Installing Python

Please refer to the [Python official website](https://www.python.org/) to install Python by yourself. The version must be 3.9-3.11.

After the installation is complete, you can check the Python version with the following command.

```bash
python --version
```

## Downloading Source Code from Code Repository

```bash
git clone -b v2.6.0-rc1 https://gitee.com/mindspore/mindspore.git
```

## Compiling MindSpore

Run the following command in the root directory of the source code to compile MindSpore:

```bash
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
bash build.sh -e cpu -S on -j4   # -j is a thread configuration when compiled, and if cpu performance is better, compile using multithreading, with parameters usually twice the number of CPU cores
```

## Installing MindSpore

```bash
# install prerequisites
pip install scipy

pip install output/mindspore-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

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

After successfully executing the compile script `build.sh` in the root path of the source code, find the whl package in path `output`, and use the following command to update your version.

```bash
pip install --upgrade mindspore*.whl
```
