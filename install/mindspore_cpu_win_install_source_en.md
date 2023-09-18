# Installing MindSpore in CPU by Source Code-Windows

<!-- TOC -->

- [Installing MindSpore in CPU by Source Code-Windows](#installing-mindspore-in-cpu-by-source-code-windows)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Downloading Source Code from Code Repository](#downloading-source-code-from-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.1/install/mindspore_cpu_win_install_source_en.md)

This document describes how to install MindSpore by compiling source code on Windows system in a CPU environment.

## System Environment Information Confirmation

- Caurrently supports Windows 10/11 with the x86 architecture 64-bit operating system.
- Ensure that [Microsoft Visual Studio Community 2019](https://learn.microsoft.com/zh-cn/visualstudio/releases/2019/release-notes) is installed. Workloads select `Desktop development with C++` and `Universal Windows Platform development`, Individual components select `Select C++ CMake tools for Windows`.
- Ensure that [git](https://github.com/git-for-windows/git/releases/download/v2.29.2.windows.2/Git-2.29.2.2-64-bit.exe) tool is installed. Meanwhile, put Git insitall directory into `Path` environment variable. If git is installed in `D:\Program Files\Git`,then add `D:\Program Files\Git\usr\bin` into `Path` variable.
- Ensure that [CMake 3.18.3](https://cmake.org/files/v3.18/cmake-3.18.3-win64-x64.msi) is installed. After installing, add the path of `cmake.exe` to the environment variable `Path`.
- Ensure that you have Python(>=3.7.5) installed. If not installed, follow the links to [Python official website](https://www.python.org/downloads/windows/) or [Huawei Cloud](https://repo.huaweicloud.com/python/) to download and install Python.
- Ensure that [wheel 0.32.0 and later](https://pypi.org/project/wheel/) is installed.

## Downloading Source Code from Code Repository

```bash
git clone -b r2.1 https://gitee.com/mindspore/mindspore.git
```

## Compiling MindSpore

Run the following command in the root directory of the source code to compile MindSpore:

```bash
call build.bat ms_vs_cpu
```

## Installing MindSpore

```bash
for %x in (output\mindspore*.whl) do pip install %x -i https://pypi.tuna.tsinghua.edu.cn/simple
```

When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/r2.1/setup.py).) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/r2.1/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r2.1/requirements.txt).

## Installation Verification

```bash
cd ..
python -c "import mindspore;mindspore.set_context(device_target='CPU');mindspore.run_check()"
```

Importing MindSpore installed by pip should be done outside MindSpore source code base directory, as Python for Windows processes environment paths differently.
The outputs should be the same as:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed on platform [CPU] successfully!
```

It means MindSpore has been installed successfully.

## Version Update

After successfully executing the compile script `build.bat` in the root path of the source code, find the whl package in path `output`, and use the following command to update your version.

```bash
pip install --upgrade mindspore*.whl
```
