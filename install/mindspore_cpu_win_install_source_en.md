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

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.10/install/mindspore_cpu_win_install_source_en.md)

This document describes how to quickly install MindSpore by source code in a Windows system with a CPU environment.

For the detailed steps, see the third-party dependency software installation section in the [Installing MindSpore Using Source Code Build on Windows (CPU)](https://www.mindspore.cn/news/newschildren?id=364) provided by the community. Thanks to the community member [lvmingfu](https://gitee.com/lvmingfu) for sharing.

## System Environment Information Confirmation

- Ensure that Windows 10 is installed with the x86 architecture 64-bit operating system.
- Ensure that [Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/zh-CN/download/details.aspx?id=48145) is installed.
- Ensure that [git](https://github.com/git-for-windows/git/releases/download/v2.29.2.windows.2/Git-2.29.2.2-64-bit.exe) tool is installed. And make sure you put the git directory, say if it's installed in `d:\Git`, then add `d:\Git\usr\bin` directory into environment variable `Path`.
- Ensure that [MinGW-W64 GCC-7.3.0](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/7.3.0/threads-posix/seh/x86_64-7.3.0-release-posix-seh-rt_v5-rev0.7z) is installed.
- Ensure that path of source code does not include special characters (Chinese, Janpanese characters etc.). After installing, add the path `MinGW\bin`to the environment variable PATH.For example, the installation directory is in `D:\gcc`, then you will need to add `D:\gcc\MinGW\bin` to the system environment variable PATH.
- Ensure that [CMake 3.18.3](https://github.com/Kitware/Cmake/releases/tag/v3.18.3) is installed.
- Ensure that path of source code does not include special characters (Chinese, Janpanese characters etc.). After installing, add the path of `cmake.exe` to the environment variable PATH.
- Ensure that you have Python versions between 3.7 to 3.9 installed. If not installed, follow the links to [Python official website](https://www.python.org/downloads/windows/) or [Huawei Cloud](https://repo.huaweicloud.com/python/) to download and install Python.
    - Ensure that path of source code does not include special characters (Chinese, Janpanese characters etc.). After installing, add the path of `python.exe` to the environment variable Path. The `Scripts` folder in the same directory of `python.exe` is the pip file that comes with Python, and you also need to add the path of the pip file to the environment variable Path.

> Note: Python 3.7.5 is the oldest version that supports compiling MindSpore for Windows from source code.

- Ensure that [wheel 0.32.0 and later](https://pypi.org/project/wheel/) is installed.

## Downloading Source Code from Code Repository

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.10
```

## Compiling MindSpore

Run the following command in the root directory of the source code to compile MindSpore:

```bash
call build.bat
```

## Installing MindSpore

```bash
for %x in (output\mindspore*.whl) do pip install %x -i https://pypi.tuna.tsinghua.edu.cn/simple
```

When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/r1.10/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/r1.10/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.10/requirements.txt).

## Installation Verification

```bash
cd ..
python -c "import mindspore;mindspore.run_check()"
```

Importing MindSpore installed by pip should be done outside MindSpore source code base directory, as Python for Windows processes environment paths differently.
The outputs should be the same as:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

It means MindSpore has been installed successfully.

## Version Update

Use the following command if you need to update the MindSpore version:

- Update online directly

    ```bash
    pip install --upgrade mindspore
    ```

- Update after source code compilation

    After successfully executing the compile script `build.bat` in the source code of the root directory, find the .whl package generated by compilation in directory `output`, and use the following command to update your version.

```bash
pip install --upgrade mindspore-{version}-{python_version}-win_amd64.whl
```
