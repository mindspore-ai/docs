# Installing MindSpore in CPU by Source Code (Windows)

<!-- TOC -->

- [Installing MindSpore in CPU by Source Code (Windows)](#installing-mindspore-in-cpu-by-source-code-windows)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Downloading Source Code from Code Repository](#downloading-source-code-from-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installing-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.3/install/mindspore_cpu_win_install_source_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

This document describes how to quickly install MindSpore by source code in a Windows system with a CPU environment.

## System Environment Information Confirmation

- Confirm that Windows 10 is installed with the x86 architecture 64-bit operating system.
- Confirm that [Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/zh-CN/download/details.aspx?id=48145) is installed.
- Confirm that [git](https://github.com/git-for-windows/git/releases/download/v2.29.2.windows.2/Git-2.29.2.2-64-bit.exe) tool is installed.
    - If git was not installed in `ProgramFiles`, you will need to set environment variable to where `patch.exe` is allocated. For example, when git was install in `D:\git`, `set MS_PATCH_PATH=D:\git\usr\bin`.
- Confirm that [MinGW-W64 GCC-7.3.0](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/7.3.0/threads-posix/seh/x86_64-7.3.0-release-posix-seh-rt_v5-rev0.7z) is installed.
    - Ensure that path of source code does not include special characters (Chinese, Janpanese characters etc.). After installing, add the path `MinGW\bin`to the environment variable PATH.For example, the installation directory is in `D:\gcc`, then you will need to add `D:\gcc\MinGW\bin` to the system environment variable PATH.
- Confirm that [CMake 3.18.3](https://github.com/Kitware/Cmake/releases/tag/v3.18.3) is installed.
    - Ensure that path of source code does not include special characters (Chinese, Janpanese characters etc.). After installing, add the path of `cmake.exe` to the environment variable PATH.
- Confirm that [ActivePerl 5.28.1.2801 or later](https://downloads.activestate.com/ActivePerl/releases/5.28.1.2801/ActivePerl-5.28.1.2801-MSWin32-x64-24563874.exe) is installed.
- Confirm that [Python 3.7.5](https://www.python.org/ftp/python/3.7.5/python-3.7.5-amd64.exe) is installed.
    - Ensure that path of source code does not include special characters (Chinese, Janpanese characters etc.). After installing, add the path of `python.exe` to the environment variable PATH. The `Scripts` folder in the same directory of `python.exe` is the pip file that comes with Python, you also need to add the path of the pip file to the environment variable PATH.

## Downloading Source Code from Code Repository

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.3
```

## Compiling MindSpore

Run the following command in the root directory of the source code to compile MindSpore:

```bash
call build.bat
```

## Installing MindSpore

```bash
pip install output/mindspore-{version}-cp37-cp37m-win_amd64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/r1.3/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/mindspore/tree/r1.3/model_zoo). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.3/requirements.txt).
- `{version}` denotes the version of MindSpore. For example, when you are installing MindSpore 1.1.0, `{version}` should be 1.1.0.

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

- Update after source code compilation

    After successfully executing the compile script `build.bat` in the root path of the source code, find the whl package in path `output`, use the following command to update your version.

```bash
pip install --upgrade mindspore-{version}-cp37-cp37m-win_amd64.whl
```
