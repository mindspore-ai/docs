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

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/install/mindspore_cpu_win_install_source_en.md)

This document describes how to quickly install MindSpore by source code in a Windows system with a CPU environment.

## System Environment Information Confirmation

- Confirm that Windows 10 is installed with x86 architecture 64-bit operating system.
- Confirm that [Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/zh-CN/download/details.aspx?id=48145) is installed.
- Confirm that [git](https://github.com/git-for-windows/git/releases/download/v2.29.2.windows.2/Git-2.29.2.2-64-bit.exe) tool is installed.
    - If git was not installed in `ProgramFiles`, you will need to set environment variable to where `patch.exe` is allocated. For example, when git was install in `D:\git`, `set MS_PATCH_PATH=D:\git\usr\bin`.
- Confirm that [MinGW-W64 GCC-7.3.0](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/7.3.0/threads-posix/seh/x86_64-7.3.0-release-posix-seh-rt_v5-rev0.7z) is installed.
    - Ensure that path of source code does not include special characters (Chinese, Janpanese characters etc.). After installing, add the path `MinGW\bin`to the environment variable PATH.For example, the installation directory is in `D:\gcc`, then you will need to add `D:\gcc\MinGW\bin` to the system environment variable PATH.
- Confirm that [CMake 3.18.3](https://cmake.org/download/) is installed.
    - Ensure that path of source code does not include special characters (Chinese, Janpanese characters etc.). After installing, add the path of `cmake.exe` to the environment variable PATH.
- Confirm that [ActivePerl 5.28.1.2801](https://downloads.activestate.com/ActivePerl/releases/5.28.1.2801/ActivePerl-5.28.1.2801-MSWin32-x64-24563874.exe) is installed.
- Confirm that [Python 3.7.5](https://www.python.org/ftp/python/3.7.5/python-3.7.5-amd64.exe) is installed.
    - Ensure that path of source code does not include special characters (Chinese, Janpanese characters etc.). After installing, add the path of `python.exe` to the environment variable PATH. The `Scripts` folder in the same directory of `python.exe` is the pip file that comes with Python, you also need to add the path of the pip file to the environment variable PATH.
- Confirm that [wheel 0.32.0 or later](https://pypi.org/project/wheel/) is installed.

## Downloading Source Code from Code Repository

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.0
```

## Compiling MindSpore

Run the following command in the root directory of the source code to compile MindSpore:

```bash
call build.bat
```

## Installing MindSpore

```bash
pip install build/package/mindspore-{version}-cp37-cp37m-win_amd64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.0/requirements.txt)). In other cases, you need to manually install dependency items.
- `{version}` denotes the version of MindSpore. For example, when you are downloading MindSpore 1.0.1, `{version}` should be 1.0.1.

## Installation Verification

```bash
python -c "import mindspore;print(mindspore.__version__)"
```

If the MindSpore version number is output, it means that MindSpore is installed successfully, and if the output is `No module named 'mindspore'`, it means that the installation was not successful.

## Version Update

Using the following command if you need update MindSpore version:

- Update online

    ```bash
    pip install --upgrade mindspore
    ```

- Update after source code compilation

    After successfully executing the compile script `build.bat` in the root path of the source code, find the `whl` package in path `build/package`, use the following command to update your version.

```bash
pip install --upgrade mindspore-{version}-cp37-cp37m-win_amd64.whl
```
