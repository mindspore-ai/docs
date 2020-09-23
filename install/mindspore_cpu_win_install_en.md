# MindSpore Installation Guide

This document describes how to quickly install MindSpore in a Windows system with a CPU environment.

<!-- TOC -->

- [MindSpore Installation Guide](#mindspore-installation-guide)
    - [Environment Requirements](#environment-requirements)
        - [System Requirements and Software Dependencies](#system-requirements-and-software-dependencies)
        - [(Optional) Installing Conda](#optional-installing-conda)
    - [Installation Guide](#installation-guide)
        - [Installing Using Executable Files](#installing-using-executable-files)
        - [Installing Using the Source Code](#installing-using-the-source-code)

<!-- /TOC -->

## Environment Requirements

### System Requirements and Software Dependencies

| Version | Operating System | Executable File Installation Dependencies | Source Code Compilation and Installation Dependencies |
| ---- | :--- | :--- | :--- |
| MindSpore 1.0.0 | Windows 10 x86_64 | - [Python](https://www.python.org/downloads/) 3.7.5 <br> - For details about other dependency items, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.0/requirements.txt). | **Compilation dependencies:**<br> - [Python](https://www.python.org/downloads/) 3.7.5 <br> - [MinGW-W64 GCC-7.3.0](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/7.3.0/threads-posix/seh/x86_64-7.3.0-release-posix-seh-rt_v5-rev0.7z) x86_64-posix-seh <br> - [ActivePerl](http://downloads.activestate.com/ActivePerl/releases/5.24.3.2404/ActivePerl-5.24.3.2404-MSWin32-x64-404865.exe) 5.24.3.2404 <br> - [CMake](https://cmake.org/download/) 3.14.1 <br> - [wheel](https://pypi.org/project/wheel/) >= 0.32.0 <br> **Installation dependencies:**<br> same as the executable file installation dependencies. |

- When the network is connected, dependency items in the `requirements.txt` file are automatically downloaded during .whl package installation. In other cases, you need to manually install dependency items.

### (Optional) Installing Conda

1. Download the Conda installation package from the following path:

   - [X86 Anaconda](https://www.anaconda.com/distribution/) or [X86 Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Open Anaconda Prompt from the Windows start menu after installation.
3. Create and activate the Python environment.

    ```bash
    conda create -n {your_env_name} python=3.7.5
    conda activate {your_env_name}
    ```

> Conda is a powerful Python environment management tool. Beginners are adviced to check related information on the Internet.

## Installation Guide

### Installing Using Executable Files

1. Download the .whl package the [MindSpore website](https://www.mindspore.cn/versions/en). It is recommended to perform SHA-256 integrity verification first and run the following command to install MindSpore:

    ```bash
    pip install mindspore-{version}-cp37-cp37m-win_{arch}.whl
    ```

2. Run the following command. If no loading error message such as `No module named 'mindspore'` is displayed, the installation is successful.

    ```bash
    python
    import mindspore
    ```

### Installing Using the Source Code

1. Download the source code from the code repository.

    ```bash
    git clone https://gitee.com/mindspore/mindspore.git -b r1.0
    ```

2. Run the following command in the root directory of the source code to compile MindSpore:

    ```bash
    call build.bat
    ```
    > - Ensure that path of source code does not include special characters (Chinese, Janpanese characters etc.).
    > - Before running the preceding command, ensure that the paths of `mingw64\bin\` and the executable files `cmake` have been added to the environment variable PATH.
    > - If git was not installed in `ProgramFiles`, you will need to set environment variable to where `patch.exe` is allocated. For example, when git was install in `D:\git`, `set MS_PATCH_PATH=D:\git\usr\bin`.
    > - In the `build.bat` script, the git clone command will be executed to obtain the code in the third-party dependency database. Ensure that the network settings of Git are correct.
    > - If the compiler performance is strong, you can add -j{Number of threads} in to script to increase the number of threads(Default 6). For example, `call build.bat 12`.
    > - Before running the preceding command, ensure that [Visual C ++ Redistributable for Visual Studio 2015](https://www.microsoft.com/zh-CN/download/details.aspx?id=48145) is installed.

3. Run the following command to install MindSpore:

    ```bash
    pip install build/package/mindspore-{version}-cp37-cp37m-win_{arch}.whl
    ```

4. Run the following command. If no loading error message such as `No module named 'mindspore'` is displayed, the installation is successful.

    ```bash
    python
    import mindspore
    ```
