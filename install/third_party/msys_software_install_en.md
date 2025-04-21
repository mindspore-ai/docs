# Installing MSYS2 Software on Windows

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/install/third_party/msys_software_install_en.md)

This document describes the steps on how to install the MSYS2 software on a Windows system.

## Software Background

FFmpeg software, as a third-party software introduced by MindSpore, needs to rely on MSYS2 software for source code compilation on Windows platform.

## Installation Steps

- Download the `MSYS2` software from the [MSYS2 official website](https://www.msys2.org/) and install it(it is recommended to install it in a place with sufficient disk space).
- After installing the `MSYS2` software, add the software to the PATH in the system environment variable.
- Modify the code in the `msys2_shell.cmd` startup script in the msys64 directory by changing `rem set MSYS2_PATH_TYPE=inherit` to `set MSYS2_PATH_TYPE=inherit`.
- Open the `MSYS2 MSYS` execution terminal in the start menu and execute the following commands to install ffmpeg compilation dependencies and tools.

```bash
pacman -S mingw-w64-x86_64-toolchain
pacman -S mingw-w64-x86_64-yasm
pacman -S mingw-w64-x86_64-SDL2
pacman -S mingw-w64-x86_64-fdk-aac
pacman -S mingw-w64-x86_64-x264
pacman -S mingw-w64-x86_64-x265
pacman -S mingw-w64-x86_64-gcc
pacman -S make diffutils pkg-config git
```