# Windows上安装MSYS2软件

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/install/third_party/msys_software_install.md)

本文档介绍如何在Windows系统上，安装MSYS2软件的步骤。

## 软件背景

FFmpeg软件作为MindSpore引入的三方软件，在Windows平台上进行源码编译的时候需要依赖MSYS2软件。

## 安装步骤

- 在[MSYS2官网](https://www.msys2.org/)下载`MSYS2`软件，并安装（建议安装在磁盘空间充足的地方）。
- 安装完`MSYS2`软件之后，将该软件添加到系统环境变量中的PATH里面。
- 修改msys64目录下的启动脚本`msys2_shell.cmd`中的代码，具体修改为：将`rem set MSYS2_PATH_TYPE=inherit`修改为`set MSYS2_PATH_TYPE=inherit`。
- 在开始菜单打开`MSYS2 MSYS`执行终端，执行以下命令，安装FFmpeg编译的依赖和工具。

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