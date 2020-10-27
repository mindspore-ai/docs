# Conda方式安装MindSpore

<!-- TOC -->

- [Conda方式安装MindSpore](#conda方式安装mindspore)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装Conda](#安装conda)
    - [添加Conda清华镜像源](#添加conda清华镜像源)
    - [创建并激活Conda环境](#创建并激活conda环境)
    - [安装MindSpore](#安装mindspore)
    - [验证安装是否成功](#验证安装是否成功)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_install_conda.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

本文档介绍如何在CPU环境的Linux系统上，使用Conda方式快速安装MindSpore。

## 确认系统环境信息

- 确认安装Ubuntu18.04是64位操作系统。
- 确认安装[GCC](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)为7.3.0版本。

## 安装Conda

- 官网下载地址：[X86 Anaconda](https://www.anaconda.com/distribution/) 或 [X86 Miniconda](https://docs.conda.io/en/latest/miniconda.html)

- 清华镜像源下载地址：[X86 Anaconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.02-Linux-x86_64.sh)

## 添加Conda清华镜像源

从清华源镜像源下载Conda安装包的可跳过此步操作。

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

## 创建并激活Conda环境

```bash
conda create -n mindspore python=3.7.5
conda activate mindspore
```

## 安装MindSpore

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/cpu/ubuntu_x86/mindspore-{version}-cp37-cp37m-linux_{arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://mirrors.huaweicloud.com/repository/pypi/simple
```

> `{version}`表示MindSpore版本号，例如下载1.0.1版本MindSpore时，`{version}`应写为1.0.1。  
> `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为`x86_64`。如果系统是ARMv8架构64位，则写为`aarch64`。

## 验证安装是否成功

```bash
python -c "import mindspore;print(mindspore.__version__)"
```

如果输出MindSpore版本号，说明MindSpore安装成功了，如果输出`No module named 'mindspore'`说明未成功安装。
