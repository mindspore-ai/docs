# 源码编译方式安装MindSpore CPU版本

<!-- TOC -->

- [源码编译方式安装MindSpore CPU版本](#源码编译方式安装mindspore-cpu版本)
    - [确认系统环境信息](#确认系统环境信息)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证安装是否成功](#验证安装是否成功)
    - [升级MindSpore版本](#升级mindspore版本)
    - [安装MindArmour](#安装mindarmour)
    - [安装MindSpore Hub](#安装mindspore-hub)
    - [安装MindQuantum](#安装mindquantum)

<!-- /TOC -->

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.2/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/install/mindspore_cpu_install_source.md)

本文档介绍如何在CPU环境的Linux系统上，使用源码编译方式快速安装MindSpore。

详细步骤可以参考社区提供的实践——[在Ubuntu（CPU）上进行源码编译安装MindSpore](https://www.mindspore.cn/news/newschildren?id=365)，在此感谢社区成员[damon0626](https://gitee.com/damon0626)的分享。

## 确认系统环境信息

- 确认安装Ubuntu 18.04是64位操作系统。
- 确认安装[GCC 7.3.0版本](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)。
- 确认安装[Python 3.7.5版本](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz)。
- 确认安装[OpenSSL 1.1.1及以上版本](https://github.com/openssl/openssl.git)。
    - 安装完成后设置环境变量`export OPENSSL_ROOT_DIR=“OpenSSL安装目录”`。
- 确认安装[CMake 3.18.3版本](https://cmake.org/download/)。
    - 安装完成后需将CMake所在路径添加到系统环境变量。
- 确认安装[wheel 0.32.0及以上版本](https://pypi.org/project/wheel/)。
- 确认安装[patch 2.5及以上版本](http://ftp.gnu.org/gnu/patch/)。
    - 安装完成后需将patch所在路径添加到系统环境变量中。
- 确认安装[NUMA 2.0.11及以上版本](https://github.com/numactl/numactl)。
    如果未安装，使用如下命令下载安装：

    ```bash
    apt-get install libnuma-dev
    ```

- 确认安装git工具。  
    如果未安装，使用如下命令下载安装：

    ```bash
    apt-get install git
    ```

## 从代码仓下载源码

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.2
```

## 编译MindSpore

在源码根目录下执行如下命令。

```bash
bash build.sh -e cpu -j4
```

其中：  
如果编译机性能较好，可在执行中增加-j{线程数}来增加线程数量。如`bash build.sh -e cpu -j12`。

## 安装MindSpore

```bash
chmod +x build/package/mindspore-{version}-cp37-cp37m-linux_{arch}.whl
pip install build/package/mindspore-{version}-cp37-cp37m-linux_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中：

- 在联网状态下，安装whl包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.2/requirements.txt)），其余情况需自行安装。  
- `{version}`表示MindSpore版本号，例如安装1.1.0版本MindSpore时，`{version}`应写为1.1.0。  
- `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为`x86_64`。如果系统是ARMv8架构64位，则写为`aarch64`。

## 验证安装是否成功

```bash
python -c 'import mindspore;print(mindspore.__version__)'
```

如果输出MindSpore版本号，说明MindSpore安装成功了，如果输出`No module named 'mindspore'`说明未安装成功。

## 升级MindSpore版本

当需要升级MindSpore版本时，可执行如下命令：

- 直接在线升级

    ```bash
    pip install --upgrade mindspore
    ```

- 本地源码编译升级

    在源码根目录下执行编译脚本`build.sh`成功后，在`build/package`目录下找到编译生成的whl安装包，然后执行命令进行升级。

    ```bash
    pip install --upgrade mindspore-{version}-cp37-cp37m-linux_{arch}.whl
    ```

## 安装MindArmour

当您进行AI模型安全研究或想要增强AI应用模型的防护能力时，可以选装MindArmour。

具体安装步骤参见[MindArmour](https://gitee.com/mindspore/mindarmour/blob/r1.2/README_CN.md)。

## 安装MindSpore Hub

当您想要快速体验MindSpore预训练模型时，可以选装MindSpore Hub。

具体安装步骤参见[MindSpore Hub](https://gitee.com/mindspore/hub/blob/r1.2/README_CN.md)。

## 安装MindQuantum

当您想要搭建并训练量子神经网络，可以选装MindQuantum。

具体安装步骤参见[MindQuantum](https://gitee.com/mindspore/mindquantum/blob/r0.1/README_CN.md)。
