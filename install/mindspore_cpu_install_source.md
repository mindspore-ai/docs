# 源码编译方式安装MindSpore CPU版本

<!-- TOC -->

- [源码编译方式安装MindSpore CPU版本](#源码编译方式安装mindspore-cpu版本)
    - [确认系统环境信息](#确认系统环境信息)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证安装是否成功](#验证安装是否成功)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.6/install/mindspore_cpu_install_source.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

本文档介绍如何在CPU环境的Linux系统上，使用源码编译方式快速安装MindSpore。

详细步骤可以参考社区提供的实践——[在Ubuntu（CPU）上进行源码编译安装MindSpore](https://www.mindspore.cn/news/newschildren?id=365)，在此感谢社区成员[damon0626](https://gitee.com/damon0626)的分享。

## 确认系统环境信息

- 确认安装64位操作系统，其中Ubuntu 18.04是经过验证的。

- 确认安装[GCC 7.3.0版本](https://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)。

- 确认安装[gmp 6.1.2版本](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz)。

- 确认安装[llvm 12.0.1版本](https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-12.0.1.tar.gz)（可选，图算融合需要）。

- 确认安装Python 3.7.5或3.9.0版本。如果未安装或者已安装其他版本的Python，可以选择下载并安装：

    - Python 3.7.5版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz)或[华为云](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz)。
    - Python 3.9.0版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz)或[华为云](https://mirrors.huaweicloud.com/python/3.9.0/Python-3.9.0.tgz)。

- 确认安装[CMake 3.18.3及以上版本](https://cmake.org/download/)。
    - 安装完成后需将CMake所在路径添加到系统环境变量。

- 确认安装[wheel 0.32.0及以上版本](https://pypi.org/project/wheel/)。

- 确认安装[patch 2.5及以上版本](https://ftp.gnu.org/gnu/patch/)。
    - 安装完成后需将patch所在路径添加到系统环境变量中。

- 确认安装[NUMA 2.0.11及以上版本](https://github.com/numactl/numactl)。
    如果未安装，使用如下命令下载安装：

    ```bash
    apt-get install libnuma-dev
    ```

- 确认安装git工具。
    如果未安装，使用如下命令下载安装：

    ```bash
    apt-get install git # for linux distributions using apt, e.g. ubuntu
    yum install git     # for linux distributions using yum, e.g. centos
    ```

## 从代码仓下载源码

```bash
git clone https://gitee.com/mindspore/mindspore.git
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
pip install output/mindspore-{version}-{python_version}-linux_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中：

- 在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/r1.6/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/r1.6/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.6/requirements.txt)。
- `{version}`表示MindSpore版本号，例如安装1.5.0-rc1版本MindSpore时，`{version}`应写为1.5.0rc1。
- `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为`x86_64`。如果系统是ARMv8架构64位，则写为`aarch64`。
- `{python_version}`表示用户的Python版本，Python版本为3.7.5时，`{python_version}`应写为`cp37-cp37m`。Python版本为3.9.0时，则写为`cp39-cp39`。

## 验证安装是否成功

```bash
python -c "import mindspore;mindspore.run_check()"
```

如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

说明MindSpore安装成功了。

## 升级MindSpore版本

当需要升级MindSpore版本时，可执行如下命令：

- 直接在线升级

    ```bash
    pip install --upgrade mindspore
    ```

- 本地源码编译升级

    在源码根目录下执行编译脚本`build.sh`成功后，在`output`目录下找到编译生成的whl安装包，然后执行命令进行升级。

    ```bash
    pip install --upgrade mindspore-{version}-{python_version}-linux_{arch}.whl
    ```
