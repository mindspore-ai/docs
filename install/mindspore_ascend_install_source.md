# 源码编译方式安装MindSpore Ascend版本

<!-- TOC -->

- [源码编译方式安装MindSpore Ascend版本](#源码编译方式安装mindspore-ascend版本)
    - [安装依赖软件](#安装依赖软件)
        - [安装Python](#安装python)
        - [安装昇腾AI处理器配套软件包](#安装昇腾ai处理器配套软件包)
        - [安装wheel setuptools PyYAML和Numpy](#安装wheel-setuptools-pyyaml和numpy)
        - [安装GCC](#安装gcc)
        - [安装git tclsh patch NUMA Flex](#安装git-tclsh-patch-numa-flex)
        - [安装git-lfs](#安装git-lfs)
        - [安装CMake](#安装cmake)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [配置环境变量](#配置环境变量)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/install/mindspore_ascend_install_source.md)

本文档介绍如何在Ascend环境的Linux系统上，使用源码编译方式快速安装MindSpore。

## 安装依赖软件

下表列出了编译安装MindSpore所需的系统环境和第三方依赖。

|软件名称|版本|作用|
|-|-|-|
|Ubuntu 18.04 / CentOS 7.6 / EulerOS 2.8 / openEuler 20.03 / KylinV10 SP1|-|编译和运行MindSpore的操作系统|
|[Python](#安装python)|3.9-3.11|MindSpore的使用依赖Python环境|
|[昇腾AI处理器配套软件包](#安装昇腾ai处理器配套软件包)|-|MindSpore使用的Ascend平台AI计算库|
|[wheel](#安装wheel-setuptools-pyyaml和numpy)|0.32.0及以上|MindSpore使用的Python打包工具|
|[setuptools](#安装wheel-setuptools-pyyaml和numpy)|44.0及以上|MindSpore使用的Python包管理工具|
|[PyYAML](#安装wheel-setuptools-pyyaml和numpy)|6.0-6.0.2|MindSpore里的算子编译功能依赖PyYAML模块|
|[Numpy](#安装wheel-setuptools-pyyaml和numpy)|1.19.3-1.26.4|MindSpore里的Numpy相关功能依赖Numpy模块|
|[GCC](#安装gcc)|7.3.0|用于编译MindSpore的C++编译器|
|[git](#安装git-tclsh-patch-numa-flex)|-|MindSpore使用的源代码管理工具|
|[git-lfs](#安装git-lfs)|-|MindSpore使用的源代码管理拓展工具|
|[CMake](#安装cmake)|3.22.2及以上|编译构建MindSpore的工具|
|[Flex](#安装git-tclsh-patch-numa-flex)|2.5.35及以上|MindSpore使用的词法分析器|
|[tclsh](#安装git-tclsh-patch-numa-flex)|-|MindSpore sqlite编译依赖|
|[patch](#安装git-tclsh-patch-numa-flex)|2.5及以上|MindSpore使用的源代码补丁工具|
|[NUMA](#安装git-tclsh-patch-numa-flex)|2.0.11及以上|MindSpore使用的非一致性内存访问库|

下面给出第三方依赖的安装方法。

### 安装Python

[Python](https://www.python.org/)可通过Conda进行安装。

安装Miniconda：

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-$(arch).sh
bash Miniconda3-py37_4.10.3-Linux-$(arch).sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

安装完成后，可以为Conda设置清华源加速下载，参考[此处](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)。

创建虚拟环境，以Python 3.9.11为例：

```bash
conda create -n mindspore_py39 python=3.9.11 -y
conda activate mindspore_py39
```

可以通过以下命令查看Python版本。

```bash
python --version
```

### 安装昇腾AI处理器配套软件包

昇腾软件包提供商用版和社区版两种下载途径：

- 商用版下载需要申请权限，下载链接即将发布。

- 社区版下载不受限制，下载链接请前往[CANN社区版](https://www.hiascend.com/developer/download/community/result?module=cann)，推荐优先选择`8.1.RC1.beta1`版本，以及在[固件与驱动](https://www.hiascend.com/hardware/firmware-drivers/community)链接中获取对应的固件和驱动安装包，安装包的选择与安装方式请参照上述的商用版安装指引文档。

安装包默认安装路径为`/usr/local/Ascend`。安装后确认当前用户有权限访问昇腾AI处理器配套软件包的安装路径，若无权限，需要root用户将当前用户添加到`/usr/local/Ascend`所在的用户组。

安装昇腾AI处理器配套软件所包含的whl包。如果之前已经安装过昇腾AI处理器配套软件包，需要先使用以下命令卸载对应的whl包。

```bash
pip uninstall te topi hccl -y
```

默认安装路径使用以下命令安装。如果安装路径不是默认路径，需要将命令中的路径替换为安装路径。

```bash
pip install sympy
pip install protobuf
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-*-py3-none-any.whl
```

### 安装wheel setuptools PyYAML和Numpy

在安装完成Python后，使用以下命令安装。

```bash
pip install wheel
pip install -U setuptools
pip install pyyaml
pip install "numpy>=1.19.3,<=1.26.4"
```

注意：运行环境使用的Numpy版本需不小于编译环境的Numpy版本，以保证框架内Numpy相关能力的正常使用。

### 安装GCC

- Ubuntu 18.04可以使用以下命令安装。

    ```bash
    sudo apt-get install gcc-7 -y
    ```

- CentOS 7可以使用以下命令安装。

    ```bash
    sudo yum install centos-release-scl
    sudo yum install devtoolset-7
    ```

    安装完成后，需要使用以下命令切换到GCC 7。

    ```bash
    scl enable devtoolset-7 bash
    ```

- EulerOS和openEuler可以使用以下命令安装。

    ```bash
    sudo yum install gcc -y
    ```

### 安装git tclsh patch NUMA Flex

- Ubuntu 18.04可以使用以下命令安装。

    ```bash
    sudo apt-get install git tcl patch libnuma-dev flex -y
    ```

- CentOS 7，EulerOS和openEuler可以使用以下命令安装。

    ```bash
    sudo yum install git tcl patch numactl-devel flex -y
    ```

### 安装git-lfs

- Ubuntu使用以下命令安装。

    ```bash
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs -y
    git lfs install
    ```

- CentOS 7使用以下命令安装。

    ```bash
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash
    sudo yum install git-lfs -y
    git lfs install
    ```

- EulerOS和openEuler使用以下命令安装。

    根据系统架构选择相应的版本下载。

    ```bash
    # x64
    curl -OL https://github.com/git-lfs/git-lfs/releases/download/v3.1.2/git-lfs-linux-amd64-v3.1.2.tar.gz
    # arm64
    curl -OL https://github.com/git-lfs/git-lfs/releases/download/v3.1.2/git-lfs-linux-arm64-v3.1.2.tar.gz
    ```

    解压并安装。

    ```bash
    mkdir git-lfs
    tar xf git-lfs-linux-*-v3.1.2.tar.gz -C git-lfs
    cd git-lfs
    sudo bash install.sh
    ```

### 安装CMake

- 使用以下命令安装。

    根据系统架构，选择不同的下载链接。

    ```bash
    # x86使用
    curl -O https://cmake.org/files/v3.22/cmake-3.22.2-linux-x86_64.sh
    # aarch64使用
    curl -O https://cmake.org/files/v3.22/cmake-3.22.2-linux-aarch64.sh
    ```

    执行安装脚本安装CMake，默认安装到`/usr/local`目录下。

    ```bash
    sudo mkdir /usr/local/cmake-3.22.2
    sudo bash cmake-3.22.2-linux-*.sh --prefix=/usr/local/cmake-3.22.2 --exclude-subdir
    ```

    最后需要将CMake添加到`PATH`环境变量中。如果使用默认安装目录执行以下命令，其他安装目录需要做相应修改。

    ```bash
    echo -e "export PATH=/usr/local/cmake-3.22.2/bin:\$PATH" >> ~/.bashrc
    source ~/.bashrc
    ```

## 从代码仓下载源码

```bash
git clone -b v2.6.0-rc1 https://gitee.com/mindspore/mindspore.git
```

## 配置环境变量

**如果昇腾AI处理器配套软件包没有安装在默认路径**，源码编译前以及安装好MindSpore之后，需要导出Runtime相关环境变量，以下命令中`LOCAL_ASCEND=/usr/local/Ascend`的`/usr/local/Ascend`表示配套软件包的安装路径，需注意将其改为配套软件包的实际安装路径。

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# environment variables
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# set environmet variables using script provided by CANN, swap "ascend-toolkit" with "nnae" if you are using CANN-nnae package instead
source ${LOCAL_ASCEND}/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_PATH=${LOCAL_ASCEND}/ascend-toolkit/
```

## 编译MindSpore

进入MindSpore根目录，然后执行编译脚本。

```bash
cd mindspore
bash build.sh -e ascend -S on
```

其中：

- `build.sh`中默认的编译线程数为8，如果编译机性能较差，可能会出现编译错误，可在执行中增加`-j{线程数}`来减少线程数量。如`bash build.sh -e ascend -j4`。
- 默认从github下载依赖源码，当-S选项设置为`on`时，从对应的Gitee镜像下载。
- 关于`build.sh`更多用法，请参看脚本头部的说明。

## 安装MindSpore

```bash
pip install output/mindspore-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在联网状态下，安装whl包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/setup.py)中的required_package），其余情况需自行安装依赖。

## 验证是否成功安装

**方法一：**

执行以下命令：

```bash
python -c "import mindspore;mindspore.set_device('Ascend');mindspore.run_check()"
```

如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!
```

说明MindSpore安装成功了。

**方法二：**

执行以下代码：

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_device("Ascend")
x = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))
```

如果输出：

```text
[[[[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]]]
```

说明MindSpore安装成功了。

## 升级MindSpore版本

在源码根目录下执行编译脚本`build.sh`成功后，在`output`目录下找到编译生成的whl安装包，然后执行以下命令进行升级。

从MindSpore 1.x升级到MindSpore 2.x版本时，需要先手动卸载旧版本：

```bash
pip uninstall mindspore-ascend
```

然后安装新版本：

```bash
pip install mindspore-*.whl
```

从MindSpore 2.x版本升级到最新版本时，执行以下命令：

```bash
pip install --upgrade mindspore-*.whl
```
