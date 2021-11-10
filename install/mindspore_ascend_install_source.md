# 源码编译方式安装MindSpore Ascend 910版本

<!-- TOC -->

- [源码编译方式安装MindSpore Ascend 910版本](#源码编译方式安装mindspore-ascend-910版本)
    - [确认系统环境信息](#确认系统环境信息)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [配置环境变量](#配置环境变量)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/install/mindspore_ascend_install_source.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

本文档介绍如何在Ascend 910环境的Linux系统上，使用源码编译方式快速安装MindSpore。

## 确认系统环境信息

- 确认安装64位操作系统，其中Ubuntu 18.04/CentOS 7.6/EulerOS 2.8/OpenEuler 20.03/KylinV10 SP1是经过验证的。

- 确认安装[GCC 7.3.0版本](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)。

- 确认安装[gmp 6.1.2版本](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz)。

- 确认安装[Flex 2.5.35及以上版本](https://github.com/westes/flex/)。

- 确认安装[OpenMPI 4.0.3版本](https://www.open-mpi.org/faq/?category=building#easy-build)（可选，单机多卡/多机多卡训练需要）。

- 确认安装Python 3.7.5或3.9.0版本。如果未安装或者已安装其他版本的Python，可以选择下载并安装：

    - Python 3.7.5版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz)或[华为云](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz)。
    - Python 3.9.0版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz)或[华为云](https://mirrors.huaweicloud.com/python/3.9.0/Python-3.9.0.tgz)。

- 确认安装[CMake 3.18.3及以上版本](https://cmake.org/download/)。
    - 安装完成后将CMake所在路径添加到系统环境变量。

- 确认安装[patch 2.5及以上版本](http://ftp.gnu.org/gnu/patch/)。
    - 安装完成后将patch所在路径添加到系统环境变量中。

- 确认安装[wheel 0.32.0及以上版本](https://pypi.org/project/wheel/)。

- 确认安装Ascend 910 AI处理器配套软件包（[Ascend Data Center Solution 21.0.3](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-data-center-solution-pid-251167910/software/252504583)）。

    - 软件包安装方式请参考[产品文档](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-data-center-solution-pid-251167910)。
    - 配套软件包包括驱动和固件和CANN。
        - [A800-9000 1.0.12 ARM平台](https://support.huawei.com/enterprise/zh/ascend-computing/a800-9000-pid-250702818/software/253845425) 或 [A800-9010 1.0.12 x86平台](https://support.huawei.com/enterprise/zh/ascend-computing/a800-9010-pid-250702809/software/253845445)
        - [CANN 5.0.3](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/252806307)

    - 确认当前用户有权限访问Ascend 910 AI处理器配套软件包的安装路径`/usr/local/Ascend`，若无权限，需要root用户将当前用户添加到`/usr/local/Ascend`所在的用户组。
    - 安装Ascend 910 AI处理器配套软件包提供的whl包，whl包随配套软件包发布，参考如下命令完成安装。

        ```bash
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-{version}-py3-none-any.whl
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-{version}-py3-none-any.whl
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-{version}-py3-none-any.whl
        ```

    - 如果升级了Ascend 910 AI处理器配套软件包，配套的whl包也需要重新安装，先将原来的安装包卸载，再参考上述命令重新安装。

        ```bash
        pip uninstall te topi hccl -y
        ```

- 确认安装[NUMA 2.0.11及以上版本](https://github.com/numactl/numactl)。
    Ubuntu系统用户，如果未安装，使用如下命令下载安装：

    ```bash
    apt-get install libnuma-dev
    ```

    EulerOS和CentOS系统用户，如果未安装，使用如下命令下载安装：

    ```bash
    yum install numactl-devel
    ```

- 确认安装git工具。
    如果未安装，使用如下命令下载安装：

    ```bash
    apt-get install git # for linux distributions using apt, e.g. ubuntu
    yum install git     # for linux distributions using yum, e.g. centos
    ```

- 确认安装[git-lfs工具](https://github.com/git-lfs/git-lfs/wiki/installation)。

## 从代码仓下载源码

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.5
```

## 编译MindSpore

在源码根目录下执行如下命令。

```bash
bash build.sh -e ascend
```

其中：  
`build.sh`中默认的编译线程数为8，如果编译机性能较差可能会出现编译错误，可在执行中增加-j{线程数}来减少线程数量。如`bash build.sh -e ascend -j4`。

## 安装MindSpore

```bash
pip install output/mindspore_ascend-{version}-{python_version}-linux_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中：

- 在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/r1.5/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/r1.5/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.5/requirements.txt)。
- `{version}`表示MindSpore版本号，例如安装1.5.0-rc1版本MindSpore时，`{version}`应写为1.5.0rc1。
- `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为`x86_64`。如果系统是ARM架构64位，则写为`aarch64`。
- `{python_version}`表示用户的Python版本，Python版本为3.7.5时，`{python_version}`应写为`cp37-cp37m`。Python版本为3.9.0时，则写为`cp39-cp39`。

## 配置环境变量

**如果Ascend 910 AI处理器配套软件包没有安装在默认路径**，安装好MindSpore之后，需要导出Runtime相关环境变量，下述命令中`LOCAL_ASCEND=/usr/local/Ascend`的`/usr/local/Ascend`表示配套软件包的安装路径，需注意将其改为配套软件包的实际安装路径。

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}                 # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                # Python library that TBE implementation depends on
```

## 验证是否成功安装

方法一：

```bash
python -c "import mindspore;mindspore.run_check()"
```

如果输出：

```text
mindspore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

说明MindSpore安装成功了。

方法二：

```python
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="Ascend")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))
```

如果输出：

```text
[[[ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.]],

    [[ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.]],

    [[ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.]]]
```

说明MindSpore安装成功了。

## 升级MindSpore版本

当需要升级MindSpore版本时，可执行如下命令：

- 直接在线升级

    ```bash
    pip install --upgrade mindspore-ascend
    ```

- 本地源码编译升级

    在源码根目录下执行编译脚本`build.sh`成功后，在`output`目录下找到编译生成的whl安装包，然后执行命令进行升级。

    ```bash
    pip install --upgrade mindspore_ascend-{version}-{python_version}-linux_{arch}.whl
    ```
