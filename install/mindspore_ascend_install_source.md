# 源码编译方式安装MindSpore Ascend版本

<!-- TOC -->

- [源码编译方式安装MindSpore Ascend版本](#源码编译方式安装mindspore-ascend版本)
    - [确认系统环境信息](#确认系统环境信息)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [配置环境变量](#配置环境变量)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)
    - [安装MindInsight](#安装mindinsight)
    - [安装MindArmour](#安装mindarmour)
    - [安装MindSpore Hub](#安装mindspore-hub)

<!-- /TOC -->

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/install/mindspore_ascend_install_source.md)

本文档介绍如何在Ascend 910环境的Linux系统上，使用源码编译方式快速安装MindSpore。

## 确认系统环境信息

- 确认安装Ubuntu 18.04/CentOS 7.6/EulerOS 2.8是64位操作系统。
- 确认安装[GCC 7.3.0版本](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)。
- 确认安装[gmp 6.1.2版本](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz)。
- 确认安装[Python 3.7.5版本](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz)。
- 确认安装[OpenSSL 1.1.1及以上版本](https://github.com/openssl/openssl.git)。
    - 安装完成后设置环境变量`export OPENSSL_ROOT_DIR=“OpenSSL安装目录”`。
- 确认安装[CMake 3.18.3及以上版本](https://cmake.org/download/)。
    - 安装完成后将CMake所在路径添加到系统环境变量。
- 确认安装[patch 2.5及以上版本](http://ftp.gnu.org/gnu/patch/)。
    - 安装完成后将patch所在路径添加到系统环境变量中。
- 确认安装[wheel 0.32.0及以上版本](https://pypi.org/project/wheel/)。
- 确认安装Ascend 910 AI处理器软件配套包（Atlas Data Center Solution V100R020C10：[A800-9000 1.0.8 (aarch64)](https://support.huawei.com/enterprise/zh/ascend-computing/a800-9000-pid-250702818/software/252069004?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C250702818)，[A800-9010 1.0.8 (x86_64)](https://support.huawei.com/enterprise/zh/ascend-computing/a800-9010-pid-250702809/software/252062130?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C250702809)，[CANN V100R020C10](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/251174283?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373)）。
    - 确认当前用户有权限访问Ascend 910 AI处理器配套软件包的安装路径`/usr/local/Ascend`，若无权限，需要root用户将当前用户添加到`/usr/local/Ascend`所在的用户组，具体配置请详见配套软件包的说明文档。
    - 安装Ascend 910 AI处理器配套软件包提供的whl包，whl包随配套软件包发布，升级配套软件包之后需要重新安装。

        ```bash
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-{version}-py3-none-any.whl
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-{version}-py3-none-any.whl
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-{version}-py3-none-any.whl
        ```

- 确认安装git工具。

    Ubuntu系统用户，如果未安装，使用如下命令下载安装：

    ```bash
    apt-get install git
    ```

    EulerOS和CentOS系统用户，如果未安装，使用如下命令下载安装：

    ```bash
    yum install git
    ```

## 从代码仓下载源码

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.0
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
chmod +x build/package/mindspore_ascend-{version}-cp37-cp37m-linux_{arch}.whl
pip install build/package/mindspore_ascend-{version}-cp37-cp37m-linux_{arch}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中：

- 在联网状态下，安装whl包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.0/requirements.txt)），其余情况需自行安装。
- `{version}`表示MindSpore版本号，例如下载1.0.1版本MindSpore时，`{version}`应写为1.0.1。
- `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为`x86_64`。如果系统是ARM架构64位，则写为`aarch64`。

## 配置环境变量

**如果Ascend 910 AI处理器配套软件包没有安装在默认路径**，安装好MindSpore之后，需要导出Runtime相关环境变量，下述命令中`LOCAL_ASCEND=/usr/local/Ascend`的`/usr/local/Ascend`表示配套软件包的安装路径，需注意将其改为配套软件包的实际安装路径。

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/add-ons/:${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}                 # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                # Python library that TBE implementation depends on
```

## 验证是否成功安装

```python
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="Ascend")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.tensor_add(x, y))
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

    在源码根目录下执行编译脚本`build.sh`成功后，在`build/package`目录下找到编译生成的whl安装包，然后执行命令进行升级。

    ```bash
    pip install --upgrade mindspore_ascend-{version}-cp37-cp37m-linux_{arch}.whl
    ```

## 安装MindInsight

当您需要查看训练过程中的标量、图像、计算图以及模型超参等信息时，可以选装MindInsight。

具体安装步骤参见[MindInsight](https://gitee.com/mindspore/mindinsight/blob/master/README_CN.md)。

## 安装MindArmour

当您进行AI模型安全研究或想要增强AI应用模型的防护能力时，可以选装MindArmour。

具体安装步骤参见[MindArmour](https://gitee.com/mindspore/mindarmour/blob/master/README_CN.md)。

## 安装MindSpore Hub

当您想要快速体验MindSpore预训练模型时，可以选装MindSpore Hub。

具体安装步骤参见[MindSpore Hub](https://gitee.com/mindspore/hub/blob/master/README_CN.md)。
