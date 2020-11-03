# 源码编译方式安装MindSpore

<!-- TOC -->

- [源码编译方式安装MindSpore](#源码编译方式安装mindspore)
    - [确认系统环境信息](#确认系统环境信息)
    - [从代码仓下载源码](#从代码仓下载源码)
    - [编译MindSpore](#编译mindspore)
    - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_gpu_install_source.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

本文档介绍如何在GPU环境的Linux系统上，使用源码编译方式快速安装MindSpore。

## 确认系统环境信息

- 确认安装Ubuntu18.04是64位操作系统。
- [Python](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz)，选择3.7.5版本。
- [wheel](https://pypi.org/project/wheel/)，选择0.32.0版本及以上。
- [GCC](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)，选择7.3.0（7.X版本可行）。
- [CMake](https://cmake.org/download/)，选择3.14.1版本及以上，安装完成后将CMake添加到系统环境变量。
- [Autoconf](https://www.gnu.org/software/autoconf)，选择2.69版本以上（可使用系统自带版本）。
- [Libtool](https://www.gnu.org/software/libtool)，选择2.4.6-29.fc30版本以上（可使用系统自带版本）。
- [Automake](https://www.gnu.org/software/automake)，选择1.15.1版本以上（可使用系统自带版本）。
- [CuDNN](https://developer.nvidia.com/rdp/cudnn-archive)，选择7.6版本以上。
- [gmp](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz)，选择6.1.2版本以上。
- [Flex](https://github.com/westes/flex/)，选择2.5.35版本以上。
- [wheel](https://pypi.org/project/wheel/)，选择0.32.0版本以上。
- [patch](http://ftp.gnu.org/gnu/patch/)，选择2.5版本以上，安装完成后将patch添加到系统环境变量中。
- [CUDA10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base)按默认配置安装。  
    CUDA安装后，若CUDA没有安装在默认位置，需要设置环境变量PATH（如：`export PATH=/usr/local/cuda-${version}/bin:$PATH`）和`LD_LIBRARY_PATH`（如：`export LD_LIBRARY_PATH=/usr/local/cuda-${version}/lib64:$LD_LIBRARY_PATH`），详细安装后的设置可参考[CUDA安装手册](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)。
- `git`工具。  
    如果未安装，使用如下命令下载安装：

    ```bash
    apt-get install git
    ```

## 从代码仓下载源码

```bash
git clone https://gitee.com/mindspore/mindspore.git
```

## 编译MindSpore

在源码根目录下执行如下命令。

```bash
bash build.sh -e gpu
```

> `build.sh`中默认的编译线程数为8，如果编译机性能较差可能会出现编译错误，可在执行中增加-j{线程数}来减少线程数量。如`bash build.sh -e gpu -j4`。

## 安装MindSpore

```bash
chmod +x build/package/mindspore_gpu-{version}-cp37-cp37m-linux_{arch}.whl
pip install build/package/mindspore_gpu-{version}-cp37-cp37m-linux_{arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://mirrors.huaweicloud.com/repository/pypi/simple
```

> `{version}`表示MindSpore版本号，例如下载1.0.1版本MindSpore时，`{version}`应写为1.0.1。  
> `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为`x86_64`。如果系统是ARMv8架构64位，则写为`aarch64`。

## 验证是否成功安装

```python
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="GPU")
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
