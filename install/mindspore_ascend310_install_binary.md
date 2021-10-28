# 二进制包方式安装MindSpore Ascend 310版本

<!-- TOC -->

- [二进制包方式安装MindSpore Ascend 310版本](#二进制包方式安装mindspore-ascend-310版本)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装MindSpore](#安装mindspore)
    - [配置环境变量](#配置环境变量)
    - [验证是否成功安装](#验证是否成功安装)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/install/mindspore_ascend310_install_binary.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

本文档介绍如何在Ascend 310环境的Linux系统上，使用二进制包方式快速安装MindSpore，Ascend 310版本仅支持推理。

## 确认系统环境信息

- 确认安装64位操作系统，[glibc](https://www.gnu.org/software/libc/)>=2.17，其中Ubuntu 18.04/CentOS 7.6/EulerOS 2.8是经过验证的。

- 确认安装[GCC 7.3.0版本](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)。

- 确认安装[gmp 6.1.2版本](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz)。

- 确认安装[CMake 3.18.3及以上版本](https://cmake.org/download/)。

    - 安装完成后将CMake所在路径添加到系统环境变量。

- 确认安装Python 3.7.5或3.9.0版本。如果未安装或者已安装其他版本的Python，可以选择下载并安装：

    - Python 3.7.5版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz)或[华为云](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz)。
    - Python 3.9.0版本 64位，下载地址：[官网](https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz)或[华为云](https://mirrors.huaweicloud.com/python/3.9.0/Python-3.9.0.tgz)。

- 确认安装Ascend 310 AI处理器配套软件包（[Ascend Data Center Solution]）。
    - 软件包安装方式请参考[产品文档](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-data-center-solution-pid-251167910)。
    - 配套软件包包括驱动和固件和CANN。
        - [驱动和固件A300-3000 1.0.12 ARM平台]和[驱动和固件A300-3010 1.0.12 x86平台]
        - [CANN 5.0.3](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/252806307)
    - 确认当前用户有权限访问Ascend 310 AI处理器配套软件包的安装路径`/usr/local/Ascend`，若无权限，需要root用户将当前用户添加到`/usr/local/Ascend`所在的用户组。
    - 安装Ascend 310 AI处理器配套软件包提供的whl包，whl包随配套软件包发布，升级配套软件包之后需要重新安装。

    ```bash
    pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-{version}-py3-none-any.whl
    pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-{version}-py3-none-any.whl
    ```

## 安装MindSpore

参考[版本列表](https://www.mindspore.cn/versions)先进行SHA-256完整性校验，校验一致后再执行如下命令安装MindSpore。

```bash
wget https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/ascend/{arch}/mindspore_ascend-{version}-linux_{arch}.tar.gz --no-check-certificate
tar -zxf mindspore_ascend-{version}-linux_{arch}.tar.gz
```

其中：

- `{version}`表示MindSpore版本号，例如安装1.5.0版本MindSpore时，`{version}`应写为1.5.0，而安装1.5.0-rc1版本时，第一个`{version}`代表下载路径应写为1.5.0-rc1，第二个`{version}`代表版本号应写为1.5.0rc1。
- `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为`x86_64`。如果系统是ARM架构64位，则写为`aarch64`。

## 配置环境变量

安装好MindSpore之后，需要导出Runtime相关环境变量，下述命令中`LOCAL_ASCEND=/usr/local/Ascend`的`/usr/local/Ascend`表示配套软件包的安装路径，需注意将其改为配套软件包的实际安装路径。

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
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                       # Python library that TBE implementation depends on

# Set path to extracted MindSpore accordingly
export LD_LIBRARY_PATH={mindspore_path}:${LD_LIBRARY_PATH}
```

其中：

- `{mindspore_path}`表示MindSpore二进制包所在位置的绝对路径。

## 验证是否成功安装

创建目录放置样例代码工程，例如`/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_single_op_sample`，代码可以从[官网示例下载](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/sample_resources/ascend310_single_op_sample.zip)获取，这是一个`[1, 2, 3, 4]`与`[2, 3, 4, 5]`相加的简单样例，代码工程目录结构如下:

```text
└─ascend310_single_op_sample
    ├── CMakeLists.txt                    // 编译脚本
    ├── README.md                         // 使用说明
    ├── main.cc                           // 主函数
    └── tensor_add.mindir                 // MindIR模型文件
```

进入样例工程目录，按照实际情况修改路径路径：

```bash
cd /home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_single_op_sample
```

参照`README.md`说明，构建工程，其中`{mindspore_path}`表示MindSpore二进制包所在位置的绝对路径，根据实际情况替换。

```bash
cmake . -DMINDSPORE_PATH={mindspore_path}
make
```

构建成功后，执行用例。

```bash
./tensor_add_sample
```

如果输出：

```text
3
5
7
9
```

说明MindSpore安装成功了。

