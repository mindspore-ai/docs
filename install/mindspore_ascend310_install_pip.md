# pip方式安装MindSpore Ascend 310版本

<!-- TOC -->

- [pip方式安装MindSpore Ascend 310版本](#pip方式安装mindspore-ascend-310版本)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装MindSpore](#安装mindspore)
    - [配置环境变量](#配置环境变量)
    - [验证是否成功安装](#验证是否成功安装)
    - [安装MindSpore Serving](#安装mindspore-serving)

<!-- /TOC -->

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.1/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/install/mindspore_ascend310_install_pip.md)

本文档介绍如何在Ascend 310环境的Linux系统上，使用pip方式快速安装MindSpore，Ascend 310版本仅支持推理。

## 确认系统环境信息

- 确认安装Ubuntu 18.04/CentOS 8.2/EulerOS 2.8是64位操作系统。
- 确认安装正确[GCC 版本](http://ftp.gnu.org/gnu/gcc/)，Ubuntu 18.04/EulerOS 2.8用户，GCC>=7.3.0；CentOS 8.2用户 GCC>=8.3.1。
- 确认安装[gmp 6.1.2版本](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz)。
- 确认安装[CMake 3.18.3及以上版本](https://cmake.org/download/)。
    - 安装完成后将CMake所在路径添加到系统环境变量。
- 确认安装Python 3.7.5版本。
    - 如果未安装或者已安装其他版本的Python，可从[官网](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz)或者[华为云](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz)下载Python 3.7.5版本 64位，进行安装。
- 确认安装Ascend 310 AI处理器软件配套包（[Atlas Intelligent Edge Solution V100R020C20](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-intelligent-edge-solution-pid-251167903/software/251687140)）。
    - 确认当前用户有权限访问Ascend 310 AI处理器配套软件包的安装路径`/usr/local/Ascend`，若无权限，需要root用户将当前用户添加到`/usr/local/Ascend`所在的用户组，具体配置请详见配套软件包的说明文档。
    - 需要安装配套GCC 7.3版本的Ascend 310 AI处理器软件配套包。
    - 安装Ascend 310 AI处理器配套软件包提供的whl包，whl包随配套软件包发布，升级配套软件包之后需要重新安装。

        ```bash
        pip install /usr/local/Ascend/ascend-toolkit/latest/atc/lib64/topi-{version}-py3-none-any.whl
        pip install /usr/local/Ascend/ascend-toolkit/latest/atc/lib64/te-{version}-py3-none-any.whl
        ```

## 安装MindSpore

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/ascend/ascend310/{system}/mindspore_ascend-{version}-cp37-cp37m-linux_{arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中：

- 在联网状态下，安装whl包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.1/requirements.txt)），其余情况需自行安装。
- `{version}`表示MindSpore版本号，例如安装1.1.0版本MindSpore时，`{version}`应写为1.1.0。
- `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为`x86_64`。如果系统是ARM架构64位，则写为`aarch64`。
- `{system}`表示系统版本，例如使用的欧拉系统ARM架构，`{system}`应写为`euleros_aarch64`，目前Ascend 310版本可支持以下系统`euleros_aarch64`/`centos_aarch64`/`centos_x86`/`ubuntu_aarch64`/`ubuntu_x86`。

## 配置环境变量

安装好MindSpore之后，需要导出Runtime相关环境变量，下述命令中`LOCAL_ASCEND=/usr/local/Ascend`的`/usr/local/Ascend`表示配套软件包的安装路径，需注意将其改为配套软件包的实际安装路径。

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/acllib/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/atc/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# lib libraries that the mindspore depends on, modify "pip3" according to the actual situation
export LD_LIBRARY_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore/lib"}' | xargs realpath`:${LD_LIBRARY_PATH}

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/atc/ccec_compiler/bin/:${PATH}                       # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                       # Python library that TBE implementation depends on
```

## 验证是否成功安装

创建目录放置样例代码工程，例如`/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_single_op_sample`，代码可以从[官网示例下载](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/sample_resources/ascend310_single_op_sample.zip)获取，这是一个`[1, 2, 3, 4]`与`[2, 3, 4, 5]`相加的简单样例，代码工程目录结构如下:

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

参照`README.md`说明，构建工程，其中`pip3`需要按照实际情况修改。

```bash
cmake . -DMINDSPORE_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`
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

## 安装MindSpore Serving

当您想要快速体验MindSpore在线推理服务时，可以选装MindSpore Serving。

具体安装步骤参见[MindSpore Serving](https://gitee.com/mindspore/serving/blob/r1.1/README_CN.md)。
