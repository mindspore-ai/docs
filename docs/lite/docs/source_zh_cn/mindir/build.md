# 编译云侧MindSpore Lite

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/mindir/build.md)

本章节介绍如何快速编译出云侧MindSpore Lite。

云侧MindSpore Lite包含模块：

| 模块               | 支持平台 | 说明         |
| ------------------ | -------- | ------------ |
| converter          | Linux    | 模型转换工具 |
| runtime(cpp、java) | Linux    | 模型推理框架 |
| benchmark          | Linux    | 基准测试工具 |
| minddata           | Linux    | 图像处理库   |
| akg                | Linux                   | 基于Polyhedral的算子编译器（[Auto Kernel Generator](https://gitee.com/mindspore/akg)） |

## 环境要求

- 系统环境：Linux x86_64或arm64，推荐使用Ubuntu 18.04.02LTS。
- C++编译依赖
    - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
    - [CMake](https://cmake.org/download/) >= 3.18.3
    - [Git](https://git-scm.com/downloads) >= 2.28.0
- Java API模块的编译依赖（可选），未设置JAVA_HOME环境变量则不编译该模块。
    - [Gradle](https://gradle.org/releases/) >= 6.6.1
        - 配置环境变量：`export GRADLE_HOME=GRADLE路径`和`export GRADLE_USER_HOME=GRADLE路径`
        - 将bin目录添加到PATH中：`export PATH=${GRADLE_HOME}/bin:$PATH`
    - [Maven](https://archive.apache.org/dist/maven/maven-3/) >= 3.3.1
        - 配置环境变量：`export MAVEN_HOME=MAVEN路径`
        - 将bin目录添加到PATH中：`export PATH=${MAVEN_HOME}/bin:$PATH`
    - [OpenJDK](https://openjdk.java.net/install/) 1.8 到 1.15
        - 配置环境变量：`export JAVA_HOME=JDK路径`
        - 将bin目录添加到PATH中：`export PATH=${JAVA_HOME}/bin:$PATH`
- Python API模块的编译依赖（可选），未安装Python3或者NumPy则不编译该模块。
    - [Python](https://www.python.org/) >= 3.7.0
    - [NumPy](https://numpy.org/) >= 1.17.0 (如果用pip安装失败，请先升级pip版本：`python -m pip install -U pip`)
    - [wheel](https://pypi.org/project/wheel/) >= 0.32.0 (如果用pip安装失败，请先升级pip版本：`python -m pip install -U pip`)
- AKG（可选，默认编译），未安装LLVM-12或者Python3则不编译akg，未安装git-lfs则无法编译ascend后端的akg。
  - [llvm](#安装llvm-可选) == 12.0.1
  - [git-lfs](https://git-lfs.com/)

> Gradle建议采用[gradle-6.6.1-complete](https://gradle.org/next-steps/?version=6.6.1&format=all)版本，配置其他版本gradle将会采用gradle wrapper机制自动下载`gradle-6.6.1-complete`。
>
> 也可直接使用已配置好上述依赖的Docker编译镜像。
>
> - 下载镜像：`docker pull swr.cn-south-1.myhuaweicloud.com/mindspore-build/mindspore-lite:ubuntu18.04.2-20210530`
> - 创建容器：`docker run -tid --net=host --name=docker01 swr.cn-south-1.myhuaweicloud.com/mindspore-build/mindspore-lite:ubuntu18.04.2-20210530`
> - 进入容器：`docker exec -ti -u 0 docker01 bash`

## 编译选项

MindSpore根目录下的`build.sh`脚本可用于云侧MindSpore Lite的编译。

### `build.sh`的参数使用说明

| 参数  | 参数说明                                         | 取值范围      | 默认值 |
| ----- | ------------------------------------------------ | ------------- | ------ |
| -I    | 选择目标架构                                     | arm64、x86_64 | 无     |
| -d    | 设置该参数，则编译Debug版本，否则编译Release版本 | 无            | 无     |
| -i    | 设置该参数，则进行增量编译，否则进行全量编译     | 无            | 无     |
| -j[n] | 设定编译时所用的线程数，否则默认设定为8线程      | Integer       | 8      |
| -K    | 设定编译时是否编译akg，否则默认编译akg      | on、off       | on      |

> - 若配置了JAVA_HOME环境变量并安装了Gradle，则同时编译JAR包。
> - 在`-I`参数变动时，如`-I x86_64`变为`-I arm64`，添加`-i`参数进行增量编译不生效。
> - 不支持交叉编译，即arm64版本需要在arm环境编译。

### 模块构建编译选项

模块的构建通过环境变量进行控制，用户可通过声明相关环境变量，控制编译构建的模块。在修改编译选项后，为使选项生效，在使用`build.sh`脚本进行编译时，不可添加`-i`参数进行增量编译。

通用模块编译选项如下：

| 选项                                 | 参数说明                                   | 取值范围      | 默认值               |
| ------------------------------------ | ------------------------------------------ | ------------- | -------------------- |
| MSLITE_GPU_BACKEND                   | 设置GPU后端，在`-I x86_64`时仅tensorrt有效 | tensorrt、off | 在`-I x86_64`时为off |
| MSLITE_ENABLE_TOOLS                  | 是否编译配套Benchmark基准测试工具          | on、off       | on                   |
| MSLITE_ENABLE_TESTCASES              | 是否编译测试用例                           | on、off       | off                  |
| MSLITE_ENABLE_ACL                    | 是否使能昇腾ACL                            | on、off       | off                  |
| MSLITE_ENABLE_CLOUD_INFERENCE | 是否使能云侧推理                           | on、off       | off                  |
| MSLITE_ENABLE_SSE | 是否启用SSE指令集，仅在`-I x86_64`时有效 | on、off | off |
| MSLITE_ENABLE_AVX512 | 是否启用AVX512指令集，仅在`-I x86_64`时有效 | on、off | off |

> - 云侧推理版本依赖模型转换工具，因此当``MSLITE_ENABLE_CLOUD_INFERENCE``配置为``on``时，会同时编译``converter``。
> - 若环境只支持SSE指令集，AVX512指令集需配置为``off``。

## 编译示例

首先，在进行编译之前，需从MindSpore代码仓下载源码。

```bash
git clone -b v2.6.0rc1 https://gitee.com/mindspore/mindspore.git
```

### 环境准备

#### Ascend

- 确认安装昇腾AI处理器配套软件包。

    - 昇腾软件包提供商用版和社区版两种下载途径：

        1. 商用版下载需要申请权限，下载链接与安装方式请参考[Ascend Data Center Solution 23.0.RC3安装指引文档](https://support.huawei.com/enterprise/zh/doc/EDOC1100336282)。

        2. 社区版下载不受限制，下载链接请前往[CANN社区版](https://www.hiascend.com/developer/download/community/result?module=cann)，选择`7.0.RC1.beta1`版本，以及在[固件与驱动](https://www.hiascend.com/hardware/firmware-drivers?tag=community)链接中获取对应的固件和驱动安装包，安装包的选择与安装方式请参照上述的商用版安装指引文档。
    - 安装包默认安装路径为`/usr/local/Ascend`。安装后确认当前用户有权限访问昇腾AI处理器配套软件包的安装路径，若无权限，需要root用户将当前用户添加到`/usr/local/Ascend`所在的用户组。
    - 安装昇腾AI处理器配套软件所包含的whl包。如果之前已经安装过昇腾AI处理器配套软件包，需要先使用如下命令卸载对应的whl包。

      ```bash
      pip uninstall te topi -y
      ```

      默认安装路径使用以下指令安装。如果安装路径不是默认路径，需要将命令中的路径替换为安装路径。

      ```bash
      pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/topi-{version}-py3-none-any.whl
      pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-{version}-py3-none-any.whl
      ```

- 配置环境变量

    - 安装好Ascend软件包之后，需要导出Runtime相关环境变量，下述命令中`LOCAL_ASCEND=/usr/local/Ascend`的`/usr/local/Ascend`表示配套软件包的安装路径，需注意将其改为配套软件包的实际安装路径。

        ```bash
        # control log level. 0-EBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
        export GLOG_v=2

        # Conda environmental options
        LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

        # lib libraries that the run package depends on
        export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

        # Environment variables that must be configured
        export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
        export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
        export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/compiler/ccec_compiler/bin/:${PATH}                  # TBE operator compilation tool path
        export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                       # Python library that TBE implementation depends on
        ```

#### GPU

GPU环境编译，使用TensorRT需要集成CUDA、TensorRT。当前版本适配[CUDA 11.1](https://developer.nvidia.com/cuda-11.1.1-download-archive) 和 [TensorRT 8.5.1](https://developer.nvidia.com/nvidia-tensorrt-8x-download)。

安装相应版本的CUDA，并将安装后的目录设置为环境变量`${CUDA_HOME}`。构建脚本将使用这个环境变量寻找CUDA。

下载对应版本的TensorRT压缩包，并将压缩包解压后的目录设置为环境变量`${TENSORRT_PATH}`。构建脚本将使用这个环境变量寻找TensorRT。

#### CPU

使用x86_64或ARM64环境。

#### 安装LLVM-可选

模型转换工具中图算融合功能的CPU后端需要依赖LLVM-12，可以通过以下命令安装[LLVM](https://llvm.org/)。如果没有安装LLVM-12则图算融合功能仅能支持GPU和Ascend后端。

```shell
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo add-apt-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-12 main"
sudo apt-get update
sudo apt-get install llvm-12-dev -y
```

### 执行编译

三后端合一包需配置如下环境变量：

```bash
export MSLITE_ENABLE_CLOUD_INFERENCE=on
export MSLITE_GPU_BACKEND=tensorrt
export MSLITE_ENABLE_ACL=on
```

> - 如无需Ascend后端，可配置``export MSLITE_ENABLE_ACL=off``
> - 如无需GPU后端，可配置``export MSLITE_GPU_BACKEND=off``

在源码根目录下执行如下命令，可编译不同版本的MindSpore Lite。

- 编译x86_64架构版本，同时设定线程数。

    ```bash
    bash build.sh -I x86_64 -j32
    ```

- 编译arm64架构版本，同时设定线程数。

    ```bash
    bash build.sh -I arm64 -j32
    ```

- 编译x86_64架构版本，同时设定线程数，但是不编译AKG。

    ```bash
    bash build.sh -I x86_64 -j32 -K off
    ```

最后，会在`output/`目录中生成如下文件：

- `mindspore-lite-{version}-{os}-{arch}.tar.gz`：包含runtime和配套工具。

- `mindspore-lite-{version}-{python}-{os}-{arch}.whl`：包含runtime(Python)的Whl包。

> - version: 输出件版本号，与所编译的分支代码对应的版本一致。
> - python: 输出件Python版本，如：Python3.7为`cp37-cp37m`。
> - os: 输出件应部署的操作系统。
> - arch: 输出件应部署的系统架构。

若要体验Python接口，需要移动到`output/`目录下，使用以下命令进行安装Whl安装包。

```bash
pip install mindspore-lite-{version}-{python}-{os}-{arch}.whl
```

安装后可以使用以下命令检查是否安装成功：若无报错，则表示安装成功。

```bash
python -c "import mindspore_lite"
```

安装后可以使用以下命令检查mindspore_lite内置的AKG是否安装成功：若无报错，则表示安装成功。

```bash
python -c "import mindspore_lite.akg"
```

安装成功后，可使用`pip show mindspore_lite`命令查看MindSpore Lite的Python模块的安装位置。

## 目录结构

```text
mindspore-lite-{version}-linux-{arch}
├── runtime
│   ├── include
│   ├── lib
│   │   ├── libascend_kernel_plugin.so # Ascend Kernel插件动态库
│   │   ├── libdvpp_utils.so           # DVPP图像预处理工具动态库
│   │   ├── libminddata-lite.a         # 图像处理静态库
│   │   ├── libminddata-lite.so        # 图像处理动态库
│   │   ├── libmindspore-core.so       # MindSpore Core动态库
│   │   ├── libmindspore-glog.so.0     # glog动态库
│   │   ├── libmindspore-lite-jni.so   # MindSpore Lite推理框架的jni动态库
│   │   ├── libmindspore-lite.so       # MindSpore Lite推理框架动态库
│   │   ├── libmsplugin-ge-litert.so   # GE LiteRT插件动态库
│   │   └── mindspore-lite-java.jar    # MindSpore Lite推理框架jar包
│   └── third_party
│       ├── glog
│       ├── libjpeg-turbo
│       └── securec
└── tools
    ├── akg
    |    └── akg-{version}-{python}-linux-{arch}.whl # AKG的whl包
    ├── benchmark              # 基准测试工具
    │   └── benchmark          # 基准测试工具可执行文件
    └── converter              # 模型转换工具
        ├── converter
        │   └── converter_lite # 转换工具可执行文件
        ├── include
        └── lib
            ├── libascend_pass_plugin.so
            ├── libmindspore_converter.so
            ├── libmindspore_core.so
            ├── libmindspore_glog.so.0
            ├── libmslite_shared_lib.so
            ├── libmslite_converter_plugin.so
            ├── libopencv_core.so.4.5
            ├── libopencv_imgcodecs.so.4.5
            └── libopencv_imgproc.so.4.5
```
