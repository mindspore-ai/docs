# 横向联邦端侧部署

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_zh_cn/deploy_federated_client.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

本文档介绍如何编译，部署Federated-Client。

## Linux 编译指导

### 系统环境和第三方依赖

本章节介绍如何完成MindSpore联邦学习的端侧编译，当前联邦学习端侧仅提供Linux上的编译指导，其他系统暂不支持。下表列出了编译所需的系统环境和第三方依赖。

| 软件名称                  | 版本  |  作用 |
|-----------------------| ------------ | ------------ |
| Ubuntu                | 18.04.02LTS   | 编译和运行MindSpore的操作系统  |
| [GCC](#安装gcc)         | 7.3.0到9.4.0之间  | 用于编译MindSpore的C++编译器 |
| [git](#安装git)         | -  | MindSpore使用的源代码管理工具 |
| [CMake](#安装cmake)     | 3.18.3及以上  | 编译构建MindSpore的工具 |
| [Gradle](#安装gradle)   | 6.6.1  | 基于JVM的构建工具  |
| [Maven](#安装maven)     | 3.3.1及以上  | Java项目的管理和构建工具  |
| [OpenJDK](#安装openjdk) | 1.8 到 1.15之间  | Java项目的管理和构建工具  |

#### 安装GCC

可以通过以下命令安装GCC。

```bash
sudo apt-get install gcc-7 git -y
```

如果要安装更高版本的GCC，使用以下命令安装GCC 8。

```bash
sudo apt-get install gcc-8 -y
```

或者安装GCC 9。

```bash
sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-9 -y
```

#### 安装git

可以通过以下命令安装git。

```bash
sudo apt-get install git -y
```

#### 安装CMake

可以通过以下命令安装[CMake](https://cmake.org/)。

```bash
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt-get install cmake -y
```

#### 安装Gradle

可以通过以下命令安装[Gradle](https://gradle.org/releases/)。

```bash
# 下载对应的压缩包，解压。
# 配置环境变量：
  export GRADLE_HOME=GRADLE路径
  export GRADLE_USER_HOME=GRADLE路径
# 将bin目录添加到PATH中：
  export PATH=${GRADLE_HOME}/bin:$PATH
```

#### 安装Maven

可以通过以下命令安装[Maven](https://archive.apache.org/dist/maven/maven-3/)。

```bash
# 下载对应的压缩包，解压。
# 配置环境变量：
  export MAVEN_HOME=MAVEN路径
# 将bin目录添加到PATH中：
  export PATH=${MAVEN_HOME}/bin:$PATH
```

#### 安装OpenJDK

可以通过以下命令安装[OpenJDK](https://jdk.java.net/archive/)。

```bash
# 下载对应的压缩包，解压。
# 配置环境变量：
  export JAVA_HOME=JDK路径
# 将bin目录添加到PATH中：
  export PATH=${JAVA_HOME}/bin:$PATH
```

### 验证是否成功安装

确认[系统环境和第三方依赖](#系统环境和第三方依赖)中安装是否成功。

```text
打开命令窗口数输入：gcc --version
输出以下结果标识安装成功：
  gcc version 版本号

打开命令窗口数输入：git --version
输出以下结果标识安装成功：
  git version 版本号

打开命令窗口数输入：cmake --version
输出以下结果标识安装成功：
  cmake version 版本号

打开命令窗口数输入：gradle --version
输出以下结果标识安装成功：
  Gradle 版本号

打开命令窗口数输入：mvn --version
输出以下结果标识安装成功：
  Apache Maven 版本号

打开命令窗口数输入：java --version
输出以下结果标识安装成功：
  openjdk version 版本号

```

### 编译选项

联邦学习device_client目录下的`cli_build.sh`脚本用于联邦学习端侧的编译。

#### cli_build.sh的参数使用说明

| 参数 | 参数说明                 | 取值范围 | 默认值       |
| ---- | ------------------------ | -------- | ------------ |
| -p   | 依赖外部包的下载存放路径 | 字符串   | third |
| -c   | 是否复用之前下载的依赖包 | on、off  | on           |

### 编译示例

1. 首先，在进行编译之前，需从gitee代码仓下载源码。

    ```bash
    git clone https://gitee.com/mindspore/federated.git ./
    ```

2. 然后进入目录mindspore_federated/device_client，执行如下命令：

    ```bash
    bash cli_build.sh
    ```

3. 由于端侧框架和模型是解耦的，我们提供的x86架构包mindspore-lite-{version}-linux-x64.tar.gz不包含模型相关脚本，因此需要用户自行生成模型脚本对应的jar包，我们提供的模型脚本对应jar包可采用如下方式获取：

    ```bash
    cd federated/example/quick_start_flclient
    bash build.sh -r mindspore-lite-java-flclient.jar #-r 后需要给出最新x86架构包绝对路径(步骤2生成，federated/mindspore_federated/device_client/build/libs/jarX86/mindspore-lite-java-flclient.jar)
    ```

运行以上指令后生成jar包路径为：federated/example/quick_start_flclient/target/quick_start_flclient.jar。

### 构建依赖环境

1. 将文件`federated/mindspore_federated/device_client/third/mindspore-lite-{version}-linux-x64.tar.gz`解压后，所得到的目录结构如下所示：

    ```sh
    mindspore-lite-{version}-linux-x64
    ├── tools
    │   ├── benchmark_train # 训练模型性能与精度调测工具
    │   ├── converter       # 模型转换工具
    │   └── cropper         # 库裁剪工具
    │       ├── cropper                 # 库裁剪工具可执行文件
    │       └── cropper_mapping_cpu.cfg # 裁剪cpu库所需的配置文件
    └── runtime
        ├── include  # 训练框架头文件
        │   └── registry # 自定义算子注册头文件
        ├── lib      # 训练框架库
        │   ├── libminddata-lite.a          # 图像处理静态库文件
        │   ├── libminddata-lite.so        # 图像处理动态库文件
        │   ├── libmindspore-lite-jni.so   # MindSpore Lite推理框架依赖的jni动态库
        │   ├── libmindspore-lite-train.a  # MindSpore Lite训练框架依赖的静态库
        │   ├── libmindspore-lite-train.so # MindSpore Lite训练框架依赖的动态库
        │   ├── libmindspore-lite-train-jni.so # MindSpore Lite训练框架依赖的jni动态库
        │   ├── libmindspore-lite.a  # MindSpore Lite推理框架依赖的静态库
        │   ├── libmindspore-lite.so  # MindSpore Lite推理框架依赖的动态库
        │   ├── mindspore-lite-java.jar    # MindSpore Lite训练框架jar包
        │   └── mindspore-lite-java-flclient.jar  # 联邦学习框架jar包
        └── third_party
            └── libjpeg-turbo
                └── lib
                    ├── libjpeg.so.62   # 图像处理动态库文件
                    └── libturbojpeg.so.0  # 图像处理动态库文件
    ```

2. 其中联邦学习所需的相关x86包名如下：

    ```sh
    libminddata-lite.so  # 图像处理动态库文件
    libmindspore-lite.so  # MindSpore Lite推理框架依赖的动态库
    libmindspore-lite-jni.so  # MindSpore Lite推理框架依赖的jni动态库
    libmindspore-lite-train.so  # MindSpore Lite训练框架依赖的动态库
    libmindspore-lite-train-jni.so # MindSpore Lite训练框架的jni动态库
    libjpeg.so.62   # 图像处理动态库文件
    libturbojpeg.so.0  # 图像处理动态库文件
    ```

3. 可将路径`mindspore-lite-{version}-linux-x64/runtime/lib/`以及`mindspore-lite-{version}-linux-x64/runtime/third_party/libjpeg-turbo/lib`中联邦学习所依赖的so文件（共7个）放入一个文件夹，比如`/resource/x86libs/`。然后在x86中设置环境变量(下面需提供绝对路径)：

    ```sh
    export LD_LIBRARY_PATH=/resource/x86libs/:$LD_LIBRARY_PATH
    ```

4. 设置好依赖环境之后，可参考应用实践教程[实现一个端云联邦的图像分类应用(x86)](https://www.mindspore.cn/federated/docs/zh-CN/master/image_classification_application.html)在x86环境中模拟启动多个客户端进行联邦学习。