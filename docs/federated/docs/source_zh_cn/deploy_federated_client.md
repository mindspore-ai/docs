# 端侧部署

<!-- TOC -->

- [端侧部署](#端侧部署)
    - [Android环境](#android环境)
        - [编译出包](#编译出包)
        - [运行依赖](#运行依赖)
        - [构建依赖环境](#构建依赖环境)
    - [x86环境](#x86环境)
        - [编译出包](#编译出包-1)
        - [运行依赖](#运行依赖-1)
        - [构建依赖环境](#构建依赖环境-1)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/federated/docs/source_zh_cn/deploy_federated_client.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

下面分别介绍如何在Android环境和x86环境部署Federated-Client。

## Android环境

### 编译出包

1. 配置编译环境。

    目前只支持Linux环境编译，Linux编译环境配置可参考[这里](https://www.mindspore.cn/lite/docs/zh-CN/r1.5/use/build.html#linux)。

2. 开启联邦编译选项，在mindspore根目录进行编译，编译包含aarch64和aarch32的AAR包。

    ```sh
    export MSLITE_ENABLE_FL=on
    bash build.sh -A on -j32
    ```

3. 获取生成的Android AAR包。

    ```text
    mindspore-lite-maven-{version}.zip
    ```

### 运行依赖

- [Android Studio](https://developer.android.google.cn/studio) >= 4.0
- [Android SDK](https://developer.android.com/studio?hl=zh-cn#cmdline-tools) >= 29

### 构建依赖环境

将文件`mindspore-lite-maven-{version}.zip`解压后，所得到的目录结构如下所示：

```text
mindspore-lite-maven-{version}
└── mindspore
    └── mindspore-lite
        └── {version}
            └── mindspore-lite-{version}.aar  # MindSpore Lite训练框架AAR包
```

由此可知联邦学习相关的AAR包路径是：

```text
mindspore/output/mindspore/mindspore-lite/{version}/mindspore-lite-{version}.aar
```

其中AAR包中与联邦学习相关的目录结构如下：

```text
mindspore-lite-{version}
├── jni
│   ├── arm64-v8a
│   │   ├── libjpeg.so   # 图像处理动态库文件
│   │   ├── libminddata-lite.so  # 图像处理动态库文件
│   │   ├── libmindspore-lite.so  # MindSpore Lite推理框架依赖的动态库
│   │   ├── libmindspore-lite-jni.so  # MindSpore Lite推理框架依赖的jni动态库
│   │   ├── libmindspore-lite-train.so  # MindSpore Lite训练框架依赖的动态库
│   │   ├── libmindspore-lite-train-jni.so  # MindSpore Lite训练框架依赖的jni动态库
│   │   └── libturbojpeg.so  # 图像处理动态库文件
│   └── armeabi-v7a
│       ├── libjpeg.so   # 图像处理动态库文件
│       ├── libminddata-lite.so  # 图像处理动态库文件
│       ├── libmindspore-lite.so  # MindSpore Lite推理框架依赖的动态库
│       ├── libmindspore-lite-jni.so  # MindSpore Lite推理框架依赖的jni动态库
│       ├── libmindspore-lite-train.so  # MindSpore Lite训练框架依赖的动态库
│       ├── libmindspore-lite-train-jni.so  # MindSpore Lite训练框架依赖的jni动态库
│       └── libturbojpeg.so  # 图像处理动态库文件
├── libs
│   ├── mindspore-lite-java-common.jar  # MindSpore Lite训练框架jar包
│   └── mindspore-lite-java-flclient.jar  # 联邦学习框架jar包
└── classes.jar  # MindSpore Lite训练框架jar包
```

注意1，由于生成Android环境中的联邦学习jar包时未包含所依赖的第三方开源软件包，因此在Android环境中，使用AAR包前，需要用户在Android工程下的app/build.gradle文件的dependencies{}字段中添加相关依赖语句，用于加载联邦学习所依赖的三个开源软件，如下所示：

```java
dependencies {

//添加联邦学习所依赖第三方开源软件
implementation group: 'com.squareup.okhttp3', name: 'okhttp', version: '3.14.9'
implementation group: 'com.google.flatbuffers', name: 'flatbuffers-java', version: '1.11.0'
implementation(group: 'org.bouncycastle',name: 'bcprov-jdk15on', version: '1.68')
}
```

具体实现可参考文档[情感分类应用](https://www.mindspore.cn/federated/docs/zh-CN/r1.5/sentiment_classification_application.html)中 `Android工程配置依赖项`部分提供的`app/build.gradle` 文件示例。

注意2，由于联邦学习依赖的第三方开源软件`bcprov-jdk15on`包含多版本class文件，为防止低版本jdk编译高版本class文件出错，在Android工程的`gradle.properties`文件中可添加如下设置语句：

```java
android.jetifier.blacklist=bcprov
```

在Android工程中设置好了如上所示依赖之后，只需依赖 AAR包即可调用联邦学习提供的相关接口，接口的具体调用和运行方式可参考[联邦学习接口介绍部分](https://www.mindspore.cn/federated/api/zh-CN/r1.5/index.html)。

## x86环境

### 编译出包

1. 配置编译环境。

    目前只支持Linux环境编译，Linux编译环境配置可参考[这里](https://www.mindspore.cn/lite/docs/zh-CN/r1.5/use/build.html#linux)。

2. 在mindspore根目录进行编译，编译x86架构相关包。

    ```sh
    bash build.sh -I x86_64 -j32
    ```

3. 获取生成的x86架构相关包。

    ```text
    mindspore/output/mindspore-lite-{version}-linux-x64.tar.gz
    ```

### 运行依赖

- [Python](https://www.python.org/downloads/)>=3.7.5
- [OpenJDK](https://openjdk.java.net/install/) >= 1.9

### 构建依赖环境

将文件`mindspore/output/mindspore-lite-{version}-linux-x64.tar.gz`解压后，所得到的目录结构如下所示：

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
    │   ├── libmindspore-lite.so  # MindSpore Lite推理依赖的动态库
    │   ├── mindspore-lite-java.jar    # MindSpore Lite训练框架jar包
    │   └── mindspore-lite-java-flclient.jar  # 联邦学习框架jar包
    └── third_party
        └── libjpeg-turbo
            └── lib
                ├── libjpeg.so.62   # 图像处理动态库文件
                └── libturbojpeg.so.0  # 图像处理动态库文件
```

其中联邦学习所需的相关x86包名如下：

```sh
libjpeg.so.62   # 图像处理动态库文件
libminddata-lite.so  # 图像处理动态库文件
libmindspore-lite.so  # MindSpore Lite推理框架依赖的动态库
libmindspore-lite-jni.so  # MindSpore Lite推理框架依赖的jni动态库
libmindspore-lite-train.so  # MindSpore Lite训练框架依赖的动态库
libmindspore-lite-train-jni.so # MindSpore Lite训练框架的jni动态库
libturbojpeg.so.0  # 图像处理动态库文件
mindspore-lite-java-flclient.jar  # 联邦学习框架jar包
```

可将路径`mindspore/output/mindspore-lite-{version}-linux-x64/runtime/lib/`以及`mindspore/output/mindspore-lite-{version}-linux-x64/runtime/third_party/libjpeg-turbo/lib`中联邦学习所依赖的so文件（共7个）放入一个文件夹，比如`/resource/x86libs/`。然后在x86中设置环境变量(下面需给绝对路径)：

```sh
export LD_LIBRARY_PATH=/resource/x86libs/:$LD_LIBRARY_PATH
```

设置好依赖环境之后，可参考[这里](https://www.mindspore.cn/federated/docs/zh-CN/r1.5/image_classification_application.html)教程在x86环境中模拟启动多个客户端进行联邦学习。