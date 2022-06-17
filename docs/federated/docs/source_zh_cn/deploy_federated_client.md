# 端侧部署

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/federated/docs/source_zh_cn/deploy_federated_client.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

本文档分别介绍如何面向Android aarch环境和Linux x86_64环境，部署Federated-Client。

## Android aarch环境

### 编译出包

1. 配置编译环境。

    目前只支持Linux环境编译，Linux编译环境配置可参考[这里](https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/build.html#linux环境编译)。

2. 在mindspore根目录进行编译，编译x86架构相关包。

    ```sh
    bash build.sh -I x86_64 -j32
    ```

   编译完成后在`mindspore/output`路径下将生成x86架构包，请备份备用(`mindspore/output`每次编译均会进行清理)

    ```text
    mindspore-lite-{version}-linux-x64.tar.gz
    ```

3. 开启联邦编译选项，在mindspore根目录进行编译，编译包含aarch64和aarch32的AAR包。

    ```sh
    export MSLITE_ENABLE_FL=on
    bash build.sh -A on -j32
    ```

   编译完成后在`mindspore/output`路径下将生成Android AAR包，请备份备用(`mindspore/output`每次编译均会进行清理)

    ```text
    mindspore-lite-full-{version}.zip
    ```

4. 由于端侧框架和模型是解耦的，我们提供的Android AAR包不包含模型相关脚本，因此需要用户自行生成模型脚本对应的jar包，我们提供了两个类型的模型脚本供大家参考（[有监督情感分类任务](https://gitee.com/mindspore/mindspore/tree/r1.7/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert)、[LeNet图片分类任务](https://gitee.com/mindspore/mindspore/tree/r1.7/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet)）。用户可参考这两个类型的模型脚本，自定义模型脚本后，生成对应的jar包（假设命名为`quick_start_flclient.jar`）。我们提供的模型脚本对应jar包可采用如下方式获取：

    下载[MindSpore开源仓](https://gitee.com/mindspore/mindspore)最新代码后，进行以下操作：

    ```sh
    cd mindspore/mindspore/lite/examples/quick_start_flclient
    sh build.sh -r "mindspore-lite-{version}-linux-x64.tar.gz"   # -r 后需给出最新x86版本架构包绝对路径（步骤2生成）
    ```

    运行以上指令后生成jar包路径为：`mindspore/mindspore/lite/examples/quick_start_flclient/target/quick_start_flclient.jar`。

### 运行依赖

- [Android Studio](https://developer.android.google.cn/studio) >= 4.0
- [Android SDK](https://developer.android.com/studio?hl=zh-cn#cmdline-tools) >= 29

### 构建依赖环境

将文件`mindspore-lite-full-{version}.aar`重命名为`mindspore-lite-full-{version}.zip`，解压后，所得到的目录结构如下所示：

```text
mindspore-lite-full-{version}
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

注意1，由于生成Android环境中的联邦学习jar包时未包含所依赖的第三方开源软件包，因此在Android环境中，使用AAR包前，需要用户在Android工程下的app/build.gradle文件中，为dependencies{}字段添加相关依赖语句，用于加载联邦学习所依赖的三个开源软件，如下所示：

```text
dependencies {

//添加联邦学习所依赖第三方开源软件
implementation group: 'com.squareup.okhttp3', name: 'okhttp', version: '3.14.9'
implementation group: 'com.google.flatbuffers', name: 'flatbuffers-java', version: '2.0.0'
implementation(group: 'org.bouncycastle',name: 'bcprov-jdk15on', version: '1.68')
}
```

具体实现可参考文档[情感分类应用](https://www.mindspore.cn/federated/docs/zh-CN/r1.7/sentiment_classification_application.html)中 `Android工程配置依赖项`部分提供的`app/build.gradle` 文件示例。

注意2，由于联邦学习依赖的第三方开源软件`bcprov-jdk15on`包含多版本class文件，为防止低版本jdk编译高版本class文件出错，在Android工程的`gradle.properties`文件中可添加如下设置语句：

```java
android.jetifier.blacklist=bcprov
```

在Android工程中设置好了如上所示依赖之后，只需依赖 AAR包和模型脚本对应的jar包`quick_start_flclient.jar`即可调用联邦学习提供的相关接口，接口的具体调用和运行方式可参考[联邦学习接口介绍部分](https://www.mindspore.cn/federated/docs/zh-CN/r1.7/index.html)。

## Linux x86_64环境

### 编译出包

1. 配置编译环境。

    目前只支持Linux环境编译，Linux编译环境配置可参考[这里](https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/build.html#linux环境编译)。

2. 在mindspore根目录进行编译，编译x86架构相关包。

    ```sh
    bash build.sh -I x86_64 -j32
    ```

   编译完成后在`mindspore/output`路径下将生成x86架构包，请备份备用(`mindspore/output`每次编译均会进行清理)。

    ```text
    mindspore-lite-{version}-linux-x64.tar.gz
    ```

3. 由于端侧框架和模型是解耦的，我们提供的x86架构包`mindspore-lite-{version}-linux-x64.tar.gz`不包含模型相关脚本，因此需要用户自行生成模型脚本对应的jar包，我们提供了两个类型的模型脚本供大家参考（[有监督情感分类任务](https://gitee.com/mindspore/mindspore/tree/r1.7/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert)、[LeNet图片分类任务](https://gitee.com/mindspore/mindspore/tree/r1.7/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet)）。同时，用户可参考这两个类型的模型脚本，自定义模型脚本后生成对应的jar包（假设命名为`quick_start_flclient.jar`）。我们提供的模型脚本对应jar包可采用如下方式获取：

    下载[MindSpore开源仓](https://gitee.com/mindspore/mindspore)最新代码后，进行以下操作：

    ```sh
    cd mindspore/mindspore/lite/examples/quick_start_flclient
    sh build.sh -r "mindspore-lite-{version}-linux-x64.tar.gz"   # -r 后需给出最新x86架构包绝对路径(步骤2生成)
    ```

    运行以上指令后生成jar包路径为：`mindspore/mindspore/lite/examples/quick_start_flclient/target/quick_start_flclient.jar`。

### 运行依赖

- [Python](https://www.python.org/downloads/)>=3.7.0
- [OpenJDK](https://openjdk.java.net/install/) 1.8 到 1.15

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
quick_start_flclient.jar   # 模型脚本对应的jar包
```

可将路径`mindspore/output/mindspore-lite-{version}-linux-x64/runtime/lib/`以及`mindspore/output/mindspore-lite-{version}-linux-x64/runtime/third_party/libjpeg-turbo/lib`中联邦学习所依赖的so文件（共7个）放入一个文件夹，比如`/resource/x86libs/`。然后在x86中设置环境变量(下面需提供绝对路径)：

```sh
export LD_LIBRARY_PATH=/resource/x86libs/:$LD_LIBRARY_PATH
```

设置好依赖环境之后，可参考应用实践教程[实现一个端云联邦的图像分类应用(x86)](https://www.mindspore.cn/federated/docs/zh-CN/r1.7/image_classification_application.html)在x86环境中模拟启动多个客户端进行联邦学习。
