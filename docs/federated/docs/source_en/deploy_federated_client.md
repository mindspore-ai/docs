# On-Device Deployment

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/federated/docs/source_en/deploy_federated_client.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

The following describes how to deploy the Federated-Client in the Android and x86 environments:

## Android

### Building a Package

- Configure the build environment.

    Currently, only the Linux build environment is supported. For details about how to configure the Linux build environment, click [here](https://www.mindspore.cn/lite/docs/en/r1.6/use/build.html#linux).

- Turn on Federated-Client compile option and build the AAR package that contains aarch64 and aarch32 in the mindspore home directory.

    ```sh
    export MSLITE_ENABLE_FL=on
    bash build.sh -A on -j32
    ```

- Store the generated AAR package in the following path:

    ```sh
    mindspore-lite-maven-{version}.zip
    ```

- Since the device-side framework and the model are decoupled, the Android AAR package provided by us does not contain model-related scripts, so the user needs to generate the jar corresponding to the model script. We provide two types of model scripts for your reference ([Supervised sentiment Classification Task](https://gitee.com/mindspore/mindspore/tree/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert), [LeNet image classification task](https://gitee.com/mindspore/mindspore/tree/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet)). At the same time, users can refer to these two types of model scripts, customize the model script and generate the corresponding jar package (assuming the name is `quick_start_flclient.jar`). The jar packages corresponding to the model scripts we provide can be obtained in the following ways:

    After downloading the latest code on [MindSpore Open Source Warehouse](https://gitee.com/mindspore/mindspore), perform the following operations:

    ```sh
    cd mindspore/mindspore/lite/examples/quick_start_flclient
    sh build.sh -r "mindspore-lite-{version}-linux-x64.tar.gz"   // -r need be followed by the absolute path of the latest x86 architecture package
    ```

    After running the above command, the path of the jar package generated is: `mindspore/mindspore/lite/examples/quick_start_flclient/target/quick_start_flclient.jar`

### Running Dependencies

- [Android Studio](https://developer.android.google.cn/studio) >= 4.0
- [Android SDK](https://developer.android.com/studio?hl=zh-cn#cmdline-tools) >= 29

### Building a Dependency Environment

After the `mindspore-lite-maven-{version}.zip` file is decompressed, the following directory structure is obtained:

```sh
mindspore-lite-maven-{version}
└── mindspore
    └── mindspore-lite
        └── {version}
            └── mindspore-lite-{version}.aar  # MindSpore Lite training framework AAR package
```

Therefore, the AAR package path related to federated learning is as follows:

```sh
mindspore/output/mindspore/mindspore-lite/{version}/mindspore-lite-{version}.aar
```

The directory structure related to federated learning in the AAR package is as follows:

```sh
mindspore-lite-{version}
├── jni
│   ├── arm64-v8a
│   │   ├── libjpeg.so   # Dynamic library file for image processing
│   │   ├── libminddata-lite.so  # Dynamic library file for image processing
│   │   ├── libmindspore-lite.so  # Dynamic library on which the MindSpore Lite inference framework depends
│   │   ├── libmindspore-lite-jni.so  # JNI dynamic library on which the MindSpore Lite inference framework depends
│   │   ├── libmindspore-lite-train.so  # Dynamic library on which the MindSpore Lite training framework depends
│   │   ├── libmindspore-lite-train-jni.so  # JNI dynamic library on which the MindSpore Lite training framework depends
│   │   └── libturbojpeg.so  # Dynamic library file for image processing
│   └── armeabi-v7a
 │       ├── libjpeg.so   # Dynamic library file for image processing
│       ├── libminddata-lite.so  # Dynamic library file for image processing
│       ├── libmindspore-lite.so  # Dynamic library on which the MindSpore Lite inference framework depends
│       ├── libmindspore-lite-jni.so  # JNI dynamic library on which the MindSpore Lite inference framework depends
│       ├── libmindspore-lite-train.so  # Dynamic library on which the MindSpore Lite training framework depends
│       ├── libmindspore-lite-train-jni.so  # JNI dynamic library on which the MindSpore Lite training framework depends
│       └── libturbojpeg.so  # Dynamic library file for image processing
├── libs
│   ├── mindspore-lite-java-common.jar  # MindSpore Lite training framework JAR package
│   └── mindspore-lite-java-flclient.jar  # Federated learning framework JAR package
└── classes.jar  # MindSpore Lite training framework JAR package
```

Note 1: since the federated learning jar package in the Android environment does not contain the dependent third-party open source software packages, in the Android environment, before using the AAR package, the user needs to add related dependency statements in the dependencies{} field to load the three open source software that Federated Learning depends on, and the dependencies{} field is in the app/build.gradle file under the Android project, as shown below:

```text
dependencies {

// Add third-party open source software that federated learning relies on
implementation group: 'com.squareup.okhttp3', name: 'okhttp', version: '3.14.9'
implementation group: 'com.google.flatbuffers', name: 'flatbuffers-java', version: '2.0.0'
implementation(group: 'org.bouncycastle',name: 'bcprov-jdk15on', version: '1.68')
}
```

For specific implementation, please refer to the example of `app/build.gradle` provided in the `Android project configuration dependencies` section in the document [sentiment classification application](https://www.mindspore.cn/federated/docs/en/r1.6/sentiment_classification_application.html).

Note 2: since the third-party open source software `bcprov-jdk15on` that Federated Learning relies on contains multi-version class files, in order to prevent errors in compiling high-version class files with lower version jdk, the following setting statement can be added to the `gradle.properties` file of the Android project:

```java
android.jetifier.blacklist=bcprov
```

After setting up the dependencies shown above in the Android project, you only need to rely on the AAR package and the jar package corresponding to the model script `quick_start_flclient.jar` to call APIs provided by federated learning. For details about how to call and run the APIs, see the API description of federated learning.

## x86

### Building a Package

- Configure the build environment.

    Currently, only the Linux build environment is supported. For details about how to configure the Linux build environment, click [here](https://www.mindspore.cn/lite/docs/en/r1.6/use/build.html#linux).

- Build the x86-related architecture package in the mindspore home directory.

    ```sh
    bash build.sh -I x86_64 -j32
    ```

- Store the generated x86 architecture package in the following path:

    ```sh
    mindspore/output/mindspore-lite-{version}-linux-x64.tar.gz
    ```

- Since the device-side framework and the model are decoupled, the x86 architecture package `mindspore-lite-{version}-linux-x64.tar.gz` provided by us does not contain model-related scripts, so the user needs to generate the jar corresponding to the model script. We provide two types of model scripts for your reference ([Supervised sentiment Classification Task](https://gitee.com/mindspore/mindspore/tree/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert), [LeNet image classification task](https://gitee.com/mindspore/mindspore/tree/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet)). At the same time, users can refer to these two types of model scripts, customize the model script and generate the corresponding jar package (assuming the name is `quick_start_flclient.jar`). The jar packages corresponding to the model scripts we provide can be obtained in the following ways:

    After downloading the latest code on [MindSpore Open Source Warehouse](https://gitee.com/mindspore/mindspore), perform the following operations:

    ```sh
    cd mindspore/mindspore/lite/examples/quick_start_flclient
    sh build.sh -r "mindspore-lite-{version}-linux-x64.tar.gz"   // -r need be followed by the absolute path of the latest x86 architecture package
    ```

    After running the above command, the path of the jar package generated is: `mindspore/mindspore/lite/examples/quick_start_flclient/target/quick_start_flclient.jar`

### Running Dependencies

- [Python](https://www.python.org/downloads/) >= 3.7.5
- [OpenJDK](https://openjdk.java.net/install/) >= 1.9

### Building a Dependency Environment

After the `mindspore/output/mindspore-lite-{version}-linux-x64.tar.gz` file is decompressed, the following directory structure is obtained:

```sh
mindspore-lite-{version}-linux-x64
├── tools
│   ├── benchmark_train # Tool for commissioning the performance and accuracy of the training model
│   ├── converter       # Model conversion tool
│   └── cropper         # Library cropping tool
│       ├── cropper                 # Executable file of the library cropping tool
│       └── cropper_mapping_cpu.cfg # Configuration file required for cropping the CPU library
└── runtime
    ├── include  # Header file of the training framework
    │   └── registry # Header file for custom operator registration
    ├── lib      # Training framework library
    │   ├── libminddata-lite.a          # Static library file for image processing
    │   ├── libminddata-lite.so        # Dynamic library file for image processing
    │   ├── libmindspore-lite-jni.so   # JNI dynamic library on which the MindSpore Lite inference framework depends
    │   ├── libmindspore-lite-train.a  # Static library on which the MindSpore Lite training framework depends
    │   ├── libmindspore-lite-train.so # Dynamic library on which the MindSpore Lite training framework depends
    │   ├── libmindspore-lite-train-jni.so # JNI dynamic library on which the MindSpore Lite training framework depends
    │   ├── libmindspore-lite.a  # Static library on which the MindSpore Lite inference framework depends
    │   ├── libmindspore-lite.so  # Dynamic library on which the MindSpore Lite inference framework depends
    │   ├── mindspore-lite-java.jar    # MindSpore Lite training framework JAR package
    │   └── mindspore-lite-java-flclient.jar  # Federated learning framework JAR package
    └── third_party
        └── libjpeg-turbo
            └── lib
                ├── libjpeg.so.62   # Dynamic library file for image processing
                └── libturbojpeg.so.0  # Dynamic library file for image processing
```

The x86 packages required for federated learning are as follows:

```sh
libjpeg.so.62   # Dynamic library file for image processing
libminddata-lite.so  # Dynamic library file for image processing
libmindspore-lite.so  # Dynamic library on which the MindSpore Lite inference framework depends
libmindspore-lite-jni.so  # JNI dynamic library on which the MindSpore Lite inference framework depends
libmindspore-lite-train.so  # Dynamic library on which the MindSpore Lite training framework depends
libmindspore-lite-train-jni.so # JNI dynamic library on which the MindSpore Lite training framework depends
libturbojpeg.so.0  # Dynamic library file for image processing
mindspore-lite-java-flclient.jar  # Federated learning framework JAR package
quick_start_flclient.jar  # The jar package corresponding to the model script
```

Find the seven  .so files on which federated learning depends in the directories `mindspore/output/mindspore-lite-{version}-linux-x64/runtime/lib/` and `mindspore/output/mindspore-lite-{version}-linux-x64/runtime/third_party/libjpeg-turbo/lib`. Then, place these .so files in a folder, for example, `/resource/x86libs/`.

Set environment variables in the x86 system (an absolute path must be provided):

```sh
export LD_LIBRARY_PATH=/resource/x86libs/:$LD_LIBRARY_PATH
```

After the dependency environment is set, you can simulate the startup of multiple clients in the x86 environment for federated learning. For details, click [here](https://gitee.com/mindspore/docs/blob/r1.6/docs/federated/docs/source_en/image_classification_application.md).
