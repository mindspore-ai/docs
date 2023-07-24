﻿# On-Device Deployment

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/federated/docs/source_en/deploy_federated_client.md)

The following describes how to deploy the Federated-Client in the Android and x86 environments:

## Android

### Building a Package

- Configure the build environment.

    Currently, only the Linux build environment is supported. For details about how to configure the Linux build environment, click [here](https://www.mindspore.cn/lite/docs/en/r1.3/use/build.html#linux).

- Build the AAR package that contains aarch64 and aarch32 in the mindspore home directory.

    ```sh
    bash build.sh -A on -j32
    ```

- Store the generated AAR package in the following path:

    ```sh
    mindspore-lite-maven-{version}.zip
    ```

### Running Dependencies

- [Android Studio](https://developer.android.google.cn/studio) >= 3.2 (Android 4.0 or later is recommended.)
- [Android SDK](https://developer.android.com/studio?hl=zh-cn#cmdline-tools) >= 26 (installed in Android Studio by default)
- [OpenJDK](https://openjdk.java.net/install/) >= 1.8 (installed in Android Studio by default)

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
│   │   └── libturbojpeg.so  # Dynamic library file for image processing
│   └── armeabi-v7a
 │       ├── libjpeg.so   # Dynamic library file for image processing
│       ├── libminddata-lite.so  # Dynamic library file for image processing
│       ├── libmindspore-lite.so  # Dynamic library on which the MindSpore Lite inference framework depends
│       ├── libmindspore-lite-jni.so  # JNI dynamic library on which the MindSpore Lite inference framework depends
│       ├── libmindspore-lite-train.so  # Dynamic library on which the MindSpore Lite training framework depends
│       └── libturbojpeg.so  # Dynamic library file for image processing
├── libs
│   ├── mindspore-lite-java-common.jar  # MindSpore Lite training framework JAR package
│   └── mindspore-lite-java-flclient.jar  # Federated learning framework JAR package
└── classes.jar  # MindSpore Lite training framework JAR package
```

The Android project only depends on the AAR package to call APIs provided by federated learning. For details about how to call and run the APIs, see the API description of federated learning.

## x86

### Building a Package

- Configure the build environment.

    Currently, only the Linux build environment is supported. For details about how to configure the Linux build environment, click [here](https://www.mindspore.cn/lite/docs/en/r1.3/use/build.html#linux).

- Build the x86-related architecture package in the mindspore home directory.

    ```sh
    bash build.sh -I x86_64 -j32
    ```

- Store the generated x86 architecture package in the following path:

    ```sh
    mindspore/output/mindspore-lite-{version}-linux-x64.tar.gz
    ```

### Running Dependencies

- [Python](https://www.python.org/downloads/) >= 3.7.5
- [OpenJDK](https://openjdk.java.net/install/) >= 1.8

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
    │   ├── libmindspore-lite-jni.so   # JNI dynamic library  on which the MindSpore Lite inference framework depends
    │   ├── libmindspore-lite-train.a  # Static library on which the MindSpore Lite training framework depends
    │   ├── libmindspore-lite-train.so # Dynamic library on which the MindSpore Lite training framework depends
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
libturbojpeg.so.0  # Dynamic library file for image processing
mindspore-lite-java-flclient.jar  # Federated learning framework JAR package
```

Find the six .so files on which federated learning depends in the directories `mindspore/output/mindspore-lite-{version}-linux-x64/runtime/lib/` and `mindspore/output/mindspore-lite-{version}-linux-x64/runtime/third_party/libjpeg-turbo/lib`. Then, place these .so files in a folder, for example, `/resource/x86libs/`.

Set environment variables in the x86 system (an absolute path must be provided):

```sh
export LD_LIBRARY_PATH=/resource/x86libs/:$LD_LIBRARY_PATH
```

After the dependency environment is set, you can simulate the startup of multiple clients in the x86 environment for federated learning. For details, click [here](https://gitee.com/mindspore/docs/blob/r1.3/docs/federated/docs/source_en/image_classification_application.md).
