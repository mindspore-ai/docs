# Horizontal Federated Device-side Deployment

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/federated/docs/source_en/deploy_federated_client.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

This document describes how to compile and deploy Federated-Client.

## Linux Compilation Guidance

### System Environment and Third-party Dependencies

This section describes how to complete the device-side compilation of MindSpore federated learning. Currently, the federated learning device-side only provides compilation guidance on Linux, and other systems are not supported. The following table lists the system environment and third-party dependencies required for compilation.

| Software Name                 | Version  |  Functions |
|-----------------------| ------------ | ------------ |
| Ubuntu                | 18.04.02LTS   | Compiling and running MindSpore operating system  |
| [GCC](#installing-gcc)         | Between 7.3.0 to 9.4.0  | C++ compiler for compiling MindSpore |
| [git](#installing-git)         | -  | Source code management tools used by MindSpore |
| [CMake](#installing-cmake)     | 3.18.3 and above  | Compiling and building MindSpore tools |
| [Gradle](#installing-gradle)   | 6.6.1  | JVM-based building tools  |
| [Maven](#installing-maven)     | 3.3.1 and above  | Tools for managing and building Java projects  |
| [OpenJDK](#installing-openjdk) | Between 1.8 to 1.15  | Tools for managing and building Java projects  |

#### Installing GCC

Install GCC with the following command.

```bash
sudo apt-get install gcc-7 git -y
```

To install a higher version of GCC, use the following command to install GCC 8.

```bash
sudo apt-get install gcc-8 -y
```

Or install GCC 9.

```bash
sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-9 -y
```

#### Installing git

Install git with the following command.

```bash
sudo apt-get install git -y
```

#### Installing Cmake

Install [CMake](https://cmake.org/) with the following command.

```bash
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt-get install cmake -y
```

#### Installing Gradle

Install [Gradle](https://gradle.org/releases/) with the following command.

```bash
# Download the corresponding zip package and unzip it.
# Configure environment variables:
  export GRADLE_HOME=GRADLE path
  export GRADLE_USER_HOME=GRADLE path
# Add the bin directory to the PATH:
  export PATH=${GRADLE_HOME}/bin:$PATH
```

#### Installing Maven

Install [Maven](https://archive.apache.org/dist/maven/maven-3/) with the following command.

```bash
# Download the corresponding zip package and unzip it.
# Configure environment variables:
  export MAVEN_HOME=MAVEN path
# Add the bin directory to the PATH:
  export PATH=${MAVEN_HOME}/bin:$PATH
```

#### Installing OpenJDK

Install [OpenJDK](https://jdk.java.net/archive/) with the following command.

```bash
# Download the corresponding zip package and unzip it.
# Configure environment variables:
  export JAVA_HOME=JDK path
# Add the bin directory to the PATH:
  export PATH=${JAVA_HOME}/bin:$PATH
```

### Verifying Installation

Verify that the installation in [System environment and third-party dependencies](#system-environment-and-third-party-dependencies) is successful.

```text
Open a command window and enter: gcc --version
The following output identifies a successful installation:
  gcc version version number

Open a command window and enter：git --version
The following output identifies a successful installation:
  git version version number

Open a command window and enter：cmake --version
The following output identifies a successful installation:
  cmake version version number

Open a command window and enter：gradle --version
The following output identifies a successful installation:
  Gradle version number

Open a command window and enter：mvn --version
The following output identifies a successful installation:
  Apache Maven version number

Open a command window and enter：java --version
The following output identifies a successful installation:
  openjdk version version number

```

### Compilation Options

The `cli_build.sh` script in the federated learning device_client directory is used for compilation on the federated learning device-side.

#### Instructions for Using cli_build.sh Parameters

| Parameters | Parameter Description                 | Value Range | Default Values       |
| ---- | ------------------------ | -------- | ------------ |
| -p   | the download path of dependency external packages | string   | third |
| -c   | whether to reuse dependency packages previously downloaded | on and off  | on           |

### Compilation Examples

1. First, you need to download the source code from the gitee code repository before you can compile it.

    ```bash
    git clone https://gitee.com/mindspore/federated.git ./
    ```

2. Go to the mindspore_federated/device_client directory and execute the following command:

    ```bash
    bash cli_build.sh
    ```

3. Since the end-side framework and the model are decoupled, the x86 architecture package we provide, mindspore-lite-{version}-linux-x64.tar.gz, does not contain model-related scripts, so the user needs to generate the jar package corresponding to the model scripts. The jar package corresponding to the model scripts we provide can be obtained in the following way:

    ```bash
    cd federated/example/quick_start_flclient
    bash build.sh -r mindspore-lite-java-flclient.jar # After -r, you need to give the absolute path to the latest x86 architecture package (generated in Step 2, federated/mindspore_federated/device_client/build/libs/jarX86/mindspore-lite-java-flclient.jar)
    ```

After running the above command, the path of generated jar package is federated/example/quick_start_flclient/target/quick_start_flclient.jar.

### Building Dependency Environment

1. After extracting the file `federated/mindspore_federated/device_client/third/mindspore-lite-{version}-linux-x64.tar.gz`, the obtained directory structure is as follows:

    ```sh
    mindspore-lite-{version}-linux-x64
    ├── tools
    │   ├── benchmark_train # Tool for training model performance and accuracy tuning
    │   ├── converter       # Model converter
    │   └── cropper         # Library cropper
    │       ├── cropper                 # Executable files of library cropper
    │       └── cropper_mapping_cpu.cfg # Configuration files required for cropping the cpu library
    └── runtime
        ├── include  # Header files of training framework
        │   └── registry # Custom operator registration header files
        ├── lib      # Training framework library
        │   ├── libminddata-lite.a          # Static library files for image processing
        │   ├── libminddata-lite.so        # Dynamic library files for image processing
        │   ├── libmindspore-lite-jni.so   # jni dynamic library relied by MindSpore Lite inference framework
        │   ├── libmindspore-lite-train.a  # Static library relied by MindSpore Lite training framework
        │   ├── libmindspore-lite-train.so # Dynamic library relied by MindSpore Lite training framework
        │   ├── libmindspore-lite-train-jni.so # jni dynamic library relied by MindSpore Lite training framework
        │   ├── libmindspore-lite.a  # Static library relied by MindSpore Lite inference framework
        │   ├── libmindspore-lite.so  # Dynamic library relied by MindSpore Lite inference framework
        │   ├── mindspore-lite-java.jar    # MindSpore Lite training framework jar package
        │   └── mindspore-lite-java-flclient.jar  # Federated learning framework jar package
        └── third_party
            └── libjpeg-turbo
                └── lib
                    ├── libjpeg.so.62   # Dynamic library files for image processing
                    └── libturbojpeg.so.0  # Dynamic library files for image processing
    ```

2. The names of the relevant x86 packages required for federated learning are as follows:

    ```sh
    libminddata-lite.so  # Dynamic library files for image processing
    libmindspore-lite.so  # Dynamic libraries relied by MindSpore Lite inference framework
    libmindspore-lite-jni.so  # jni dynamic library relied by MindSpore Lite inference framework
    libmindspore-lite-train.so  # Dynamic library relied by MindSpore Lite training framework
    libmindspore-lite-train-jni.so # jni dynamic library relied by MindSpore Lite training framework
    libjpeg.so.62   # Dynamic library files for image processing
    libturbojpeg.so.0  # Dynamic library files for image processing
    ```

3. Put the so files (7 in total) relied by federated learning in paths `mindspore-lite-{version}-linux-x64/runtime/lib/` and `mindspore-lite-{version}-linux-x64/runtime/third_party/libjpeg-turbo/lib` in a folder, e.g. `/resource/x86libs/`. Then set the environment variables in x86 (absolute paths need to be provided below):

    ```sh
    export LD_LIBRARY_PATH=/resource/x86libs/:$LD_LIBRARY_PATH
    ```

4. After setting up the dependency environment, you can simulate starting multiple clients in the x86 environment for federated learning by referring to the application practice tutorial [Implementing an end-cloud federation for image classification application (x86)](https://www.mindspore.cn/federated/docs/en/r2.0.0-alpha/image_classification_application.html).


