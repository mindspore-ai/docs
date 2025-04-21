# 静态库裁剪工具

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/tools/cropper_tool.md)

## 概述

MindSpore Lite提供对Runtime的`libmindspore-lite.a`静态库裁剪工具，能够筛选出`ms`模型中存在的算子，对静态库文件进行算子裁剪。若进行算子裁剪之后，仍然不能满足大小要求，可重新[编译](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html)推理框架包，在编译时使用`框架功能裁剪编译选项`进行框架功能裁剪，之后再使用本工具进行算子裁剪。

裁剪工具运行环境是x86_64，目前支持对CPU、GPU算子的裁剪，其中GPU库支持`lite/Cmakelist.txt`的MSLITE_GPU_BACKEND设置为opencl。在裁剪完算子后，可将裁剪后的静态库编译为动态库以适应不同需求。

## 环境准备

使用MindSpore Lite裁剪工具，需要进行如下环境准备工作。

- 编译：裁剪工具代码在MindSpore源码的`mindspore/lite/tools/cropper`目录中，参考构建文档中的[环境要求](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html#环境要求)和[编译示例](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html#编译示例)编译x86_64版本。

- 运行：参考构建文档中的[编译输出](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html#目录结构)，获得`cropper`工具。

## 参数说明

使用裁剪工具进行静态库的裁剪，其命令格式如下所示。

```text
./cropper [--packageFile=<PACKAGEFILE>] [--configFile=<CONFIGFILE>]
          [--modelFile=<MODELFILE>] [--modelFolderPath=<MODELFOLDERPATH>]
          [--outputFile=<MODELFILE>] [--help]
```

下面提供详细的参数说明。

| 参数                                  | 是否必选 | 参数说明                                                     | 参数类型 | 默认值 | 取值范围 |
| ------------------------------------- | -------- | ------------------------------------------------------------ | -------- | ------ | -------- |
| `--packageFile=<PACKAGEFILE>`         | 是       | 需要裁剪的`libmindspore-lite.a`文件路径。                    | String   | -      | -        |
| `--configFile=<CONFIGFILE>`           | 是       | 裁剪工具配置文件的路径，裁剪CPU、GPU库需要分别设置`cropper_mapping_cpu.cfg`、`cropper_mapping_gpu.cfg`文件路径。 | String   | -      | -        |
| `--modelFolderPath=<MODELFOLDERPATH>` | 否       | 模型文件夹路径，根据文件夹中存在的所有`ms`模型进行库裁剪。`modelFile`和`modelFolderPath`参数必须二选一。 | String   | -      | -        |
| `--modelFile=<MODELFILE>`             | 否       | 模型文件路径，根据指定的`ms`模型文件进行库裁剪，多个模型文件采用`,`分割。`modelFile`和`modelFolderPath`参数必须二选一。 | String   | -      | -        |
| `--outputFile=<OUTPUTFILE>`           | 否       | 裁剪完成的`libmindspore-lite.a`库的保存路径，默认覆盖源文件。 | String   | -      | -        |
| `--help`                              | 否       | 打印全部帮助信息。                                           | -        | -      | -        |

> 配置文件`cropper_mapping_cpu.cfg` `cropper_mapping_gpu.cfg`存在于`mindspore-lite-{version}-linux-x64`包中的`tools/cropper`目录。

## 使用示例

裁剪工具通过解析`ms`模型得到算子列表，并根据配置文件`configFile`中的映射关系来裁剪`libmindspore-lite.a`静态库。模型文件传入方式包括文件夹、文件两种：

- 通过文件夹的方式传入`ms`模型，将模型文件所在的文件夹路径传递给`modelFolderPath`参数，对arm64-cpu的`libmindspore-lite.a`静态库进行裁剪。

  ```bash
  ./cropper --packageFile=/mindspore-lite-{version}-android-aarch64/runtime/lib/libmindspore-lite.a --configFile=./cropper_mapping_cpu.cfg --modelFolderPath=/model --outputFile=/mindspore-lite/lib/libmindspore-lite.a
  ```

  本例将读取`/model`文件夹中包含的所有`ms`模型，对arm64-cpu的`libmindspore-lite.a`静态库进行裁剪，并将裁剪后的`libmindspore-lite.a`静态库保存到`/mindspore-lite/lib/`目录。

- 通过文件的方式传入`ms`模型，将模型文件所在的路径传递给`modelFile`参数，对arm64-cpu的`libmindspore-lite.a`静态库进行裁剪。

  ```bash
  ./cropper --packageFile=/mindspore-lite-{version}-android-aarch64/runtime/lib/libmindspore-lite.a --configFile=./cropper_mapping_cpu.cfg --modelFile=/model/lenet.ms,/model/retinaface.ms  --outputFile=/mindspore-lite/lib/libmindspore-lite.a
  ```

  本例将根据`modelFile`传入的`ms`模型，对arm64-cpu的`libmindspore-lite.a`静态库进行裁剪，并将裁剪后的`libmindspore-lite.a`静态库保存到`/mindspore-lite/lib/`目录。

- 通过文件夹的方式传入`ms`模型，将模型文件所在的文件夹路径传递给`modelFolderPath`参数，对arm64-gpu的`libmindspore-lite.a`静态库进行裁剪。

  ```bash
  ./cropper --packageFile=/mindspore-lite-{version}-android-aarch64/runtime/lib/libmindspore-lite.a --configFile=./cropper_mapping_gpu.cfg --modelFolderPath=/model --outputFile=/mindspore-lite/lib/libmindspore-lite.a
  ```

  本例将读取`/model`文件夹中包含的所有`ms`模型，对arm64-gpu的`libmindspore-lite.a`静态库进行裁剪，并将裁剪后的`libmindspore-lite.a`静态库保存到`/mindspore-lite/lib/`目录。

- 通过文件的方式传入`ms`模型，将模型文件所在的路径传递给`modelFile`参数，对arm64-gpu的`libmindspore-lite.a`静态库进行裁剪。

  ```bash
  ./cropper --packageFile=/mindspore-lite-{version}-android-aarch64/runtime/lib/libmindspore-lite.a --configFile=./cropper_mapping_gpu.cfg --modelFile=/model/lenet.ms,/model/retinaface.ms  --outputFile=/mindspore-lite/lib/libmindspore-lite.a
  ```

  本例将根据`modelFile`传入的`ms`模型，对arm64-gpu的`libmindspore-lite.a`静态库进行裁剪，并将裁剪后的`libmindspore-lite.a`静态库保存到`/mindspore-lite/lib/`目录。

## 裁剪后静态库编译为动态库so（可选）

在裁剪完静态库后，若有需要，可将裁剪后的静态库编译为动态库，编译环境要求参考MindSpore Lite[编译](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html)要求，不同架构下的包，所用的编译命令不同，具体命令可通过MindSpore Lite编译过程中打印的命令获取，参考示例步骤如下。

1. 在`lite/Cmakelist.txt`中添加如下命令，以开启编译过程命令打印。

    ```text
    set(CMAKE_VERBOSE_MAKEFILE on)
    ```

2. 参考MindSpore Lite[编译](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html)，编译所需特定架构上的推理包。

3. 在编译完成后，在打印的编译信息中，找到编译libmindspore-lite.so时的命令，下文为编译arm64架构的推理包时的打印命令，其中`/home/android-ndk-r20b`为安装的Android SDK路径。

    ```bash
    /home/android-ndk-r20b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --gcc-toolchain=/home/android-ndk-r20b/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=/home/android-ndk-r20b/toolchains/llvm/prebuilt/linux-x86_64/sysroot -fPIC -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -Wno-attributes -Wno-deprecated-declarations         -Wno-missing-braces -Wno-overloaded-virtual -std=c++17 -fPIC -fPIE -fstack-protector-strong  -DANDROID -fdata-sections -ffunction-sections -funwind-tables -fstack-protector-strong -no-canonical-prefixes -fno-addrsig -Wa,--noexecstack -Wformat -Werror=format-security    -fomit-frame-pointer -fstrict-aliasing -ffunction-sections         -fdata-sections -ffast-math -fno-rtti -fno-exceptions -Wno-unused-private-field -O2 -DNDEBUG  -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -s  -Wl,--exclude-libs,libgcc.a -Wl,--exclude-libs,libatomic.a -static-libstdc++ -Wl,--build-id -Wl,--warn-shared-textrel -Wl,--fatal-warnings -Wl,--no-undefined -Qunused-arguments -Wl,-z,noexecstack  -shared -Wl,-soname,libmindspore-lite.so -o libmindspore-lite.so @CMakeFiles/mindspore-lite.dir/objects1.rsp  -llog -ldl -latomic -lm
    ```

4. 修改命令，替换待编译对象，将裁剪后的静态库编译为动态库。

    以上条打印命令为例，找到原先命令里的待编译对象`@CMakeFiles/mindspore-lite.dir/objects1.rsp`，改为裁剪后的静态库对象`-Wl,--whole-archive ./libmindspore-lite.a -Wl,--no-whole-archive`，其中`./libmindspore-lite.a`为已裁剪后的静态库路径，用户可替换为自身库所在路径，修改后命令如下。

    ```bash
    /home/android-ndk-r20b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --gcc-toolchain=/home/android-ndk-r20b/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=/home/android-ndk-r20b/toolchains/llvm/prebuilt/linux-x86_64/sysroot -fPIC -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -Wno-attributes -Wno-deprecated-declarations         -Wno-missing-braces -Wno-overloaded-virtual -std=c++17 -fPIC -fPIE -fstack-protector-strong  -DANDROID -fdata-sections -ffunction-sections -funwind-tables -fstack-protector-strong -no-canonical-prefixes -fno-addrsig -Wa,--noexecstack -Wformat -Werror=format-security    -fomit-frame-pointer -fstrict-aliasing -ffunction-sections         -fdata-sections -ffast-math -fno-rtti -fno-exceptions -Wno-unused-private-field -O2 -DNDEBUG  -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -s  -Wl,--exclude-libs,libgcc.a -Wl,--exclude-libs,libatomic.a -static-libstdc++ -Wl,--build-id -Wl,--warn-shared-textrel -Wl,--fatal-warnings -Wl,--no-undefined -Qunused-arguments -Wl,-z,noexecstack  -shared -Wl,-soname,libmindspore-lite.so -o libmindspore-lite.so -Wl,--whole-archive ./libmindspore-lite.a -Wl,--no-whole-archive  -llog -ldl -latomic -lm
    ```

    使用该命令可将裁剪后的静态库编译为动态库，并在当前目录下生成`libmindspore-lite.so`。

> - 在命令示例中，`-static-libstdc++`为集成静态std库，用户可删除该命令，改为链接动态std库，以降低包大小。
