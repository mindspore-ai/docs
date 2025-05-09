# Static Library Cropper Tool

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/docs/lite/docs/source_en/tools/cropper_tool.md)

## Overview

MindSpore Lite provides the `libmindspore-lite.a` static library cropping tool for runtime, which can filter out the operators in the `ms` model, crop the static library files. If the size requirement is still not met after operator cropping, you can [recompile](https://www.mindspore.cn/lite/docs/en/r2.6.0/build/build.html) the inference framework package and use the `Framework Function Cropping Compile Option` at compile time to crop the framework function, and then use the tool to perform the operator cropping afterwards.

The operating environment of the library cropping tool is x86_64, and currently supports the cropping of CPU or GPU operators, and the GPU library supports setting CMAKE's MSLITE_GPU_BACKEND to opencl. After cropping the operator, the cropped static libraries can be compiled into dynamic libraries to suit different needs.

## Environment Preparation

To use the Cropper tool, you need to prepare the environment as follows:

- Compilation: The code of the Cropper tool is stored in the `mindspore/lite/tools/cropper` directory of the MindSpore source code. For details about the build operations, see the [Environment Requirements](https://www.mindspore.cn/lite/docs/en/r2.6.0/build/build.html#environment-requirements) and [Compilation Example](https://www.mindspore.cn/lite/docs/en/r2.6.0/build/build.html#compilation-example) in the build document to compile version x86_64.

- Run: Obtain the `cropper` tool and configure environment variables. For details, see [Output Description](https://www.mindspore.cn/lite/docs/en/r2.6.0/build/build.html#environment-requirements) in the build document.

## Parameter Description

The command used for crop the static library based on Cropper is as follows:

```text
./cropper [--packageFile=<PACKAGEFILE>] [--configFile=<CONFIGFILE>]
          [--modelFile=<MODELFILE>] [--modelFolderPath=<MODELFOLDERPATH>]
          [--outputFile=<MODELFILE>] [--help]
```

The following describes the parameters in detail.

| Parameter                                  | Attribute | Function                                                     | Parameter Type | Default Value | Value Range |
| ------------------------------------- | -------- | ------------------------------------------------------------ | -------- | ------ | -------- |
| `--packageFile=<PACKAGEFILE>`         | Mandatory       |The path of the `libmindspore-lite.a` to be cropped.                  | String   | -      | -        |
| `--configFile=<CONFIGFILE>`           | Mandatory       | The path of the configuration file of the cropper tool. The file path of `cropper_mapping_cpu.cfg` or `cropper_mapping_gpu.cfg` needs to be set to crop the CPU or GPU library. | String   | -      | -        |
| `--modelFolderPath=<MODELFOLDERPATH>` | Optional       | The model folder path, according to all the `ms` models existing in the folder for library cropping. `modelFile` or `modelFolderPath` parameters must be selected. | String   | -      | -        |
| `--modelFile=<MODELFILE>`             | Optional       | The model file path is cut according to the specified `ms` model file. Multiple model files are divided by `,`. `modelFile` or `modelFolderPath` parameters must be selected. | String   | -      | -        |
| `--outputFile=<OUTPUTFILE>`           | Optional       | The saved path of the cut library `libmindspore-lite.a`, it overwrites the source file by default. | String   | -      | -        |
| `--help`                              | Optional       | Displays the help information about the `cropper` command. | -        | -      | -        |

> The configuration file `cropper_mapping_cpu.cfg`  `cropper_mapping_gpu.cfg` exists in the `tools/cropper` directory in the `mindspore-lite-{version}-linux-x64` package.

## Example

The Cropper tool obtains the operator list by parsing the `ms` model, and crop the `libmindspore-lite.a` static library according to the mapping relationship in the configuration file `configFile`.

- Pass in the `ms` model through the folder, and pass the folder path where the model file is located to the `modelFolderPath` parameter to crop the `libmindspore-lite.a` static library of arm64-cpu.

  ```bash
  ./cropper --packageFile=/mindspore-lite-{version}-android-aarch64/runtime/lib/libmindspore-lite.a --configFile=./cropper_mapping_cpu.cfg --modelFolderPath=/model --outputFile=/mindspore-lite/lib/libmindspore-lite.a
  ```

  This example will read all the `ms` models contained in the `/model` folder, crop the `libmindspore-lite.a` static library of arm64-cpu, and the cropped `libmindspore-lite.a` static library will be saved to `/mindspore-lite/lib/` directory.

- Pass in the `ms` model by file, pass the path where the model file is located to the `modelFile` parameter, and crop the `libmindspore-lite.a` static library of arm64-cpu.

  ```bash
  ./cropper --packageFile=/mindspore-lite-{version}-android-aarch64/runtime/lib/libmindspore-lite.a --configFile=./cropper_mapping_cpu.cfg --modelFile=/model/lenet.ms,/model/retinaface.ms  --outputFile=/mindspore-lite/lib/libmindspore-lite.a
  ```

  In this example, the `libmindspore-lite.a` static library of arm64-cpu will be cropped according to the `ms` model passed by `modelFile`, and the cropped `libmindspore-lite.a` static library will be saved to `/mindspore-lite/lib/` directory.

- Pass in the `ms` model through the folder, and pass the folder path where the model file is located to the `modelFolderPath` parameter to crop the `libmindspore-lite.a` static library of arm64-gpu.

  ```bash
  ./cropper --packageFile=/mindspore-lite-{version}-android-aarch64/runtime/lib/libmindspore-lite.a --configFile=./cropper_mapping_gpu.cfg --modelFolderPath=/model --outputFile=/mindspore-lite/lib/libmindspore-lite.a
  ```

  This example will read all the `ms` models contained in the `/model` folder, crop the `libmindspore-lite.a` static library of arm64-gpu, and the cropped `libmindspore-lite.a` static library will be saved to `/mindspore-lite/lib/` directory.

- Pass in the `ms` model by file, pass the path where the model file is located to the `modelFile` parameter, and crop the `libmindspore-lite.a` static library of arm64-gpu.

  ```bash
  ./cropper --packageFile=/mindspore-lite-{version}-android-aarch64/runtime/lib/libmindspore-lite.a --configFile=./cropper_mapping_gpu.cfg --modelFile=/model/lenet.ms,/model/retinaface.ms  --outputFile=/mindspore-lite/lib/libmindspore-lite.a
  ```

  In this example, the `libmindspore-lite.a` static library of arm64-gpu will be cropped according to the `ms` model passed by `modelFile`, and the cropped `libmindspore-lite.a` static library will be saved to `/mindspore-lite/lib/` directory.

## Compiling Static Library Into Dynamic Library (Optional)

After cropping the static library, if necessary, the cropped static library can be compiled into a dynamic library.
The compilation environment requirements refer to the compilation requirements of MindSpore Lite [compilation](https://www.mindspore.cn/lite/docs/en/master/build/build.html).
The compilation commands used for packages under different architectures are different.
The specific commands can be obtained through the commands printed during the compilation of MindSpore Lite .
Refer to the example steps below.

1. Add the following command to `lite/Cmakelist.txt` to enable the compilation process to print.

    ```text
    set(CMAKE_VERBOSE_MAKEFILE on)
    ```

2. Refer to the [MindSpore Lite compilation](https://www.mindspore.cn/lite/docs/en/r2.6.0/build/build.html) to compile the runtime package on the specific architecture required.

3. After the compilation is completed, find the command for compiling libminspore-lite.so in the printed compilation information. The following is the print command when compiling the runtime package of arm64 architecture, where `/home/android-ndk-r20b` is the path of the installed Android SDK.

    ```bash
    /home/android-ndk-r20b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --gcc-toolchain=/home/android-ndk-r20b/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=/home/android-ndk-r20b/toolchains/llvm/prebuilt/linux-x86_64/sysroot -fPIC -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -Wno-attributes -Wno-deprecated-declarations         -Wno-missing-braces -Wno-overloaded-virtual -std=c++17 -fPIC -fPIE -fstack-protector-strong  -DANDROID -fdata-sections -ffunction-sections -funwind-tables -fstack-protector-strong -no-canonical-prefixes -fno-addrsig -Wa,--noexecstack -Wformat -Werror=format-security    -fomit-frame-pointer -fstrict-aliasing -ffunction-sections         -fdata-sections -ffast-math -fno-rtti -fno-exceptions -Wno-unused-private-field -O2 -DNDEBUG  -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -s  -Wl,--exclude-libs,libgcc.a -Wl,--exclude-libs,libatomic.a -static-libstdc++ -Wl,--build-id -Wl,--warn-shared-textrel -Wl,--fatal-warnings -Wl,--no-undefined -Qunused-arguments -Wl,-z,noexecstack  -shared -Wl,-soname,libmindspore-lite.so -o libmindspore-lite.so @CMakeFiles/mindspore-lite.dir/objects1.rsp  -llog -ldl -latomic -lm
    ```

4. Modify the command, replace the object to be compiled, and compile the cropped static library into a dynamic library.

    Take the above print command as an example to find the object `@CMakeFiles/mindspore-lite.dir/objects1.rsp` to be compiled in the command, replace with the cropped static library object `-Wl,--whole-archive ./libmindspore-lite.a -Wl,--no-whole-archive`, Where `./libmindspore-lite.a` is the cropped static library path. You can replace it with the path of your own library. The modified command is as follows.

    ```bash
    /home/android-ndk-r20b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --gcc-toolchain=/home/android-ndk-r20b/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=/home/android-ndk-r20b/toolchains/llvm/prebuilt/linux-x86_64/sysroot -fPIC -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -Wno-attributes -Wno-deprecated-declarations         -Wno-missing-braces -Wno-overloaded-virtual -std=c++17 -fPIC -fPIE -fstack-protector-strong  -DANDROID -fdata-sections -ffunction-sections -funwind-tables -fstack-protector-strong -no-canonical-prefixes -fno-addrsig -Wa,--noexecstack -Wformat -Werror=format-security    -fomit-frame-pointer -fstrict-aliasing -ffunction-sections         -fdata-sections -ffast-math -fno-rtti -fno-exceptions -Wno-unused-private-field -O2 -DNDEBUG  -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -s  -Wl,--exclude-libs,libgcc.a -Wl,--exclude-libs,libatomic.a -static-libstdc++ -Wl,--build-id -Wl,--warn-shared-textrel -Wl,--fatal-warnings -Wl,--no-undefined -Qunused-arguments -Wl,-z,noexecstack  -shared -Wl,-soname,libmindspore-lite.so -o libmindspore-lite.so -Wl,--whole-archive ./libmindspore-lite.a -Wl,--no-whole-archive  -llog -ldl -latomic -lm
    ```

    Use this command to compile the clipped static library into a dynamic library and generate `libminspore-lite.so` in the current directory.

> - In the command example, `-static-libstdc++` indicates the integration of static STD library. You can delete the command and link the dynamic STD library instead to reduce the package size.
