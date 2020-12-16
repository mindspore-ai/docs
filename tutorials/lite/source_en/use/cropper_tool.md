# Use Cropper Tool To Reduce Library File Size

`Windows` `Linux` `Environment Preparation` `Static Library Cropping` `Intermediate` `Expert`

<!-- TOC -->

- [Use Cropper Tool To Reduce Library File Size](#use-cropper-tool-to-reduce-library-file-size)
    - [Overview](#overview)
    - [Environment Preparation](#environment-preparation)
    - [Parameter Description](#parameter-description)
    - [Example](#example)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/lite/source_en/use/cropper_tool.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

MindSpore Lite provides the `libmindspore-lite.a` static library cropping tool for runtime, which can filter out the operators in the `ms` model, crop the static library files, and effectively reduce the size of the library files.

The operating environment of the library cutting tool is x86_64, and currently supports the cropping of CPU operators, namely `mindspore-lite-{version}-runtime-arm64-cpu`, `mindspore-lite-{version}-runtime-arm32-cpu`, the `libmindspore-lite.a` static library of `mindspore-lite-{version}-runtime-x86-cpu`.

## Environment Preparation

To use the Cropper tool, you need to prepare the environment as follows:

- Compilation: The code of the Cropper tool is stored in the `mindspore/lite/tools/lib_cropper` directory of the MindSpore source code. For details about the build operations, see the [Environment Requirements](https://www.mindspore.cn/tutorial/lite/en/master/use/build.html#environment-requirements) and [Compilation Example](https://www.mindspore.cn/tutorial/lite/en/master/use/build.html#compilation-example) in the build document to compile version x86_64 .

- Run: Obtain the `lib_cropper` tool and configure environment variables. For details, see [Output Description](https://www.mindspore.cn/tutorial/lite/en/master/use/build.html#output-description) in the build document.

## Parameter Description

The command used for crop the static library based on Cropper is as follows:

```bash
./lib_cropper [--packageFile=<PACKAGEFILE>] [--configFile=<CONFIGFILE>]
              [--modelFile=<MODELFILE>] [--modelFolderPath=<MODELFOLDERPATH>]
              [--outputFile=<MODELFILE>] [--help]
```

The following describes the parameters in detail.

| Parameter                                  | Attribute | Function                                                     | Parameter Type | Default Value | Value Range |
| ------------------------------------- | -------- | ------------------------------------------------------------ | -------- | ------ | -------- |
| `--packageFile=<PACKAGEFILE>`         | Mandatory       |The path of the `libmindspore-lite.a` to be cropped.                  | String   | -      | -        |
| `--configFile=<CONFIGFILE>`           | Mandatory       | The path of the configuration file of the cropper tool. The file path of `cropper_mapping_cpu.cfg` needs to be set to crop the CPU library. | String   | -      | -        |
| `--modelFolderPath=<MODELFOLDERPATH>` | Optional       | The model folder path, according to all the `ms` models existing in the folder for library cropping. `modelFile` or `modelFolderPath` parameters must be selected. | String   | -      | -        |
| `--modelFile=<MODELFILE>`             | Optional       | The model file path is cut according to the specified `ms` model file. Multiple model files are divided by `,`. `modelFile` or `modelFolderPath` parameters must be selected. | String   | -      | -        |
| `--outputFile=<OUTPUTFILE>`           | Optional       | The saved path of the cut library `libmindspore-lite.a`, it overwrites the source file by default. | String   | -      | -        |
| `--help`                              | Optional       | Displays the help information about the `lib_cropper` command. | -        | -      | -        |

> The configuration file `cropper_mapping_cpu.cfg` exists in the `lib_cropper` directory in the `mindspore-lite-{version}-runtime-x86-cpu` package.

## Example

The Cropper tool obtains the operator list by parsing the `ms` model, and crop the `libmindspore-lite.a` static library according to the mapping relationship in the configuration file `configFile`.

- Pass in the `ms` model through the folder, and pass the folder path where the model file is located to the `modelFolderPath` parameter to crop the `libmindspore-lite.a` static library of arm64-cpu.

```bash
./lib_cropper --packageFile=/mindspore-lite-{version}-runtime-arm64-cpu/lib/libmindspore-lite.a --configFile=./cropper_mapping_cpu.cfg --modelFolderPath=/model --outputFile=/mindspore-lite/lib/libmindspore-lite.a
```

This example will read all the `ms` models contained in the `/model` folder, crop the `libmindspore-lite.a` static library of arm64-cpu, and the cropped `libmindspore-lite.a` static library will be saved to `/mindspore-lite/lib/` directory.

- Pass in the `ms` model by file, pass the path where the model file is located to the `modelFile` parameter, and crop the `libmindspore-lite.a` static library of arm64-cpu.

```bash
./lib_cropper --packageFile=/mindspore-lite-{version}-runtime-arm64-cpu/lib/libmindspore-lite.a --configFile=./cropper_mapping_cpu.cfg --modelFile=/model/lenet.ms,/model/retinaface.ms  --outputFile=/mindspore-lite/lib/libmindspore-lite.a
```

In this example, the `libmindspore-lite.a` static library of arm64-cpu will be cropped according to the `ms` model passed by `modelFile`, and the cropped `libmindspore-lite.a` static library will be saved to `/mindspore-lite/lib/` directory.
