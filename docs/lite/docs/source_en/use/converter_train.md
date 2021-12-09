# Creating MindSpore Lite Models

`Linux` `Environment Preparation` `Model Export` `Model Converting` `Intermediate` `Expert`

<!-- TOC -->

- [Creating MindSpore Lite Models](#creating-mindspore-lite-model)
    - [Overview](#overview)
    - [Linux Environment](#linux-environment)
        - [Environment Preparation](#environment-preparation)
        - [Parameters Description](#parameters-description)
        - [Example](#example)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/use/converter_train.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

Creating your MindSpore Lite(Train on Device) model is a two step procedure:

- In the first step the model is defined and the layers that should be trained must be declared. This is being done on the server, using a MindSpore-based [Python code](https://www.mindspore.cn/docs/programming_guide/en/master/save_model.html#export-mindir-model). The model is then <b>exported</b> into a protobuf format, which is called MINDIR.
- In the seconde step this `.mindir` model is <b>converted</b> into a `.ms` format that can be loaded onto an embedded device and can be trained using the MindSpore Lite framework. The converted `.ms` models can be used for both training and inference.

## Linux Environment

### Environment Preparation

MindSpore Lite model transfer tool (only suppot Linux OS) has provided multiple parameters. The procedure is as follows:

- [Compile](https://www.mindspore.cn/lite/docs/en/master/use/build.html) or [download](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html) model transfer tool.

- Add the path of dynamic library required by the conversion tool to the environment variables LD_LIBRARY_PATH.

    ```bash
    export LD_LIBRARY_PATH=${PACKAGE_ROOT_PATH}/tools/converter/lib:${LD_LIBRARY_PATH}
    ````

    ${PACKAGE_ROOT_PATH} is the decompressed package path obtained by compiling or downloading.

### Parameters Description

The table below shows the parameters used in the MindSpore Lite model training transfer tool.

| Parameters                  | required | Parameter Description                                        | Value Range | Default Value |
| --------------------------- | -------- | ------------------------------------------------------------ | ----------- | ------------- |
| `--help`                    | no       | Prints all the help information.                             | -           | -             |
| `--fmk=<FMK>`               | yes      | Original format of the input model.                          | MINDIR      | -             |
| `--modelFile=<MODELFILE>`   | yes      | Path of the input model.                                     | -           | -             |
| `--outputFile=<OUTPUTFILE>` | yes      | Path of the output model. The suffix `.ms` can be automatically generated. | -           | -             |
| `--trainModel=true`         | yes      | Training on Device or not                                    | true, false | false         |
| `--configFile=<CONFIGFILE>` | No | 1) Configure quantization parameter; 2) Profile path for extension. | - | - |

> The parameter name and parameter value are separated by an equal sign (=) and no space is allowed between them.
>
> The calibration dataset configuration file uses the `key=value` mode to define related parameters. For the configuration parameters related to quantization, please refer to [post training quantization](https://www.mindspore.cn/lite/docs/en/master/use/post_training_quantization.html).

If running the conversion command is failed, an errorcode will be output.

### Example

Suppose the file to be converted is `my_model.mindir` and run the following command:

```bash
./converter_lite --fmk=MINDIR --trainModel=true --modelFile=my_model.mindir --outputFile=my_model
```

If the command executes successfully, the `model.ms` target file will be obtained and the console will print as follows:

```bash
CONVERTER RESULT SUCCESS:0
```

If running the conversion command is failed, an errorcode will be output.