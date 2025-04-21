# Device-side Training Model Conversion

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/train/converter_train.md)

## Overview

Creating your MindSpore Lite(Train on Device) model is a two step procedure:

- In the first step, create a network model based on the MindSpore architecture using Python and export it as a `.mindir` file. See [saving model](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/save_load.html#saving-and-loading-mindir) in the cloud.
- In the seconde step, this `.mindir` model is <b>converted</b> into a `.ms` format that can be loaded onto an embedded device and can be trained using the MindSpore Lite framework.

## Linux Environment

### Environment Preparation

MindSpore Lite model transfer tool (only suppot Linux OS) has provided multiple parameters. The procedure is as follows:

- [Compile](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html) or [download](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) model transfer tool.
- Add the path of dynamic library required by the conversion tool to the environment variables LD_LIBRARY_PATH.

    ```bash
    export LD_LIBRARY_PATH=${PACKAGE_ROOT_PATH}/tools/converter/lib:${LD_LIBRARY_PATH}
    ````

    ${PACKAGE_ROOT_PATH} is the decompressed package path obtained by compiling or downloading.

### Parameters Description

The table below shows the parameters used in the MindSpore Lite model training transfer tool.

| Parameters                  | required | Parameter Description                                                      | Value Range | Default Value |
| --------------------------- |----------|----------------------------------------------------------------------------| ----------- | ------------- |
| `--help`                    | no       | Prints all the help information.                                           | -           | -             |
| `--fmk=<FMK>`               | yes      | Original format of the input model.                                        | MINDIR      | -             |
| `--modelFile=<MODELFILE>`   | yes      | Path of the input model.                                                   | -           | -             |
| `--outputFile=<OUTPUTFILE>` | yes      | Path of the output model. The suffix `.ms` can be automatically generated. | -           | -             |
| `--trainModel=true`         | no       | If the original model is a training model, the value must be true.         | true, false | false         |
| `--configFile=<CONFIGFILE>` | No       | 1. Configure quantization parameter; 2. Profile path for extension.        | - | - |

> The parameter name and parameter value are separated by an equal sign (=) and no space is allowed between them.
>
> The calibration dataset configuration file uses the `key=value` mode to define related parameters. For the configuration parameters related to quantization, please refer to [quantization](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/advanced/quantization.html).

### Example

Suppose the file to be converted is `my_model.mindir` and run the following command:

```bash
./converter_lite --fmk=MINDIR --trainModel=true --modelFile=my_model.mindir --outputFile=my_model
```

The output of successful conversion is as follows:

```text
CONVERT RESULT SUCCESS:0
```

This indicates that the MindSpore model is successfully converted to a MindSpore end-side model and a new file `my_model.ms` is generated. If the output of conversion failure is as follows:

```text
CONVERT RESULT FAILED:
```

The program returns error codes and error messages.
