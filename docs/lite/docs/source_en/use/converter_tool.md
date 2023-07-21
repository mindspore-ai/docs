# Converting Models for Inference

`Windows` `Linux` `Model Converting` `Intermediate` `Expert`

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/lite/docs/source_en/use/converter_tool.md)

## Overview

MindSpore Lite provides a tool for offline model conversion. It supports conversion of multiple types of models. The converted models can be used for inference. The command line parameters contain multiple personalized options, providing a convenient conversion method for users.

Currently, the following input formats are supported: MindSpore, TensorFlow Lite, Caffe, TensorFlow and ONNX.

The ms model converted by the conversion tool supports the conversion tool and the higher version of the Runtime framework to perform inference.

## Linux Environment Instructions

### Environment Preparation

To use the MindSpore Lite model conversion tool, you need to prepare the environment as follows:

- [Compile](https://www.mindspore.cn/lite/docs/en/r1.3/use/build.html) or [download](https://www.mindspore.cn/lite/docs/en/r1.3/use/downloads.html) model transfer tool.

- Add the path of dynamic library required by the conversion tool to the environment variables LD_LIBRARY_PATH.

    ```bash
    export LD_LIBRARY_PATH=${PACKAGE_ROOT_PATH}/tools/converter/lib:${LD_LIBRARY_PATH}
    ````

    ${PACKAGE_ROOT_PATH} is the decompressed package path obtained by compiling or downloading.

### Directory Structure

```text
mindspore-lite-{version}-linux-x64
└── tools
    └── converter
        ├── include
        ├── converter                # Model conversion tool
        │   └── converter_lite       # Executable program
        └── lib                      # The dynamic link library that converter depends
            └── libglog.so.0         # Dynamic library of Glog
```

### Parameter Description

MindSpore Lite model conversion tool provides multiple parameters.
You can enter `./converter_lite --help` to obtain the help information in real time.

The following describes the parameters in detail.

| Parameter  |  Mandatory or Not   |  Parameter Description  | Value Range | Default Value |
| -------- | ------- | ----- | --- | ---- |
| `--help` | No | Prints all the help information. | - | - |
| `--fmk=<FMK>`  | Yes | Original format of the input model. | MINDIR, CAFFE, TFLITE, TF, or ONNX | - |
| `--modelFile=<MODELFILE>` | Yes | Path of the input model. | - | - |
| `--outputFile=<OUTPUTFILE>` | Yes | Path of the output model. The suffix `.ms` can be automatically generated. | - | - |
| `--weightFile=<WEIGHTFILE>` | Yes (for Caffe models only) | Path of the weight file of the input model. | - | - |
| `--quantType=<QUANTTYPE>` | No | Sets the quantization type of the model. | PostTraining: quantization after training <br>WeightQuant: only do weight quantization after training | - |
| `--bitNum=<BITNUM>` | No | Sets the quantization bitNum when quantType is set as WeightQuant, now supports 1 bit to 16 bit quantization. | \[1, 16] | 8 |
| `--quantWeightSize=<QUANTWEIGHTSIZE>` | No | Sets a size threshold of convolution filter when quantType is set as WeightQuant. If the size is bigger than this value, it will trigger weight quantization. | \[0, +∞) | 0 |
| `--quantWeightChannel=<QUANTWEIGHTCHANNEL>` | No | Sets a channel number threshold of convolution filter when quantType is set as WeightQuant. If the number is bigger than this, it will trigger weight quantization. | \[0, +∞) | 16 |
| `--configFile=<CONFIGFILE>` | No | 1) Profile path of calibration dataset when quantType is set as PostTraining; 2) Profile path of converter. | - | - |
| `--fp16=<FP16>` | No | Serialize const tensor in Float16 data type, only effective for const tensor in Float32 data type. | on or off | off |
| `--inputShape=<INPUTSHAPE>` | No | Set the dimension of the model input, the default is the same as the input of the original model. The model can be further optimized in some scenarios, such as models with shape operator, but the output model will lose the feature of dymatic shape. e.g. inTensorName: 1,32,32,4 | - | - |

> - The parameter name and parameter value are separated by an equal sign (=) and no space is allowed between them.
> - The Caffe model is divided into two files: model structure `*.prototxt`, corresponding to the `--modelFile` parameter; model weight `*.caffemodel`, corresponding to the `--weightFile` parameter.
> - In order to ensure the accuracy of weight quantization, the "--bitNum" parameter should better be set to a range from 8bit to 16bit.
> - PostTraining method currently only supports activation quantization and weight quantization in 8 bit.
> - The priority of `--fp16` option is very low. For example, if quantization is enabled, `--fp16` will no longer take effect on const tensors that have been quantized. All in all, this option only takes effect on const tensors of Float32 when serializing model.

The calibration dataset configuration file uses the `key=value` mode to define related parameters. The `key` to be configured is as follows:

| Parameter Name | Attribute | Function Description | Parameter Type | Default Value | Value Range |
| -------- | ------- | -----          | -----    | -----     |  ----- |
| image_path | Mandatory for full quantization | Directory for storing a calibration dataset. If a model has multiple inputs, enter directories where the corresponding data is stored in sequence. Use commas (,) to separate them. | String | - | The directory stores the input data that can be directly used for inference. Since the current framework does not support data preprocessing, all data must be converted in advance to meet the input requirements of inference. |
| batch_count | Optional | Number of used inputs | Integer | 100 | (0, +∞) |
| method_x | Optional | Input and output data quantization algorithms at the network layer | String  |  KL  | KL, MAX_MIN, or RemovalOutlier.  <br> KL: quantizes and calibrates the data range based on [KL divergence](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf).  <br> MAX_MIN: data quantization parameter computed based on the maximum and minimum values.  <br> RemovalOutlier: removes the maximum and minimum values of data based on a certain proportion and then calculates the quantization parameters.  <br> If the calibration dataset is consistent with the input data during actual inference, MAX_MIN is recommended. If the noise of the calibration dataset is large, KL or RemovalOutlier is recommended.
| thread_num | Optional | Number of threads used when the calibration dataset is used to execute the inference process | Integer | 1 | (0, +∞) |
| bias_correction | Optional | Indicate whether to correct the quantization error. | Boolean | false | True or false. After this parameter is enabled, the accuracy of the converted model can be improved. You are advised to set this parameter to true. |
| plugin_path | Optional | Third-party library path | String | - | If there are more than one, please use `;` to separate. |
| disable_fusion | Optional | Indicate whether to correct the quantization error | String | off | off or on. |

### Example

The following describes how to use the conversion command by using several common examples.

- Take the Caffe model LeNet as an example. Run the following conversion command:

   ```bash
   ./converter_lite --fmk=CAFFE --modelFile=lenet.prototxt --weightFile=lenet.caffemodel --outputFile=lenet
   ```

   In this example, the Caffe model is used. Therefore, the model structure and model weight files are required. Two more parameters `fmk` and `outputFile` are also required.

   The output is as follows:

   ```text
   CONVERTER RESULT SUCCESS:0
   ```

   This indicates that the Caffe model is successfully converted into the MindSpore Lite model and the new file `lenet.ms` is generated.

- The following uses the MindSpore, TensorFlow Lite, TensorFlow and ONNX models as examples to describe how to run the conversion command.

    - MindSpore model `model.mindir`

      ```bash
      ./converter_lite --fmk=MINDIR --modelFile=model.mindir --outputFile=model
      ```

     > The `MindIR` model exported by MindSpore v1.1.1 or earlier is recommended to be converted to the `ms` model using the converter tool of the corresponding version. MindSpore v1.1.1 and later versions, the converter tool will be forward compatible.

    - TensorFlow Lite model `model.tflite`

      ```bash
      ./converter_lite --fmk=TFLITE --modelFile=model.tflite --outputFile=model
      ```

    - TensorFlow model `model.pb`

      ```bash
      ./converter_lite --fmk=TF --modelFile=model.pb --outputFile=model
      ```

    - ONNX model `model.onnx`

      ```bash
      ./converter_lite --fmk=ONNX --modelFile=model.onnx --outputFile=model
      ```

   In the preceding scenarios, the following information is displayed, indicating that the conversion is successful. In addition, the target file `model.ms` is obtained.

   ```text
   CONVERTER RESULT SUCCESS:0
   ```

- If running the conversion command is failed, an [errorcode](https://www.mindspore.cn/lite/api/en/r1.3/api_cpp/errorcode_and_metatype.html) will be output.

## Windows Environment Instructions

### Environment Preparation  

To use the MindSpore Lite model conversion tool, the following environment preparations are required.

- [Compile](https://www.mindspore.cn/lite/docs/en/r1.3/use/build.html) or [download](https://www.mindspore.cn/lite/docs/en/r1.3/use/downloads.html) model transfer tool.

- Add the path of dynamic library required by the conversion tool to the environment variables PATH.

    ```bash
    set PATH=%PACKAGE_ROOT_PATH%\tools\converter\lib;%PATH%
    ````

    %PACKAGE_ROOT_PATH% is the decompressed package path obtained by compiling or downloading.

### Directory Structure

```text
mindspore-lite-{version}-win-x64
└── tools
    └── converter # Model conversion tool
        ├── include
        ├── converter
        │   └── converter_lite.exe    # Executable program
        └── lib
            ├── libgcc_s_seh-1.dll    # Dynamic library of MinGW
            ├── libglog.dll           # Dynamic library of Glog
            ├── libssp-0.dll          # Dynamic library of MinGW
            ├── libstdc++-6.dll       # Dynamic library of MinGW
            └── libwinpthread-1.dll   # Dynamic library of MinGW
```

### Parameter Description

Refer to the Linux environment model conversion tool [parameter description](https://www.mindspore.cn/lite/docs/en/r1.3/use/converter_tool.html#parameter-description).

### Example

Set the log printing level to INFO.

```bat
set GLOG_v=1
```

> Log level: 0 is DEBUG, 1 is INFO, 2 is WARNING, 3 is ERROR.

Several common examples are selected below to illustrate the use of conversion commands.

- Take the Caffe model LeNet as an example to execute the conversion command.

   ```bat
   call converter_lite --fmk=CAFFE --modelFile=lenet.prototxt --weightFile=lenet.caffemodel --outputFile=lenet
   ```

   In this example, because the Caffe model is used, two input files of model structure and model weight are required. Then with the fmk type and output path two parameters which are required, you can successfully execute.

   The result is shown as:

   ```text
   CONVERTER RESULT SUCCESS:0
   ```

   This means that the Caffe model has been successfully converted to the MindSpore Lite model and the new file `lenet.ms` has been obtained.

- Take MindSpore, TensorFlow Lite, ONNX model format and perceptual quantization model as examples to execute conversion commands.

    - MindSpore model `model.mindir`

      ```bat
      call converter_lite --fmk=MINDIR --modelFile=model.mindir --outputFile=model
      ```

     > The `MindIR` model exported by MindSpore v1.1.1 or earlier is recommended to be converted to the `ms` model using the converter tool of the corresponding version. MindSpore v1.1.1 and later versions, the converter tool will be forward compatible.

    - TensorFlow Lite model`model.tflite`

      ```bat
      call converter_lite --fmk=TFLITE --modelFile=model.tflite --outputFile=model
      ```

    - TensorFlow model `model.pb`

      ```bat
      call converter_lite --fmk=TF --modelFile=model.pb --outputFile=model
      ```

    - ONNX model`model.onnx`

      ```bat
      call converter_lite --fmk=ONNX --modelFile=model.onnx --outputFile=model
      ```

   In the above cases, the following conversion success prompt is displayed, and the `model.ms` target file is obtained at the same time.

   ```text
   CONVERTER RESULT SUCCESS:0
   ```

- If running the conversion command is failed, an [errorcode](https://www.mindspore.cn/lite/api/en/r1.3/api_cpp/errorcode_and_metatype.html) will be output.
