# Converting Models for Inference

`Windows` `Linux` `Model Converting` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r1.1/tutorials/lite/source_en/use/converter_tool.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

MindSpore Lite provides a tool for offline model conversion. It supports conversion of multiple types of models. The converted models can be used for inference. The command line parameters contain multiple personalized options, providing a convenient conversion method for users.

Currently, the following input formats are supported: MindSpore, TensorFlow Lite, Caffe, and ONNX.

## Linux Environment Instructions

### Environment Preparation

To use the MindSpore Lite model conversion tool, you need to prepare the environment as follows:

- Compilation: Install basic and additional build dependencies and perform build. The build version is x86_64. The code of the model conversion tool is stored in the `mindspore/lite/tools/converter` directory of the MindSpore source code. For details about the build operations, see the [Environment Requirements](https://www.mindspore.cn/tutorial/lite/en/r1.1/use/build.html#environment-requirements) and [Compilation Example](https://www.mindspore.cn/tutorial/lite/en/r1.1/use/build.html#compilation-example) in the build document.

- Run: Obtain the `converter` tool and configure environment variables by referring to [Output Description](https://www.mindspore.cn/tutorial/lite/en/r1.1/use/build.html#output-description) in the build document.

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
| `--configFile=<CONFIGFILE>` | No | Profile path of calibration dataset when quantType is set as PostTraining. | - | - |

> - The parameter name and parameter value are separated by an equal sign (=) and no space is allowed between them.
> - The Caffe model is divided into two files: model structure `*.prototxt`, corresponding to the `--modelFile` parameter; model weight `*.caffemodel`, corresponding to the `--weightFile` parameter.
> - In order to ensure the accuracy of weight quantization, the "--bitNum" parameter should better be set to a range from 8bit to 16bit.
> - PostTraining method currently only supports activation quantization and weight quantization in 8 bit.
> - Currently, TensorFlow converter is in the Beta stage, the range of supported parsers for TensorFlow is relatively limited.

### Example

First, in the root directory of the source code, run the following command to perform compilation.

```bash
bash build.sh -I x86_64
```

> Currently, the model conversion tool supports only the x86_64 architecture.

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

- The following uses the MindSpore, TensorFlow Lite, ONNX models as examples to describe how to run the conversion command.

    - MindSpore model `model.mindir`

      ```bash
      ./converter_lite --fmk=MINDIR --modelFile=model.mindir --outputFile=model
      ```

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

- If running the conversion command is failed, an [errorcode](https://www.mindspore.cn/doc/api_cpp/en/r1.1/errorcode_and_metatype.html) will be output.

## Windows Environment Instructions

### Environment Preparation  

To use the MindSpore Lite model conversion tool, the following environment preparations are required.

- Get the toolkit: To obtain the 'Converter' tool, download the [zip package](https://www.mindspore.cn/tutorial/lite/en/r1.1/use/downloads.html) of windows conversion tool and unzip it to the local directory.

### Parameter Description

Refer to the Linux environment model conversion tool [parameter description](https://www.mindspore.cn/tutorial/lite/en/r1.1/use/converter_tool.html#parameter-description).

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

- If running the conversion command is failed, an [errorcode](https://www.mindspore.cn/doc/api_cpp/en/r1.1/errorcode_and_metatype.html) will be output.
