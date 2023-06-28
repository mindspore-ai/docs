# Converter Tool

<a href="https://gitee.com/mindspore/docs/blob/r0.7/lite/tutorials/source_en/use/converter_tool.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

MindSpore Lite provides a tool for offline model conversion. It supports conversion of multiple types of models. The converted models can be used for inference. The command line parameters contain multiple personalized options, providing a convenient conversion method for users.

Currently, the following input formats are supported: MindSpore, TensorFlow Lite, Caffe, and ONNX.

## Linux Environment Instructions

### Environment Preparation

To use the MindSpore Lite model conversion tool, you need to prepare the environment as follows:

- Compilation: Install basic and additional build dependencies and perform build. The build version is x86_64. The code of the model conversion tool is stored in the `mindspore/lite/tools/converter` directory of the MindSpore source code. For details about the build operations, see the [Environment Requirements](https://www.mindspore.cn/lite/tutorial/en/r0.7/build.html#environment-requirements) and [Compilation Example](https://www.mindspore.cn/lite/tutorial/en/r0.7/build.html#compilation-example) in the build document.

- Run: Obtain the `converter` tool and configure environment variables by referring to [Output Description](https://www.mindspore.cn/lite/tutorial/en/r0.7/build.html#output-description) in the build document.

### Parameter Description

You can use `./converter_lite ` to complete the conversion. In addition, you can set multiple parameters as required.
You can enter `./converter_lite --help` to obtain help information in real time.

The following describes the parameters in detail.


| Parameter  |  Mandatory or Not   |  Parameter Description  | Value Range | Default Value |
| -------- | ------- | ----- | --- | ---- |
| `--help` | No | Prints all help information. | - | - |
| `--fmk=<FMK>`  | Yes | Original format of the input model. | MS, CAFFE, TFLITE, or ONNX | - |
| `--modelFile=<MODELFILE>` | Yes | Path of the input model. | - | - |
| `--outputFile=<OUTPUTFILE>` | Yes | Path of the output model. (If the path does not exist, a directory will be automatically created.) The suffix `.ms` can be automatically generated. | - | - |
| `--weightFile=<WEIGHTFILE>` | Yes (for Caffe models only) | Path of the weight file of the input model. | - | - |
| `--quantType=<QUANTTYPE>` | No | Sets the quant type of the model. | PostTraining: quantization after training <br>AwareTraining: perceptual quantization | - |
|`--inputInferenceType=<INPUTRINFERENCETYPE>`  | No(supported by aware quant models only) | Sets the input data type of the converted model. If the type is different from the origin model, the convert tool will insert data type convert op before the model to make sure the input data type is same as the input of origin model. | FLOAT or INT8 | FLOAT |
|`--inferenceType=<INFERENCETYPE> `| No(supported by aware quant models only) | Sets the output data type of the converted model. If the type is different from the origin model, the convert tool will insert data type convert op before the model to make sure the output data type is same as the input of origin model. | FLOAT or INT8 | FLOAT |
|`--stdDev=<STDDEV>`| No(supported by aware quant models only) | Sets the standard deviation of the input data. | （0，+∞） | 128 |
|`--mean=<MEAN>`| No(supported by aware quant models only) | Sets the mean value of the input data. | [-128, 127] | -0.5 |

> - The parameter name and parameter value are separated by an equal sign (=) and no space is allowed between them.
> - The Caffe model is divided into two files: model structure `*.prototxt`, corresponding to the `--modelFile` parameter; model weight `*.caffemodel`, corresponding to the `--weightFile` parameter


### Example

First, in the root directory of the source code, run the following command to perform compilation. For details, see `build.md`.
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
   ```
   INFO [converter/converter.cc:190] Runconverter] CONVERTER RESULT: SUCCESS!
   ```
   This indicates that the Caffe model is successfully converted into the MindSpore Lite model and the new file `lenet.ms` is generated.
   
- The following uses the MindSpore, TensorFlow Lite, ONNX and perception quantization models as examples to describe how to run the conversion command.

   - MindSpore model `model.mindir`
      ```bash
      ./converter_lite --fmk=MS --modelFile=model.mindir --outputFile=model
      ```
   
   - TensorFlow Lite model `model.tflite`
      ```bash
      ./converter_lite --fmk=TFLITE --modelFile=model.tflite --outputFile=model
      ```
   
   - ONNX model `model.onnx`
      ```bash
      ./converter_lite --fmk=ONNX --modelFile=model.onnx --outputFile=model
      ```

   - TensorFlow Lite aware quantization model `model_quant.tflite`
      ```bash
      ./converter_lite --fmk=TFLITE --modelFile=model.tflite --outputFile=model --quantType=AwareTraining
      ```
     
  - TensorFlow Lite aware quantization model `model_quant.tflite` set the input and output data type to be int8
      ```bash
      ./converter_lite --fmk=TFLITE --modelFile=model.tflite --outputFile=model --quantType=AwareTraining  --inputInferenceType=INT8  --inferenceType=INT8
      ```

   In the preceding scenarios, the following information is displayed, indicating that the conversion is successful. In addition, the target file `model.ms` is obtained.
   ```
   INFO [converter/converter.cc:190] Runconverter] CONVERTER RESULT: SUCCESS!
   ```
