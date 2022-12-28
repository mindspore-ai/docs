# Offline Conversion of Inference Models

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/use/cloud_infer/converter_tool.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore Lite cloud-side inference provides tools for offline conversion of models, and supports many types of model conversions, and the converted models can be used for inference. Command line parameters include a variety of personalized options to provide users with convenient conversion paths.

The currently supported input formats are MindSpore, TensorFlow Lite, Caffe, TensorFlow, and ONNX.

The `mindir` model converted by the converter supports the converter companion and higher versions of the Runtime inference framework to perform inference.

## Linux Environment Usage Instructions

### Environment Preparation

To use MindSpore Lite cloud-side inference model converter, the following environment preparation is required.

- [Compile](https://www.mindspore.cn/lite/docs/en/master/use/cloud_infer/build.html) or [download](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html) model converter.
- Add the dynamic link libraries required by the converter to the environment variable LD_LIBRARY_PATH.

    ```bash
    export LD_LIBRARY_PATH=${PACKAGE_ROOT_PATH}/tools/converter/lib:${LD_LIBRARY_PATH}
    ```

    ${PACKAGE_ROOT_PATH} is the path of the compiled or downloaded package after unpacking.

### Directory Structure

```text
mindspore-lite-{version}-linux-x64
└── tools
    └── converter
        ├── include                            # Custom operators, model parsing, node parsing, conversion optimization registration headers
        ├── converter                          # Model converter
        │   └── converter_lite                 # Executable programs
        └── lib                                # Dynamic libraries that the converter depends on
            ├── libmindspore_glog.so.0         # Glog dynamic libraries
            ├── libascend_pass_plugin.so       # Register for Ascend Backend Graph Optimization Plugin Dynamic Library
            ├── libmindspore_shared_lib.so     # Adaptation of the dynamic library in the backend of Ascend
            ├── libmindspore_converter.so      # Dynamic library for model conversion
            ├── libmslite_converter_plugin.so  # Model conversion plugin
            ├── libmindspore_core.so           # MindSpore Core dynamic libraries
            ├── libopencv_core.so.4.5          # Dynamic libraries for OpenCV
            ├── libopencv_imgcodecs.so.4.5     # Dynamic libraries for OpenCV
            └── libopencv_imgproc.so.4.5       # Dynamic libraries for OpenCV
        ├── third_party                        # Third party model proto definition
```

### Description of Parameters

MindSpore Lite cloud-side inference model converter provides various parameter settings that users can choose to use according to their needs. In addition, users can enter `. /converter_lite --help` for live help.

Detailed parameter descriptions are provided below.

| Parameters  |  Required or not   |  Description of parameters  | Value range | Default values | Remarks |
| -------- | ------- | ----- | --- | ---- | ---- |
| `--help` | Not | Print all help information. | - | - |
| `--fmk=<FMK>`  | Required | The original format of the input model. | MINDIR、CAFFE、TFLITE、TF、ONNX | - | - |
| `--modelFile=<MODELFILE>` | Required | The path of the input model. | - | - | - |
| `--outputFile=<OUTPUTFILE>` | Required | The path to the output model, without the suffix, can be automatically generated with the `.mindir` suffix. | - | - | - |
| `--weightFile=<WEIGHTFILE>` | Required when converting Caffe models | The path to the input model weight file. | - | - | - |
| `--configFile=<CONFIGFILE>` | Not | 1. can be used as a post-training quantization profile path; 2. can be used as an extended function profile path.  | - | - | - |
| `--inputShape=<INPUTSHAPE>` | Not | Set the dimensions of the model inputs, and keep the order of the input dimensions the same as the original model. The model structure can be further optimized for some specific models, but the converted model will probably lose the dynamic shape properties. Multiple inputs are split by `;`, along with double quotes `""`. | e.g.  "inTensorName_1: 1,32,32,4;inTensorName_2:1,64,64,4;" | - | - |
| `--device=<DEVICE>` | Not | Set specific hardware backend | Ascend310, Ascend310P | - | - |
| `--exportMindIR=<EXPORTMINDIR>` | Required | Set the exported model as `mindir` model or `ms` model. | MINDIR, MINDIR_LITE | MINDIR | This version can only be reasoned with models turned out by setting to MINDIR |
| `--NoFusion=<NOFUSION>` | Not | Set whether the process of converting the model is completed with correlation graph optimization. | true, false | ture | - |
| `--fp16=<FP16>` | Not | Set whether the weights in Float32 data format need to be stored in Float16 data format during model serialization. | on, off | off | Not supported at the moment|
| `--decryptKey=<DECRYPTKEY>` | Not | Set the key used to load the cipher text MindIR. The key is expressed in hexadecimal and is only valid when `fmk` is MINDIR. | - | - | Not supported at the moment |
| `--decryptMode=<DECRYPTMODE>` | Not | Set the mode to load the cipher MindIR, valid only when decryptKey is specified. | AES-GCM, AES-CBC | AES-GCM | Not supported at the moment |
| `--inputDataType=<INPUTDATATYPE>` | Not | Set the data type of the quantized model input tensor. Only if the quantization parameters (scale and zero point) of the model input tensor are available. The default is to keep the same data type as the original model input tensor. | FLOAT32, INT8, UINT8, DEFAULT | DEFAULT | Not supported at the moment |
| `--outputDataType=<OUTPUTDATATYPE>` | Not | Set the data type of the quantized model output tensor. Only if the quantization parameters (scale and zero point) of the model output tensor are available. The default is to keep the same data type as the original model output tensor. | FLOAT32, INT8, UINT8, DEFAULT | DEFAULT | Not supported at the moment |
| `--encryptKey=<ENCRYPTKEY>` | Not | Set the key to export the encryption `mindir` model. The key is expressed in hexadecimal. Only AES-GCM is supported, and the key length is only 16Byte. | - | - | Not supported at the moment |
| `--encryption=<ENCRYPTION>` | Not | Set whether to encrypt when exporting `mindir` models. Export encryption protects model integrity, but increases runtime initialization time. | true, false | true | Not supported at the moment |
| `--infer=<INFER>` | Not | Set whether to perform pre-inference when the conversion is completed. | true, false | false | Not supported at the moment |
| `--inputDataFormat=<INPUTDATAFORMAT>` | Not | Set the input format of the exported model, valid only for 4-dimensional inputs. | NHWC, NCHW | NHWC | Not supported at the moment |

Notes:

- The parameter name and the parameter value are connected by an equal sign without any space between them.
- Caffe models are generally divided into two files: `*.prototxt` model structure, corresponding to the `--modelFile` parameter, and `*.caffemodel` model weights, corresponding to the `--weightFile` parameter.
- The `configFile` configuration file uses the `key=value` approach to define the relevant parameters.
- `--NoFusion` parameter is used to set whether the graph optimization operation is completed during the offline conversion. If this parameter is set to true, no relevant graph optimization operations will be performed during the offline conversion phase of the model, and the relevant graph optimization operations will be done during the execution of the inference phase. The advantage of this parameter is that the converted model can be deployed directly to any CPU/GPU/Ascend hardware backend since it is not optimized in a specific way, while the disadvantage is that the initialization time of the model increases during inference execution.
- For the MindSpore model, since it is already a `mindir` model, two approaches are suggested:

    Inference is performed directly without offline conversion.

    Using offline conversion and setting --NoFusion to false. The relevant optimization is done in the offline phase to reduce the initialization time of inference execution.

### Usage Examples

The following selects common examples to illustrate the use of the conversion command.

- Take the Caffe model LeNet as an example and execute the conversion command.

    ```bash
    ./converter_lite --fmk=CAFFE --exportMindIR=MINDIR --modelFile=lenet.prototxt --weightFile=lenet.caffemodel --outputFile=lenet
    ```

    In this example, because the Caffe model is used, two input files, model structure and model weights, are required. Together with the other two required parameters, fmk type and output path, it can be executed successfully.

    The result is shown as:

    ```text
    CONVERT RESULT SUCCESS:0
    ```

    This indicates that the Caffe model has been successfully converted into a MindSpore Lite cloud-side inference model, obtaining the new file `lenet.mindir`.

- Take MindSpore, TensorFlow Lite, TensorFlow and ONNX models as examples and execute the conversion command.

    - MindSpore model `model.mindir`

    ```bash
    ./converter_lite --fmk=MINDIR --NoFusion=false --modelFile=model.mindir --outputFile=model
    ```

    - TensorFlow Lite model `model.tflite`

    ```bash
    ./converter_lite --fmk=TFLITE --exportMindIR=MINDIR --modelFile=model.tflite --outputFile=model
    ```

    - TensorFlow model `model.pb`

    ```bash
    ./converter_lite --fmk=TF --exportMindIR=MINDIR --modelFile=model.pb --outputFile=model
    ```

    - ONNX model `model.onnx`

    ```bash
    ./converter_lite --fmk=ONNX --exportMindIR=MINDIR --modelFile=model.onnx --outputFile=model
    ```

    In all of the above cases, the following conversion success message is displayed and the `model.mindir` target file is obtained at the same time.

    ```text
    CONVERTER RESULT SUCCESS:0
    ```
