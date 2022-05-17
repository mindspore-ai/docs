# Usage Description of the Integrated NNIE

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/use/nnie.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Directory Structures

### The converter Directory Structure of the Model Conversion Tool

```text
mindspore-lite-{version}-runtime-linux-x64
└── tools
    └── converter
        └── providers
            └── Hi3516D                # Embedded board model number
                ├── libmslite_nnie_converter.so        # Dynamic library for converting the integrated NNIE
                ├── libmslite_nnie_data_process.so     # Dynamic library for processing NNIE input data
                ├── libnnie_mapper.so        # Dynamic library for building NNIE binary files
                └── third_party       # Third-party dynamic library on which the NNIE depends
                    ├── opencv-4.2.0
                    │   └── libopencv_xxx.so
                    └── protobuf-3.9.0
                        ├── libprotobuf.so
                        └── libprotoc.so
```

The preceding shows the integration directory structure of the NNIE. For details about other directory structures of the converter, see [Converting Models for Inference](https://www.mindspore.cn/lite/docs/en/master/use/converter_tool.html).

### The runtime Directory Structure of the Model Inference Tool

```text
mindspore-lite-{version}-linux-aarch32
└── providers
    └── Hi3516D        # Embedded board model number
        └── libmslite_nnie.so  # Dynamic library of the integrated NNIE
        └── libmslite_proposal.so  # Sample dynamic library of the integrated proposal
```

The preceding shows the integration directory structure of the NNIE. For details about other directory structures of the inference tool runtime, see [Directory Structure](https://www.mindspore.cn/lite/docs/en/master/use/build.html#directory-structure).

## Using Tools

### Converter

#### Overview

MindSpore Lite provides a tool for offline model conversion. It can convert models of multiple types (only Caffe is supported currently) into board-dedicated models that support NNIE hardware acceleration inference and can run on the Hi3516 board.
The converted NNIE `ms` model can be used only on the associated embedded board. The runtime inference framework matching the conversion tool can be used to perform inference. For more information about the conversion tool, see [Converting Models for Inference](https://www.mindspore.cn/lite/docs/en/master/use/converter_tool.html).

#### Environment Preparation

To use the MindSpore Lite model conversion tool, you need to prepare the environment as follows:

1. [Download](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html) NNIE-dedicated converter. Currently, only Linux is supported.

2. Decompress the downloaded package.

     ```bash
     tar -zxvf mindspore-lite-{version}-linux-x64.tar.gz
     ```

     {version} indicates the version number of the release package.

3. Add the dynamic link library required by the conversion tool to the environment variable LD_LIBRARY_PATH.

    ```bash
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PACKAGE_ROOT_PATH}/tools/converter/lib:${PACKAGE_ROOT_PATH}/runtime/lib:${PACKAGE_ROOT_PATH}/tools/converter/providers/Hi3516D/third_party/opencv-4.2.0:${PACKAGE_ROOT_PATH}/tools/converter/providers/Hi3516D/third_party/protobuf-3.9.0
    ```

    ${PACKAGE_ROOT_PATH} indicates the path of the folder obtained after the decompression.

#### Extension Configuration

To load the extension module when converting, users need to configure the path of extended dynamic library. The parameters related to the extension include `plugin_path`, `disable_fusion`. The detailed description of the parameters is as follows:

| Parameter | Attribute | Function Description | Parameter Type | Default Value | Value Range |
| --------- | --------- | -------------------- | -------------- | ------------- | ----------- |
| plugin_path | Optional | Third-party library path | String | - | If there are more than one, please use `;` to separate. |
| disable_fusion | Optional | Indicate whether to correct the quantization error | String | off | off or on. |

We have generated the default configuration file which restores the relative path of the NNIE dynamic library for the users in the released package. The users need to decide whether the configuration file needs to be modified manually. The content is as follows:

```ini
[registry]
plugin_path=../providers/Hi3516D/libmslite_nnie_converter.so
```

#### NNIE Configuration

The NNIE model can use the NNIE hardware to accelerate the model running. To do so, the users also need to prepare NNIE's own configuration file. Users can configure the configuration file required by MindSpore Lite by referring to the `description of configuration items for nnie_mapper` in the HiSVP Development Guide provided by HiSilicon. `nnie.cfg` indicates the configuration file.

The following is an example of the `nnie.cfg` file:

```text
[net_type] 0
[image_list] ./input_nchw.txt
[image_type] 0
[norm_type] 0
[mean_file] null
```

> `input_nchw.txt` is the input data of the floating-point text format of the Caffe model to be converted. For details, see the description of `image_list` in the HiSVP Development Guide. In the configuration file, you can configure theitems other than caffemodel_file, prototxt_file, is_simulation and instructions_name.

#### Executing Converter

1. Go to the converter directory.

    ```bash
    cd ${PACKAGE_ROOT_PATH}/tools/converter/converter
    ```

2. (Optional) Set environment variables.

    Skip this step if you have entered the converter directory by Step 1, and the default value takes effect. If you have not entered the converter directory, you need to declare the path of the .so files and benchmark binary programs on which the conversion tool depends in the environment variables, as shown in the following figure:

    ```bash
    export NNIE_MAPPER_PATH=${PACKAGE_ROOT_PATH}/tools/converter/providers/Hi3516D/libnnie_mapper.so
    export NNIE_DATA_PROCESS_PATH=${PACKAGE_ROOT_PATH}/tools/converter/providers/Hi3516D/libmslite_nnie_data_process.so
    export BENCHMARK_PATH=${PACKAGE_ROOT_PATH}/tools/benchmark
    ```

    ${PACKAGE_ROOT_PATH} indicates the path of the decompressed package.

3. Copy the `nnie.cfg` file to the converter directory and set the following environment variable:

    ```bash
    export NNIE_CONFIG_PATH=./nnie.cfg
    ```

   Skip this step if the actual configuration file is `nnie.cfg` and is at the same level as the `converter_lite` file.

4. Execute the Converter to generate an NNIE `ms` model.

    ```bash
    ./converter_lite --fmk=CAFFE --modelFile=${model_name}.prototxt --weightFile=${model_name}.caffemodel --configFile=./converter.cfg --outputFile=${model_name}
    ```

    ${model_name} indicates the model file name. The execution result is as follows:

     ```text
     CONVERTER RESULT SUCCESS:0
     ```

     For details about the parameters of the converter_lite conversion tool, see ["Parameter Description" in Converting Models for Inference](https://www.mindspore.cn/lite/docs/en/master/use/converter_tool.html#example).

### Runtime

#### Overview

After the converted model is obtained, you can perform inference on the associated embedded board by using the Runtime inference framework matching the board. MindSpore Lite provides a benchmark test tool, which can be used to perform quantitative analysis (performance) on the execution time consumed by forward inference of the MindSpore Lite model. In addition, you can perform comparative error analysis (accuracy) based on the output of a specified model.
For details about the inference tool, see [benchmark](https://www.mindspore.cn/lite/docs/en/master/use/benchmark_tool.html).

#### Environment Preparation

You can perform equivalent operations based on the actual situation. See the following example:

1. [Download](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html) NNIE-dedicated model inference tool. Currently, only Hi3516D is supported.

2. Decompress the downloaded package.

     ```bash
     tar -zxvf mindspore-lite-{version}-linux-aarch32.tar.gz
     ```

     {version} indicates the version number of the release package.

3. Create a storage directory on the Hi3516D board.

   Log in to the board and create a working directory.

   ```bash
   mkdir /user/mindspore          # Stores benchmark execution files and models.
   mkdir /user/mindspore/lib      # Stores dependent library files.
   ```

4. Transfer files.

   Transfer the benchmark tool, model, and .so library to the Hi3516D board. `libmslite_proposal.so` is an implementation sample .so file of the proposal operator provided by MindSpore Lite. If the user model contains a custom proposal operator, you need to generate the `libnnie_proposal.so` file by referring to the [Proposal Operator Usage Description](#proposal-operator-usage-description) to replace the .so file for correct inference.

   ```bash
   scp libmindspore-lite.so libmslite_nnie.so libmslite_proposal.so root@${device_ip}:/user/mindspore/lib
   scp benchmark ${model_path} root@${device_ip}:/user/mindspore
   ```

   ${model_path} indicates the path of the MS model file after conversion.

5. Set the dynamic library path.

   NNIE model inference depends on the NNIE-related board dynamic libraries provided by HiSilicon, including libnnie.so, libmpi.so, libVoiceEngine.so, libupvqe.so and libdnvqe.so.

   You need to save these .so files on the board and pass the path to the LD_LIBRARY_PATH environment variable.
   In the example, the .so files are stored in /usr/lib. You need to configure the .so files according to the actual situation.

   ```bash
   export LD_LIBRARY_PATH=/user/mindspore/lib:/usr/lib:${LD_LIBRARY_PATH}
   ```

6. (Optional) Set configuration items.

   If the user model contains the proposal operator, configure the MAX_ROI_NUM environment variable based on the implementation of the proposal operator.

   ```bash
   export MAX_ROI_NUM=300    # Maximum number of ROIs supported by a single image. The value is a positive integer. The default value is 300.
   ```

   If the user model is a loop or LSTM network, you need to configure the TIME_STEP environment variable based on the actual network running status. For details about other requirements, see [Multi-image Batch Running and Multi-step Running](#multi-image-batch-running-and-multi-step-running).

   ```bash
   export TIME_STEP=1        # Number of steps for loop or LSTM network running. The value is a positive integer. The default value is 1.
   ```

   If there are multiple NNIE hardware devices on the board, you can specify the NNIE device on which the model runs by using the CORE_IDS environment variable.
   If the model is segmented (you can open the model using the Netron to observe the segmentation status), you can configure the device on which each segment runs in sequence. The segments that are not configured run on the last configured NNIE device.

   ```bash
   export CORE_IDS=0         # Kernel ID for NNIE running. Model segments can be configured independently and are separated by commas (,), for example, export CORE_IDS=1,1. The default value is 0.
   ```

7. (Optional) Build image input.

   If the calibration set sent by the Converter to the mapper is an image when the model is exported, the input data transferred to the benchmark must be of the int8 type. That is, the image must be converted into the int8 type before being transferred to the benchmark.
   Python is used to provide a conversion example.

   ``` python
   import sys
   import cv2

   def usage():
       print("usage:\n"
             "example: python generate_input_bin.py xxx.img BGR 224 224\n"
             "argv[1]: origin image path\n"
             "argv[2]: RGB_order[BGR, RGB], should be same as nnie mapper config file's [RGB_order], default is BGR\n"
             "argv[3]: input_h\n"
             "argv[4]: input_w"
             )

   def main(argvs):
       if argvs[1] == "-h":
           usage()
           print("EXIT")
           exit()
       img_path = argvs[1]
       rgb_order = argvs[2]
       input_h = int(argvs[3])
       input_w = int(argvs[4])
       img = cv2.imread(img_path)
       if rgb_order == "RGB":
           img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       img_hwc = cv2.resize(img, (input_w, input_h))
       outfile_name = "1_%s_%s_3_nhwc.bin" %(argvs[3], argvs[4])
       img_hwc.tofile(outfile_name)
       print("Generated " + outfile_name + " file success in current dir.")

   if __name__ == "__main__":
       if len(sys.argv) == 1:
           usage()
           print("EXIT")
           exit()
       elif len(sys.argv) != 5:
           print("Input argument is invalid.")
           usage()
           print("EXIT")
           exit()
       else:
           main(sys.argv)
   ```

#### Executing the Benchmark

```text
cd /user/mindspore
./benchmark --modelFile=${model_path}
```

`model_path` indicates the path of the MS model file after conversion.

After this command is executed, random input of the model is generated and forward inference is performed. For details about how to use the benchmark, such as time consumption analysis and inference error analysis, see [benchmark](https://www.mindspore.cn/lite/docs/en/master/use/benchmark_tool.html).

For details about the input data format requirements of the model, see [(Optional) SVP Tool Chain-related Functions and Precautions](#svp-tool-chain-related-functions-and-precautions-advanced-options).

## Integration

For details about integration, see [Using C++ Interface to Perform Inference](https://www.mindspore.cn/lite/docs/en/master/use/runtime_cpp.html).

## SVP Tool Chain-related Functions and Precautions (Advanced Options)

During model conversion, the `nnie.cfg` file declared by the NNIE_CONFIG_PATH environment variable provides functions related to the SVP tool chain and supports the configuration of fields except caffemodel_file, prototxt_file, is_simulation and instructions_name. The implementation is as follows:

### NHWC, the Format of the Running Input on the Board

  The `ms` model after conversion accepts only the data input in NHWC format. If image_type is declared as 0, float32 data in NHWC format is received. If image_type is declared as 1, uint8 data input in NHWC format is received.

### image_list Description

  The meaning of the image_list field in the `nnie.cfg` file remains unchanged. When image_type is declared as 0, data in the CHW format is provided by row, regardless of whether the original model is the NCHW input.

### image_type Restrictions

  MindSpore Lite does not support the network input when image_type is set to 3 or 5. You can set it to 0 or 1.

### image_list and roi_coordinate_file Quantity

  You only need to provide image_list whose quantity is the same as that of model inputs. If the model contains the ROI pooling or PSROI pooling layer, you need to provide roi_coordinate_file, the quantity and sequence correspond to the number and sequence of the ROI pooling or PSROI pooling layer in the .prototxt file.

### Suffix _cpu of the Node Name in the prototxt File

  In the .prototxt file, you can add _cpu to the end of the node name to declare CPU custom operator. The_cpu suffix is ignored in MindSpore Lite and is not supported. If you want to redefine the implementation of an existing operator or add an operator, you can register the operator in custom operator mode.

### Custom Operator in the prototxt File

  In the SVP tool chain, the custom layer is declared in the .prototxt file to implement inference by segment and implement the CPU code by users. In MindSpore Lite, you need to add the op_type attribute to the custom layer and register the online inference code in custom operator mode.

  An example of modifying the custom layer is as follows:

  ```text
  layer {
    name: "custom1"
    type: "Custom"
    bottom: "conv1"
    top: "custom1_1"
    custom_param {
      type: "MY_CUSTOM"
      shape {
          dim: 1
          dim: 256
          dim: 64
          dim: 64
      }
  }
  }
  ```

  In this example, a custom operator of the MY_CUSTOM type is defined. During inference, you need to register a custom operator of the MY_CUSTOM type.

### Suffix _report of the Top Domain in the prototxt File

  When converting the NNIE model, MindSpore Lite fuses most operators into the binary file for NNIE running. Users cannot view the output of the intermediate operators. In this case, you can add the _report suffix to the top domain, during image composition conversion, the output of the intermediate operator is added to the output of the fused layer. If the operator has output (not fused), the output remains unchanged.

  During the inference running, you can obtain the output of the intermediate operator by referring to [Using C++ Interface to Perform Inference](https://www.mindspore.cn/lite/docs/en/master/use/runtime_cpp.html#using-c++-interface-to-perform-inference).

  MindSpore Lite parses the corresponding rules of _report and resolves the conflict with the [Inplace Mechanism](#inplace-mechanism). For details, see the definition in the HiSVP Development Guide.

### Inplace Mechanism

  The inplace layer can be used to run the chip in efficient mode. By default, the conversion tool rewrites all layers in the .prototxt file that support the inplace layer. To disable this function, you can declare the following environment:

  ```bash
  export NNIE_DISABLE_INPLACE_FUSION=off         # When this parameter is set to on or is not set, inplace automatic rewriting is enabled.
  ```

  When the automatic rewriting function is disabled, you can manually rewrite the corresponding layer in the .prototxt file to enable the efficient mode for some layers.

### Multi-image Batch Running and Multi-step Running

  If you need to infer multiple input data (multiple images) at the same time, you can resize the first dimension of the model input to the quantity of input data by referring to [Resizing the Input Dimension](https://www.mindspore.cn/lite/docs/en/master/use/runtime_cpp.html#resizing-the-input-dimension). In the NNIE model, only the first dimension ('n' dimension) can be resized, and other dimensions ('hwc') cannot be resized.

  For the loop or LSTM network, you need to configure the TIME_STEP environment variable based on the step value and resize the model input.
  Assume that the number of data records for forward inference at a time is `input_num`. The resize value of the input node of sequence data is `input_num x step`, and the resize value of the input node of non-sequence data is `input_num`.

  Models with the proposal operator do not support batch running or the resizie operation.

### Node Name Change

  After a model is converted into an NNIE model, the name of each node may change. You can use the Netron to open the model and obtain the new node name.

### Proposal Operator Usage Description

  MindSpore Lite provides the sample code of the proposal operator. In this sample, the proposal operator and its infer shape are registered in custom operator mode. You can change it to the implementation that matches your own model, and then perform [integration](#integration).
  > Download address of the complete sample code:
  >
  > <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/nnie_proposal>

### Segmentation Mechanism and Restrictions

  Due to the restrictions on the operators supported by the NNIE chip, if there are operators that are not supported by the NNIE chip, the model needs to be divided into supported layers and unsupported layers.
  The chip on the board supports a maximum of eight supported layers. If the number of supported layers after segmentation is greater than 8, the model cannot run. You can observe the custom operator (whose attribute contains type:NNIE) by using Netron to obtain the number of supported layers after conversion.