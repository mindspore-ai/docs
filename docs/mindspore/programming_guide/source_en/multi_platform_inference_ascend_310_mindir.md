# Inference Using the MindIR Model on Ascend 310 AI Processors

`Linux` `Ascend` `Inference Application` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [Inference Using the MindIR Model on Ascend 310 AI Processors](#inference-using-the-mindir-model-on-ascend-310-ai-processors)
    - [Overview](#overview)
    - [Preparing the Development Environment](#preparing-the-development-environment)
    - [Exporting the MindIR Model](#exporting-the-mindir-model)
    - [Inference Directory Structure](#inference-directory-structure)
    - [Inference Code](#inference-code)
    - [Introduce to Building Script](#introduce-to-building-script)
    - [Building Inference Code](#building-inference-code)
    - [Performing Inference and Viewing the Result](#performing-inference-and-viewing-the-result)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/multi_platform_inference_ascend_310_mindir.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

Ascend 310 is a highly efficient and integrated AI processor oriented to edge scenarios. This tutorial describes how to use MindSpore to perform inference on the Ascend 310 based on the MindIR model file. The process is as follows:

1. Export the MindIR model file. The ResNet-50 model is used as an example.

2. Build the inference code to generate an executable `main` file.

3. Load the saved MindIR model, perform inference, and view the result.

> You can obtain the complete executable sample code at <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/ascend310_resnet50_preprocess_sample>.

## Preparing the Development Environment

Refer to [Installation Guide](https://www.mindspore.cn/install/en) to install Ascend environment and MindSpore.

## Exporting the MindIR Model

Train the target network on the CPU/GPU/Ascend 910 AI Processor, save it as a checkpoint file, and export the model file in MindIR format through the network and checkpoint file. For details about the export process, see [Export MindIR Model](https://www.mindspore.cn/docs/programming_guide/en/master/save_model.html#export-mindir-model).

> The [resnet50_imagenet.mindir](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/sample_resources/ascend310_resnet50_preprocess_sample/resnet50_imagenet.mindir) is a sample MindIR file exported using the ResNet-50 model, whose BatchSize is 1.

## Inference Directory Structure

Create a directory to store the inference code project, for example, `/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_resnet50_preprocess_sample`. The directory code can be obtained from the [official website](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/ascend310_resnet50_preprocess_sample). The `model` directory stores the exported `MindIR` model files and the `test_data` directory stores the images to be classified. The directory structure of the inference code project is as follows:

```text
└─ascend310_resnet50_preprocess_sample
    ├── CMakeLists.txt                    // Build script
    ├── README.md                         // Usage description
    ├── main.cc                           // Main function
    ├── main_hide_preprocess.cc           // Main function2，infer without defining preprocess(since defined in MindIR)
    ├── model
    │   └── resnet50_imagenet.mindir      // MindIR model file
    └── test_data
        ├── ILSVRC2012_val_00002138.JPEG  // Input sample image 1
        ├── ILSVRC2012_val_00003014.JPEG  // Input sample image 2
        ├── ...                           // Input sample image n
```

## Inference Code

### Infer model with defining preprocess manually: main.cc

#### Data-preprocessing by CPU operators

Inference sample code: <https://gitee.com/mindspore/docs/blob/master/docs/sample_code/ascend310_resnet50_preprocess_sample/main.cc> .

Using namespace of `mindspore` and `mindspore::dataset`.

```c++
namespace ms = mindspore;
namespace ds = mindspore::dataset;
```

Set global context, device target is `Ascend 310` and device id is `0`:

```c++
auto context = std::make_shared<ms::Context>();
auto ascend310_info = std::make_shared<ms::Ascend310DeviceInfo>();
ascend310_info->SetDeviceID(0);
context->MutableDeviceInfo().push_back(ascend310_info);
```

Load MindIR file:

```c++
// Load MindIR model
ms::Graph graph;
ms::Status ret = ms::Serialization::Load(resnet_file, ms::ModelType::kMindIR, &graph);
// Build model with graph object
ms::Model resnet50;
ret = resnet50.Build(ms::GraphCell(graph), context);
```

Get information of this model:

```c++
std::vector<ms::MSTensor> model_inputs = resnet50.GetInputs();
```

Load image file:

```c++
// Readfile is a function to read images
ms::MSTensor ReadFile(const std::string &file);
auto image = ReadFile(image_file);
```

Image preprocess(CPU operators):

```c++
// Create the CPU operator provided by MindData to get the function object

// Decode the input to RGB format
std::shared_ptr<ds::TensorTransform> decode(new ds::vision::Decode());
// Resize the image to the given size
std::shared_ptr<ds::TensorTransform> resize(new ds::vision::Resize({256}));
// Normalize the input
std::shared_ptr<ds::TensorTransform> normalize(new ds::vision::Normalize(
    {0.485 * 255, 0.456 * 255, 0.406 * 255}, {0.229 * 255, 0.224 * 255, 0.225 * 255}));
// Crop the input image at the center
std::shared_ptr<ds::TensorTransform> center_crop(new ds::vision::CenterCrop({224, 224}));
// shape (H, W, C) to shape (C, H, W)
std::shared_ptr<ds::TensorTransform> hwc2chw(new ds::vision::HWC2CHW());

// // Define a MindData preprocessor
ds::Execute preprocessor({decode, resize, normalize, center_crop, hwc2chw});

// Call the function object to get the processed image
ret = preprocessor(image, &image);
```

Execute the model:

```c++
// Create outputs vector
std::vector<ms::MSTensor> outputs;
// Create inputs vector
std::vector<ms::MSTensor> inputs;
inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                    image.Data().get(), image.DataSize());
// Call the Predict function of Model for inference
ret = resnet50.Predict(inputs, &outputs);
```

Print the result:

```c++
// Output the maximum probability to the screen
std::cout << "Image: " << image_file << " infer result: " << GetMax(outputs[0]) << std::endl;
```

#### Data pre-processing by Ascend 310 operators

Dvpp module is a hardware decoder embedded in Ascend 310 AI chip which has a better performance on image processing compare with CPU operators. Several transforms applied on JPEG format image are supported.

Using namespace of `mindspore` and `mindspore::dataset`.

```c++
namespace ms = mindspore;
namespace ds = mindspore::dataset;
```

Set global context, device target is `Ascend 310` and device id is `0`:

```c++
auto context = std::make_shared<ms::Context>();
auto ascend310_info = std::make_shared<ms::Ascend310DeviceInfo>();
ascend310_info->SetDeviceID(0);
context->MutableDeviceInfo().push_back(ascend310_info);
```

Load image file:

```c++
// Readfile is a function to read images
ms::MSTensor ReadFile(const std::string &file);
auto image = ReadFile(image_file);
```

Image preprocess(Ascend 310 operators):

```c++
// Create the CPU operator provided by MindData to get the function object

// Decode the input to YUV420 format
std::shared_ptr<ds::TensorTransform> decode(new ds::vision::Decode());
// Resize the image to the given size
std::shared_ptr<ds::TensorTransform> resize(new ds::vision::Resize({256}));
// Normalize the input
std::shared_ptr<ds::TensorTransform> normalize(new ds::vision::Normalize(
    {0.485 * 255, 0.456 * 255, 0.406 * 255}, {0.229 * 255, 0.224 * 255, 0.225 * 255}));
// Crop the input image at the center
std::shared_ptr<ds::TensorTransform> center_crop(new ds::vision::CenterCrop({224, 224}));
```

Image preprocess (Ascend 310 operators, 130% performance increasing compare to CPU operators).

Explicitly specify the computing hardware as Ascend 310.

```c++
// Define a MindData preprocessor, set deviceType = kAscend310, device id = 0
ds::Execute preprocessor({decode, resize, center_crop, normalize}, MapTargetDevice::kAscend310, 0);

// Call the function object to get the processed image
ret = preprocessor(image, &image);
```

Load MindIR file: Ascend 310 operators must bind with Aipp module, insert Aipp module for model graph compiling.

 ```c++
// Load MindIR model
ms::Graph graph;
ms::Status ret = ms::Serialization::Load(resnet_file, ms::ModelType::kMindIR, &graph);
// Build model with graph object
ascend310_info->SetInsertOpConfigPath(preprocessor.AippCfgGenerator());
ms::Model resnet50;
ret = resnet50.Build(ms::GraphCell(graph), context);
 ```

Get input information of this model:

```c++
std::vector<ms::MSTensor> model_inputs = resnet50.GetInputs();
```

Execute the model:

```c++
// Create outputs vector
std::vector<ms::MSTensor> outputs;
// Create inputs vector
std::vector<ms::MSTensor> inputs;
inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                    image.Data().get(), image.DataSize());
// Call the Predict function of Model for inference
ret = resnet50.Predict(inputs, &outputs);
```

Print the result:

```c++
// Output the maximum probability to the screen
std::cout << "Image: " << image_file << " infer result: " << GetMax(outputs[0]) << std::endl;
```

### Infer model without defining preprocess: main_hide_preprocess.cc

> Note: Only supports CV models currently.

Inference sample code: <https://gitee.com/mindspore/docs/blob/master/docs/sample_code/ascend310_resnet50_preprocess_sample/main_hide_preprocess.cc> .

Using namespace of `mindspore` and `mindspore::dataset`.

```c++
namespace ms = mindspore;
namespace ds = mindspore::dataset;
```

Set global context, device target is `Ascend 310` and device id is `0`:

```c++
auto context = std::make_shared<ms::Context>();
auto ascend310_info = std::make_shared<ms::Ascend310DeviceInfo>();
ascend310_info->SetDeviceID(0);
context->MutableDeviceInfo().push_back(ascend310_info);
```

Load MindIR file:

```c++
// Load MindIR model
ms::Graph graph;
ms::Status ret = ms::Serialization::Load(resnet_file, ms::ModelType::kMindIR, &graph);
// Build model with graph object
ms::Model resnet50;
ret = resnet50.Build(ms::GraphCell(graph), context);
```

Get information of this model:

```c++
std::vector<ms::MSTensor> model_inputs = resnet50.GetInputs();
```

Read image and start data preprocessing and prediction:

```c++
std::vector<MSTensor> inputs = {ReadFile(image_path)};
std::vector<MSTensor> outputs;
ret = resnet50.PredictWithPreprocess(inputs, &outputs);
```

Print the result:

```c++
// Output the maximum probability to the screen
std::cout << "Image: " << image_file << " infer result: " << GetMax(outputs[0]) << std::endl;
```

## Introduce to Building Script

The building script is used to building applications: <https://gitee.com/mindspore/docs/blob/master/docs/sample_code/ascend310_resnet50_preprocess_sample/CMakeLists.txt>.

Add head files to gcc search path:

```cmake
option(MINDSPORE_PATH "mindspore install path" "")
include_directories(${MINDSPORE_PATH})
include_directories(${MINDSPORE_PATH}/include)
```

Find the shared libraries in MindSpore:

```cmake
find_library(MS_LIB libmindspore.so ${MINDSPORE_PATH}/lib)
file(GLOB_RECURSE MD_LIB ${MINDSPORE_PATH}/_c_dataengine*)
```

Use the source files to generate the target executable file, and link the MindSpore libraries for the executable file:

```cmake
add_executable(resnet50_sample main.cc)
target_link_libraries(resnet50_sample ${MS_LIB} ${MD_LIB})

add_executable(resnet50_hide_preprocess main_hide_preprocess.cc)
target_link_libraries(resnet50_hide_preprocess ${MS_LIB} ${MD_LIB})
```

## Building Inference Code

Go to the project directory `ascend310_resnet50_preprocess_sample` and set the following environment variables:

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# lib libraries that the mindspore depends on, modify "pip3" according to the actual situation
export LD_LIBRARY_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore/lib"}' | xargs realpath`:${LD_LIBRARY_PATH}
# if MindSpore is installed by binary, run "export LD_LIBRARY_PATH=path-to-your-custom-dir:${LD_LIBRARY_PATH}"

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}                 # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                       # Python library that TBE implementation depends on
```

Run the `cmake` command, modify `pip3` according to the actual situation:

```bash
cmake . -DMINDSPORE_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`
# if MindSpore is installed by binary, run "cmake . -DMINDSPORE_PATH=path-to-your-custom-dir"
```

Run the `make` command for building.

```bash
make
```

After building, the executable `main` file is generated in `ascend310_resnet50_preprocess_sample`.

## Performing Inference and Viewing the Result

Log in to the Ascend 310 server, and create the `model` directory for storing the MindIR file `resnet50_imagenet.mindir`, for example, `/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_resnet50_preprocess_sample/model`.
Create the `test_data` directory to store images, for example, `/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_resnet50_preprocess_sample/test_data`.
Then, perform the inference.

If your MindIR file does not contain preprocess information, you can execute the following command:

```bash
./resnet50_sample
```

Inference is performed on all images stored in the `test_data` directory. For example, if there are 9 images whose label is 0 in the [ImageNet2012](http://image-net.org/download-images) validation set, the inference result is as follows:

```text
Image: ./test_data/ILSVRC2012_val_00002138.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00003014.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00006697.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00007197.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00009111.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00009191.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00009346.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00009379.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00009396.JPEG infer result: 0
```

If you export the preprocess information simultaneously when you export a MindIR file, you can execute the following command:

```bash
./resnet50_hide_preprocess
```

The model will load the image file inside the `test_data` directory (for example: ILSVRC2012_val_00002138.JPEG,
configable in main_hide_preprocess.cc) and start prediction, then you get the inference result as follows:

```text
Image: ./test_data/ILSVRC2012_val_00002138.JPEG infer result: 0
```
