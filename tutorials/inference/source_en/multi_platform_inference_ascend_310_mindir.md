# Inference Using the MindIR Model on Ascend 310 AI Processors

`Linux` `Ascend` `Inference Application` `Beginner` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r1.1/tutorials/inference/source_en/multi_platform_inference_ascend_310_mindir.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Overview

Ascend 310 is a highly efficient and integrated AI processor oriented to edge scenarios. The Atlas 200 Developer Kit (Atlas 200 DK) is a developer board that uses the Atlas 200 AI accelerator module. Integrated with the HiSilicon Ascend 310 AI processor, the Atlas 200 allows data analysis, inference, and computing for various data such as images and videos, and can be widely used in scenarios such as intelligent surveillance, robots, drones, and video servers.

This tutorial describes how to use MindSpore to perform inference on the Atlas 200 DK. The process is as follows:

1. Prepare the development environment, including creating an SD card for the Atlas 200 DK, configuring the Python environment, and updating the development software package.

2. Export the MindIR model file. The ResNet-50 model is used as an example.

3. Build the inference code to generate an executable `main` file.

4. Load the saved MindIR model, perform inference, and view the result.

> You can obtain the complete executable sample code at <https://gitee.com/mindspore/docs/tree/r1.1/tutorials/tutorial_code/ascend310_resnet50_preprocess_sample>.

## Preparing the Development Environment

For details, see [Inference on the Ascend 310 AI Processor](https://www.mindspore.cn/tutorial/inference/en/r1.1/multi_platform_inference_ascend_310_air.html#preparing-the-development-environment).

## Exporting the MindIR Model

Train the target network on the Ascend 910 AI Processor, save it as a checkpoint file, and export the model file in MindIR format through the network and checkpoint file. For details about the export process, see [Export MindIR Model](https://www.mindspore.cn/tutorial/training/en/r1.1/use/save_model.html#export-mindir-model).

> The [resnet50_imagenet.mindir](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/sample_resources/ascend310_resnet50_preprocess_sample/resnet50_imagenet.mindir) is a sample MindIR file exported using the ResNet-50 model.

## Inference Directory Structure

Create a directory to store the inference code project, for example, `/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_resnet50_preprocess_sample`. The directory code can be obtained from the [official website](https://gitee.com/mindspore/docs/tree/r1.1/tutorials/tutorial_code/ascend310_resnet50_preprocess_sample). The `model` directory stores the exported `MindIR` model files and the `test_data` directory stores the images to be classified. The directory structure of the inference code project is as follows:

```text
└─ascend310_resnet50_preprocess_sample
    ├── CMakeLists.txt                    // Build script
    ├── README.md                         // Usage description
    ├── main.cc                           // Main function
    ├── model
    │   └── resnet50_imagenet.mindir      // MindIR model file
    └── test_data
        ├── ILSVRC2012_val_00002138.JPEG  // Input sample image 1
        ├── ILSVRC2012_val_00003014.JPEG  // Input sample image 2
        ├── ...                           // Input sample image n
```

## Inference Code

Inference sample code: <https://gitee.com/mindspore/docs/blob/r1.1/tutorials/tutorial_code/ascend310_resnet50_preprocess_sample/main.cc> .

Set global context, device target is `Ascend310` and evice id is `0`:

```c++
ms::GlobalContext::SetGlobalDeviceTarget(ms::kDeviceTypeAscend310);
ms::GlobalContext::SetGlobalDeviceID(0);
```

Load mindir file:

```c++
// Load MindIR model
auto graph =ms::Serialization::LoadModel(resnet_file, ms::ModelType::kMindIR);
// Build model with graph object
ms::Model resnet50((ms::GraphCell(graph)));
ms::Status ret = resnet50.Build({});
```

Get informance of this model:

```c++
std::vector<ms::MSTensor> model_inputs = resnet50.GetInputs();
```

Load image file:

```c++
// Readfile is a function to read images
ms::MSTensor ReadFile(const std::string &file);
auto image = ReadFile(image_file);
```

Image preprocess:

```c++
// Create the CPU operator provided by MindData to get the function object
ms::dataset::Execute preprocessor({ms::dataset::vision::Decode(),  // Decode the input to RGB format
                                   ms::dataset::vision::Resize({256}),  // Resize the image to the given size
                                   ms::dataset::vision::Normalize({0.485 * 255, 0.456 * 255, 0.406 * 255},
                                                                  {0.229 * 255, 0.224 * 255, 0.225 * 255}),  // Normalize the input
                                   ms::dataset::vision::CenterCrop({224, 224}),  // Crop the input image at the center
                                   ms::dataset::vision::HWC2CHW(),  // shape (H, W, C) to shape(C, H, W)
                                  });
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

## Introduce to Building Script

The building script is used to building applications: <https://gitee.com/mindspore/docs/blob/r1.1/tutorials/tutorial_code/ascend310_resnet50_preprocess_sample/CMakeLists.txt>.

Since MindSpore uses the [old C++ ABI](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html), applications must be the same with MindSpore, otherwise the building will fail.

```cmake
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_STANDARD 17)
```

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
```

## Building Inference Code

Go to the project directory `ascend310_resnet50_preprocess_sample` and set the following environment variables:

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/acllib/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/atc/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# lib libraries that the mindspore depends on, modify "pip3" according to the actual situation
export LD_LIBRARY_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore/lib"}' | xargs realpath`:${LD_LIBRARY_PATH}

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/atc/ccec_compiler/bin/:${PATH}                       # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                       # Python library that TBE implementation depends on
```

Run the `cmake` command, modify `pip3` according to the actual situation:

```bash
cmake . -DMINDSPORE_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`
```

Run the `make` command for building.

```bash
make
```

After building, the executable `main` file is generated in `ascend310_resnet50_preprocess_sample`.

## Performing Inference and Viewing the Result

Log in to the Atlas 200 DK developer board, and create the `model` directory for storing the MindIR file `resnet50_imagenet.mindir`, for example, `/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_resnet50_preprocess_sample/model`.
Create the `test_data` directory to store images, for example, `/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_resnet50_preprocess_sample/test_data`.
Then, perform the inference.

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
