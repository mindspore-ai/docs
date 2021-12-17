# Inference

`Ascend` `Device` `Beginner` `Inference Application`

<a href="https://gitee.com/mindspore/docs/blob/r1.5/tutorials/source_en/inference.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

This is the last tutorial. Introduction to inference is classified into Ascend AI Processor.

An Ascend AI Processor is an energy-efficient and highly integrated AI processor oriented to edge scenarios. It can implement multiple data analysis and inference computing, such as image and video analysis, and can be widely used in scenarios such as intelligent surveillance, robots, drones, and video servers. The following describes how to use MindSpore to perform inference on the Ascend AI Processors.

## Inference Code

Create a directory to store the inference code project, for example, `/home/HwHiAiUser/mindspore_sample/ascend910_resnet50_preprocess_sample`. You can download the [sample code](https://gitee.com/mindspore/docs/tree/r1.5/docs/sample_code/ascend910_resnet50_preprocess_sample) from the official website. The `model` directory is used to store the exported [MindIR model file](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/sample_resources/ascend310_resnet50_preprocess_sample/resnet50_imagenet.mindir), and the `test_data` directory is used to store the images to be classified, the images can be selected in [ImageNet2012](http://image-net.org/download-images) validation dataset. The directory structure of the inference code project is as follows:

```text
└─ascend910_resnet50_preprocess_sample
    ├── CMakeLists.txt                   // Build script
    ├── README.md                         // Usage description
    ├── main.cc                           // Main function
    ├── model
    │   └── resnet50_imagenet.mindir      // MindIR model file
    └── test_data
        ├── ILSVRC2012_val_00002138.JPEG  // Input sample image 1
        ├── ILSVRC2012_val_00003014.JPEG  // Input sample image 2
        ├── ...                           // Input sample image n.
```

Namespaces that reference `mindspore` and `mindspore::dataset`.

```c++
namespace ms = mindspore;
namespace ds = mindspore::dataset;
```

Initialize the environment, specify the hardware platform used for inference, and set DeviceID.

Set the hardware to Ascend 910 and set DeviceID to 0. The code example is as follows:

```c++
auto context = std::make_shared<ms::Context>();
auto ascend910_info = std::make_shared<ms::Ascend910DeviceInfo>();
ascend910_info->SetDeviceID(0);
context->MutableDeviceInfo().push_back(ascend910_info);
```

Load the model file.

```c++
// Load the MindIR model.
ms::Graph graph;
ms::Status ret = ms::Serialization::Load(resnet_file, ms::ModelType::kMindIR, &graph);
// Build a model using a graph.
ms::Model resnet50;
ret = resnet50.Build(ms::GraphCell(graph), context);
```

Obtain the input information required by the model.

```c++
std::vector<ms::MSTensor> model_inputs = resnet50.GetInputs();
```

Load the image file.

```c++
// ReadFile is a function used to read images.
ms::MSTensor ReadFile(const std::string &file);
auto image = ReadFile(image_file);
```

Preprocess images.

```c++
// Use the CPU operator provided by MindData to preprocess images.

// Create an operator to encode the input into the RGB format.
std::shared_ptr<ds::TensorTransform> decode(new ds::vision::Decode());
// Create an operator to resize the image to the specified size.
std::shared_ptr<ds::TensorTransform> resize(new ds::vision::Resize({256}));
// Create an operator to normalize the input of the operator.
std::shared_ptr<ds::TensorTransform> normalize(new ds::vision::Normalize(
    {0.485 * 255, 0.456 * 255, 0.406 * 255}, {0.229 * 255, 0.224 * 255, 0.225 * 255}));
// Create an operator to perform central cropping.
std::shared_ptr<ds::TensorTransform> center_crop(new ds::vision::CenterCrop({224, 224}));
// Create an operator to transform shape (H, W, C) into shape (C, H, W).
std::shared_ptr<ds::TensorTransform> hwc2chw(new ds::vision::HWC2CHW());

// Define a MindData data preprocessing function that contains the preceding operators in sequence.
ds::Execute preprocessor({decode, resize, normalize, center_crop, hwc2chw});

// Call the data preprocessing function to obtain the processed image.
ret = preprocessor(image, &image);
```

Start inference.

```c++
// Create an output vector.
std::vector<ms::MSTensor> outputs;
// Create an input vector.
std::vector<ms::MSTensor> inputs;
inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                    image.Data().get(), image.DataSize());
// Call the Predict function of the model for inference.
ret = resnet50.Predict(inputs, &outputs);
```

Obtain the inference result.

```c++
// Maximum value of the output probability.
std::cout << "Image: " << image_file << " infer result: " << GetMax(outputs[0]) << std::endl;
```

## Build Script

Add the header file search path for the compiler:

```cmake
option(MINDSPORE_PATH "mindspore install path" "")
include_directories(${MINDSPORE_PATH})
include_directories(${MINDSPORE_PATH}/include)
```

Search for the required dynamic library in MindSpore.

```cmake
find_library(MS_LIB libmindspore.so ${MINDSPORE_PATH}/lib)
file(GLOB_RECURSE MD_LIB ${MINDSPORE_PATH}/_c_dataengine*)
```

Use the specified source file to generate the target executable file and link the target file to the MindSpore library.

```cmake
add_executable(resnet50_sample main.cc)
target_link_libraries(resnet50_sample ${MS_LIB} ${MD_LIB})
```

>For details, see
><https://gitee.com/mindspore/docs/blob/r1.5/docs/sample_code/ascend910_resnet50_preprocess_sample/CMakeLists.txt>

## Building Inference Code

Go to the project directory `ascend910_resnet50_preprocess_sample` and set the following environment variables:

> If the device is Ascend 310, go to the project directory `ascend310_resnet50_preprocess_sample`. The following code uses Ascend 910 as an example.

```bash
# Control the log print level. 0 indicates DEBUG, 1 indicates INFO, 2 indicates WARNING (default value), and 3 indicates ERROR.
export GLOG_v=2

# Select the Conda environment.
LOCAL_ASCEND=/usr/local/Ascend # Root directory of the running package

# Library on which the running package depends
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64/common:${LOCAL_ASCEND}/driver/lib64/driver:${LOCAL_ASCEND}/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Libraries on which MindSpore depends
export LD_LIBRARY_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore/lib"}' | xargs realpath`:${LD_LIBRARY_PATH}

# Configure necessary environment variables.
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # Path of the TBE operator
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}                 # Path of the TBE operator build tool
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                       # Python library that TBE depends on
```

Run the `cmake` command. In the command, `pip3` needs to be modified based on the actual situation:

```bash
cmake . -DMINDSPORE_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`
```

Run the `make` command for building.

```bash
make
```

After building, the executable `main` file is generated in `ascend910_resnet50_preprocess_sample`.

## Performing Inference and Viewing the Result

After the preceding operations are complete, you can learn how to perform inference.

Log in to the Ascend 910 environment, and create the `model` directory to store the `resnet50_imagenet.mindir` file, for example, `/home/HwHiAiUser/mindspore_sample/ascend910_resnet50_preprocess_sample/model`.
Create the `test_data` directory to store images, for example, `/home/HwHiAiUser/mindspore_sample/ascend910_resnet50_preprocess_sample/test_data`.
Then, perform the inference.

```bash
./resnet50_sample
```

Inference is performed on all images stored in the `test_data` directory. For example, if there are 2 images whose label is 0 in the [ImageNet2012](http://image-net.org/download-images) validation set, the inference result is as follows:

```text
Image: ./test_data/ILSVRC2012_val_00002138.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00003014.JPEG infer result: 0
```
