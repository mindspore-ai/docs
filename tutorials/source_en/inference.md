# Inference

`Ascend` `Device` `Beginner` `Inference Application`

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/inference.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

This is the last tutorial. To better adapt to different inference devices, inference is classified into Ascend AI Processor inference and mobile device inference.

## Ascend AI Processor Inference

An Ascend AI Processor is an energy-efficient and highly integrated AI processor oriented to edge scenarios. It can implement multiple data analysis and inference computing, such as image and video analysis, and can be widely used in scenarios such as intelligent surveillance, robots, drones, and video servers. The following describes how to use MindSpore to perform inference on the Ascend AI Processors.

### Inference Code

Create a directory to store the inference code project, for example, `/home/HwHiAiUser/mindspore_sample/ascend910_resnet50_preprocess_sample`. You can download the [sample code](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/ascend910_resnet50_preprocess_sample) from the official website. The `model` directory is used to store the exported [MindIR model file](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/sample_resources/ascend310_resnet50_preprocess_sample/resnet50_imagenet.mindir), and the `test_data` directory is used to store the images to be classified, the images can be selected in [ImageNet2012](http://image-net.org/download-images) validation dataset. The directory structure of the inference code project is as follows:

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

### Build Script

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
><https://gitee.com/mindspore/docs/blob/master/docs/sample_code/ascend910_resnet50_preprocess_sample/CMakeLists.txt>

### Building Inference Code

Go to the project directory `ascend910_resnet50_preprocess_sample` and set the following environment variables:

> If the device is Ascend 310, go to the project directory `ascend310_resnet50_preprocess_sample`. The following code uses Ascend 910 as an example. By the way, MindSpore is supporting data preprocess + model inference in one key on Ascend 310 platform. If you are interented in it, kindly refer to [more details](https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference_ascend_310_mindir.html).

```bash
# Control the log print level. 0 indicates DEBUG, 1 indicates INFO, 2 indicates WARNING (default value), 3 indicates ERROR, and 4 indicates CRITICAL.
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

After building, the executable file is generated in `ascend910_resnet50_preprocess_sample`.

### Performing Inference and Viewing the Result

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

## Mobile Device Inference

MindSpore Lite is the device part of the device-edge-cloud AI framework MindSpore and can implement intelligent applications on mobile devices such as phones. MindSpore Lite provides a high-performance inference engine and ultra-lightweight solution. It supports mobile phone operating systems such as iOS and Android, LiteOS-embedded operating systems, various intelligent devices such as mobile phones, large screens, tablets, and IoT devices, and MindSpore/TensorFlow Lite/Caffe/ONNX model applications.

The following provides a demo that runs on the Windows and Linux operating systems and is built based on the C++ API to help users get familiar with the on-device inference process. The demo uses the shuffled data as the input data, performs the inference on the MobileNetV2 model, and directly displays the output data on the computer.

> For details about the complete instance running on the mobile phone, see [Android Application Development Based on JNI](https://www.mindspore.cn/lite/docs/en/master/quick_start/quick_start.html).

### Model Conversion

The format of a model needs to be converted before the model is used for inference on the device. Currently, MindSpore Lite supports four types of AI frameworks: MindSpore, TensorFlow Lite, Caffe, and ONNX.

The following uses the [mobilenetv2.mindir](https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/1.5/mobilenetv2.mindir) model trained by MindSpore as an example to describe how to generate the `mobilenetv2.ms` model used in the demo.

> The following describes the conversion process. Skip it if you only need to run the demo.
>
> The following describes only the model used by the demo. For details about how to use the conversion tool, see [Converting Models for Inference](https://www.mindspore.cn/lite/docs/en/master/use/converter_tool.html#).

- Download the conversion tool.

  Download the [conversion tool package](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html) based on the OS in use, decompress the package to a local directory, obtain the `converter` tool, and configure environment variables.

- Use the conversion tool.

    - For Linux

        Go to the directory where the `converter_lite` executable file is located, place the downloaded `mobilenetv2.mindir` model in the same path, and run the following command on the PC to convert the model:

        ```cpp
        ./converter_lite --fmk=MINDIR --modelFile=mobilenetv2.mindir --outputFile=mobilenetv2
        ```

    - For Windows

        Go to the directory where the `converter_lite` executable file is located, place the downloaded `mobilenetv2.mindir` model in the same path, and run the following command on the PC to convert the model:

        ```cpp
        call converter_lite --fmk=MINDIR --modelFile=mobilenetv2.mindir --outputFile=mobilenetv2
        ```

    - Parameter description

        During the command execution, three parameters are set. `--fmk` indicates the original format of the input model. In this example, this parameter is set to `MINDIR`, which is the export format of the MindSpore framework training model. `--modelFile` indicates the path of the input model. `--outputFile` indicates the output path of the model. The suffix `.ms` is automatically added to the converted model.

### Environment Building and Running

#### Building and Running the Linux System

- Build. Renference [Building MindSpore Lite](https://mindspore.cn/lite/docs/en/master/use/build.html#environment-requirements) to get the Environment Requirements.

  Run the build script in the `mindspore/lite/examples/quick_start_cpp` directory to automatically download related files and build the demo.

  ```bash
  bash build.sh
  ```

- Inference

  After the build is complete, go to the `mindspore/lite/examples/quick_start_cpp/build` directory and run the following command to perform MindSpore Lite inference on the MobileNetV2 model.

  ```bash
  ./mindspore_quick_start_cpp ../model/mobilenetv2.ms
  ```

  After the execution is complete, the following information is displayed, including the tensor name, tensor size, number of output tensors, and the first 50 pieces of data:

  ```text
  tensor name is: Softmax-65 tensor size is: 4004 tensor elements num is: 1001
  output data is: 1.74225e-05 1.15919e-05 2.02728e-05 0.000106485 0.000124295 0.00140576 0.000185107 0.000762011 1.50996e-05 5.91942e-06 6.61469e-06 3.72883e-06 4.30761e-06 2.38897e-06 1.5163e-05 0.000192663 1.03767e-05 1.31953e-05 6.69638e-06 3.17411e-05 4.00895e-06 9.9641e-06 3.85127e-06 6.25101e-06 9.08853e-06 1.25043e-05 1.71761e-05 4.92751e-06 2.87637e-05 7.46446e-06 1.39375e-05 2.18824e-05 1.08861e-05 2.5007e-06 3.49876e-05 0.000384547 5.70778e-06 1.28909e-05 1.11038e-05 3.53906e-06 5.478e-06 9.76608e-06 5.32172e-06 1.10386e-05 5.35474e-06 1.35796e-05 7.12652e-06 3.10017e-05 4.34154e-06 7.89482e-05 1.79441e-05
  ```

#### Building and Running the Windows System

- Build. Renference [Building MindSpore Lite](https://mindspore.cn/lite/docs/en/master/use/build.html#id1) to get the Environment Requirements.

    - Download the library: Manually download the MindSpore Lite model inference framework [mindspore-lite-{version}-win-x64.zip](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html) whose hardware platform is CPU and operating system is Windows-x64. Copy the `libmindspore-lite.a` file in the decompressed `inference/lib` directory to the `mindspore/lite/examples/quick_start_cpp/lib` directory. Copy the `inference/include` directory to the `mindspore/lite/examples/quick_start_cpp/include` directory.

    - Download the model: Manually download the model file [mobilenetv2.ms](https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/1.5/mobilenetv2.ms) and copy it to the `mindspore/lite/examples/quick_start_cpp/model` directory.

        > You can use the mobilenetv2.ms model file obtained in "Model Conversion".

    - Build: Run the build script in the `mindspore/lite/examples/quick_start_cpp` directory to automatically download related files and build the demo.

        ```bash
        call build.bat
        ```

- Inference

  After the build is complete, go to the `mindspore/lite/examples/quick_start_cpp/build` directory and run the following command to perform MindSpore Lite inference on the MobileNetV2 model.

  ```bash
  call ./mindspore_quick_start_cpp.exe ../model/mobilenetv2.ms
  ```

  After the execution is complete, the following information is displayed, including the tensor name, tensor size, number of output tensors, and the first 50 pieces of data:

  ```text
  tensor name is: Softmax-65 tensor size is: 4004 tensor elements num is: 1001
  output data is: 1.74225e-05 1.15919e-05 2.02728e-05 0.000106485 0.000124295 0.00140576 0.000185107 0.000762011 1.50996e-05 5.91942e-06 6.61469e-06 3.72883e-06 4.30761e-06 2.38897e-06 1.5163e-05 0.000192663 1.03767e-05 1.31953e-05 6.69638e-06 3.17411e-05 4.00895e-06 9.9641e-06 3.85127e-06 6.25101e-06 9.08853e-06 1.25043e-05 1.71761e-05 4.92751e-06 2.87637e-05 7.46446e-06 1.39375e-05 2.18824e-05 1.08861e-05 2.5007e-06 3.49876e-05 0.000384547 5.70778e-06 1.28909e-05 1.11038e-05 3.53906e-06 5.478e-06 9.76608e-06 5.32172e-06 1.10386e-05 5.35474e-06 1.35796e-05 7.12652e-06 3.10017e-05 4.34154e-06 7.89482e-05 1.79441e-05
  ```

### Inference Code Parsing

The following analyzes the inference process in the demo source code and shows how to use the C++ API.

#### Model Reading

Read the MindSpore Lite model from the file system and store it in the memory buffer.

```c++
// Read model file.
size_t size = 0;
char *model_buf = ReadFile(model_path, &size);
if (model_buf == nullptr) {
  std::cerr << "Read model file failed." << std::endl;
  return -1;
}
```

#### Creating and Configuring Context

```c++
// Create and init context, add CPU device info
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
  std::cerr << "New context failed." << std::endl;
  return -1;
}
auto &device_list = context->MutableDeviceInfo();
auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
if (device_info == nullptr) {
  std::cerr << "New CPUDeviceInfo failed." << std::endl;
  return -1;
}
device_list.push_back(device_info);
```

#### Model Creating Loading and Building

Use Build of [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) to load the model directly from the memory buffer and build the model.

```c++
// Create model
auto model = new (std::nothrow) mindspore::Model();
if (model == nullptr) {
  std::cerr << "New Model failed." << std::endl;
  return -1;
}
// Build model
auto build_ret = model->Build(model_buf, size, mindspore::kMindIR, context);
delete[](model_buf);
if (build_ret != mindspore::kSuccess) {
  std::cerr << "Build model error " << build_ret << std::endl;
  return -1;
}
```

There is another method that uses Load of [Serialization](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Serialization.html#class-serialization) to load [Graph](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Graph.html#class-graph) and use Build of [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) to build the model.

```c++
// Load graph.
mindspore::Graph graph;
auto load_ret = mindspore::Serialization::Load(model_buf, size, mindspore::kMindIR, &graph);
delete[](model_buf);
if (load_ret != mindspore::kSuccess) {
  std::cerr << "Load graph file failed." << std::endl;
  return -1;
}

// Create model
auto model = new (std::nothrow) mindspore::Model();
if (model == nullptr) {
  std::cerr << "New Model failed." << std::endl;
  return -1;
}
// Build model
mindspore::GraphCell graph_cell(graph);
auto build_ret = model->Build(graph_cell, context);
if (build_ret != mindspore::kSuccess) {
  delete model;
  std::cerr << "Build model error " << build_ret << std::endl;
  return -1;
}
```

#### Model Inference

Model inference includes input data injection, inference execution, and output obtaining. In this example, the input data is randomly generated, and the output result is printed after inference.

```c++
auto inputs = model->GetInputs();
// Generate random data as input data.
auto ret = GenerateInputDataWithRandom(inputs);
if (ret != mindspore::kSuccess) {
  delete model;
  std::cerr << "Generate Random Input Data failed." << std::endl;
  return -1;
}
// Get Output
auto outputs = model->GetOutputs();

// Model Predict
auto predict_ret = model->Predict(inputs, &outputs);
if (predict_ret != mindspore::kSuccess) {
  delete model;
  std::cerr << "Predict model error " << predict_ret << std::endl;
  return -1;
}

// Print Output Tensor Data.
for (auto tensor : outputs) {
  std::cout << "tensor name is:" << tensor.Name() << " tensor size is:" << tensor.DataSize()
            << " tensor elements num is:" << tensor.ElementNum() << std::endl;
  auto out_data = reinterpret_cast<const float *>(tensor.Data().get());
  std::cout << "output data is:";
  for (int i = 0; i < tensor.ElementNum() && i <= 50; i++) {
    std::cout << out_data[i] << " ";
  }
  std::cout << std::endl;
}
```

#### Memory Release

If the inference process of MindSpore Lite is complete, release the created `Model`.

```c++
// Delete model.
delete model;
```
