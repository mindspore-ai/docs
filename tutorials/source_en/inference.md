# Inference

`Ascend` `Device` `Beginner` `Inference Application`

<a href="https://gitee.com/mindspore/docs/blob/r1.5/tutorials/source_en/inference.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

This is the last tutorial. To better adapt to different inference devices, inference is classified into Ascend AI Processor inference and mobile device inference.

## Ascend AI Processor Inference

An Ascend AI Processor is an energy-efficient and highly integrated AI processor oriented to edge scenarios. It can implement multiple data analysis and inference computing, such as image and video analysis, and can be widely used in scenarios such as intelligent surveillance, robots, drones, and video servers. The following describes how to use MindSpore to perform inference on the Ascend AI Processors.

### Inference Code

Create a directory to store the inference code project, for example, `/home/HwHiAiUser/mindspore_sample/ascend910_resnet50_preprocess_sample`. You can download the [sample code](https://gitee.com/mindspore/docs/tree/r1.5/docs/sample_code/ascend910_resnet50_preprocess_sample) from the official website. The `model` directory is used to store the exported `MindIR` model file, and the `test_data` directory is used to store the images to be classified. The directory structure of the inference code project is as follows:

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
><https://gitee.com/mindspore/docs/blob/r1.5/docs/sample_code/ascend910_resnet50_preprocess_sample/CMakeLists.txt>

### Building Inference Code

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

> For details about the complete instance running on the mobile phone, see [Android Application Development Based on JNI](https://www.mindspore.cn/lite/docs/en/r1.5/quick_start/quick_start.html).

### Model Conversion

The format of a model needs to be converted before the model is used for inference on the device. Currently, MindSpore Lite supports four types of AI frameworks: MindSpore, TensorFlow Lite, Caffe, and ONNX.

The following uses the [mobilenetv2.mindir](https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/mobilenetv2.mindir) model trained by MindSpore as an example to describe how to generate the `mobilenetv2.ms` model used in the demo.

> The following describes the conversion process. Skip it if you only need to run the demo.
>
> The following describes only the model used by the demo. For details about how to use the conversion tool, see [Converting Models for Inference](https://www.mindspore.cn/lite/docs/en/r1.5/use/converter_tool.html#).

- Download the conversion tool.

  Download the [conversion tool package](https://www.mindspore.cn/lite/docs/en/r1.5/use/downloads.html) based on the OS in use, decompress the package to a local directory, obtain the `converter` tool, and configure environment variables.

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

- Build

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
  tensor name is:Default/head-MobileNetV2Head/Softmax-op204 tensor size is:4000 tensor elements num is:1000
  output data is:5.26823e-05 0.00049752 0.000296722 0.000377607 0.000177048 .......
  ```

#### Building and Running the Windows System

- Build

    - Download the library: Manually download the MindSpore Lite model inference framework [mindspore-lite-{version}-win-x64.zip](https://www.mindspore.cn/lite/docs/en/r1.5/use/downloads.html) whose hardware platform is CPU and operating system is Windows-x64. Copy the `libmindspore-lite.a` file in the decompressed `inference/lib` directory to the `mindspore/lite/examples/quick_start_cpp/lib` directory. Copy the `inference/include` directory to the `mindspore/lite/examples/quick_start_cpp/include` directory.

    - Download the model: Manually download the model file [mobilenetv2.ms](https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_imagenet/mobilenetv2.ms) and copy it to the `mindspore/lite/examples/quick_start_cpp/model` directory.

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
  tensor name is:Default/head-MobileNetV2Head/Softmax-op204 tensor size is:4000 tensor elements num is:1000
  output data is:5.26823e-05 0.00049752 0.000296722 0.000377607 0.000177048 .......
  ```

### Inference Code Parsing

The following analyzes the inference process in the demo source code and shows how to use the C++ API.

#### Model Loading

Read the MindSpore Lite model from the file system and use the `mindspore::lite::Model::Import` function to import the model for parsing.

```c++
// Read the model file.
size_t size = 0;
char *model_buf = ReadFile(model_path, &size);
if (model_buf == nullptr) {
  std::cerr << "Read model file failed." << std::endl;
  return RET_ERROR;
}
// Load the model.
auto model = mindspore::lite::Model::Import(model_buf, size);
delete[](model_buf);
if (model == nullptr) {
  std::cerr << "Import model file failed." << std::endl;
  return RET_ERROR;
}
```

#### Model Build

Model build includes configuration context creation, session creation, and graph build.

```c++
mindspore::session::LiteSession *Compile(mindspore::lite::Model *model) {
  // Initialize the context.
  auto context = std::make_shared<mindspore::lite::Context>();
  if (context == nullptr) {
    std::cerr << "New context failed while." << std::endl;
    return nullptr;
  }

  // Create a session.
  mindspore::session::LiteSession *session = mindspore::session::LiteSession::CreateSession(context.get());
  if (session == nullptr) {
    std::cerr << "CreateSession failed while running." << std::endl;
    return nullptr;
  }

  // Graph build.
  auto ret = session->CompileGraph(model);
  if (ret != mindspore::lite::RET_OK) {
    delete session;
    std::cerr << "Compile failed while running." << std::endl;
    return nullptr;
  }

  // Note: If model->Free() is used, the model cannot be built again.
  if (model != nullptr) {
    model->Free();
  }
  return session;
}
```

#### Model Inference

Model inference includes data input, inference execution, and output obtaining. In this example, the input data is generated from randomly built data, and the output result after inference is displayed.

```c++
int Run(mindspore::session::LiteSession *session) {
  // Obtain the input data.
  auto inputs = session->GetInputs();
  auto ret = GenerateInputDataWithRandom(inputs);
  if (ret != mindspore::lite::RET_OK) {
    std::cerr << "Generate Random Input Data failed." << std::endl;
    return ret;
  }

  // Run.
  ret = session->RunGraph();
  if (ret != mindspore::lite::RET_OK) {
    std::cerr << "Inference error " << ret << std::endl;
    return ret;
  }

  // Obtain the output data.
  auto out_tensors = session->GetOutputs();
  for (auto tensor : out_tensors) {
    std::cout << "tensor name is:" << tensor.first << " tensor size is:" << tensor.second->Size()
              << " tensor elements num is:" << tensor.second->ElementsNum() << std::endl;
    auto out_data = reinterpret_cast<float *>(tensor.second->MutableData());
    std::cout << "output data is:";
    for (int i = 0; i < tensor.second->ElementsNum() && i <= 50; i++) {
      std::cout << out_data[i] << " ";
    }
    std::cout << std::endl;
  }
  return mindspore::lite::RET_OK;
}
```

#### Releasing Memory

If the MindSpore Lite inference framework is not required, you need to release the created `LiteSession` and `Model`.

```c++
// Delete the model cache.
delete model;
// Delete the session cache.
delete session;
```
