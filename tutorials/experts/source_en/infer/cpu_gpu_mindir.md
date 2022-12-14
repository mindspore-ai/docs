# Inference on a GPU

<a href="https://gitee.com/mindspore/docs/blob/r1.9/tutorials/experts/source_en/infer/cpu_gpu_mindir.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png"></a>

## Use C++ Interface to Load a MindIR File for Inferencing

Users can create C++ applications to call MindSpore's C++ interface to infer the MindIR model.

### Inference Directory Structure

Create a directory to store the inference code project, for example, `/home/mindspore_sample/gpu_resnet50_inference_sample`. You can download the [sample code](https://gitee.com/mindspore/docs/tree/r1.9/docs/sample_code/gpu_resnet50_inference_sample) from the official website. The `model` directory is used to store the exported `MindIR` [model file](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/sample_resources/ascend310_resnet50_preprocess_sample/resnet50_imagenet.mindir). The directory structure of the inference code project is as follows:

```text
└─gpu_resnet50_inference_sample
    ├── build.sh                          // Build script
    ├── CMakeLists.txt                    // CMake script
    ├── README.md                         // Usage description
    ├── src
    │   └── main.cc                       // Main function
    └── model
        └── resnet50_imagenet.mindir      // MindIR model file
```

### Inference Code

Inference sample code: [gpu_resnet50_inference_sample](https://gitee.com/mindspore/docs/blob/r1.9/docs/sample_code/gpu_resnet50_inference_sample/src/main.cc).

Using namespace of `mindspore`:

```c++
using mindspore::Context;
using mindspore::Serialization;
using mindspore::Model;
using mindspore::Status;
using mindspore::ModelType;
using mindspore::GraphCell;
using mindspore::kSuccess;
using mindspore::MSTensor;
```

Initialize the environment, specify the hardware platform used for inference, and set DeviceID and precision.

Set the hardware to GPU, set DeviceID to 0 and inference precision Mode to FP16. The code example is as follows:

```c++
auto gpu_device_info = std::make_shared<mindspore::GPUDeviceInfo>();
gpu_device_info->SetDeviceID(device_id);
gpu_device_info->SetPrecisionMode("fp16");
context->MutableDeviceInfo().push_back(gpu_device_info);
```

Load the model file.

```c++
// Load the MindIR model.
mindspore::Graph graph;
Serialization::Load(mindir_path, ModelType::kMindIR, &graph);
// Build a model using a graph.
ms::Model model;
model.Build(ms::GraphCell(graph), context);
```

Obtain the input information required by the model.

```c++
std::vector<ms::MSTensor> model_inputs = model->GetInputs();
```

Construct network inputs.

```c++
std::vector<MSTensor> inputs;
float *dummy_data = new float[1*3*224*224];
inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                      dummy_data, 1*3*224*224*sizeof(float));
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
ret = model.Predict(inputs, &outputs);
```

### Introducing Building Script

Add the header file search path for the compiler:

```cmake
option(MINDSPORE_PATH "mindspore install path" "")
include_directories(${MINDSPORE_PATH})
include_directories(${MINDSPORE_PATH}/include)
```

Search for the required dynamic library in MindSpore.

```cmake
find_library(MS_LIB libmindspore.so ${MINDSPORE_PATH}/lib)
```

Use the specified source file to generate the target executable file and link the target file to the MindSpore library.

```cmake
add_executable(main src/main.cc)
target_link_libraries(main ${MS_LIB})
```

>For details, see [CMakeLists.txt](https://gitee.com/mindspore/docs/blob/r1.9/docs/sample_code/gpu_resnet50_inference_sample/CMakeLists.txt)

### Building Inference Code

Next compile the inference code, and go to the project directory `gpu_resnet50_inference_sample`:

According to the actual situation, the `pip3` in the build.sh can be modified, and the `bash build.sh` command can be compiled after the modification is completed.

```bash
bash build.sh
```

After building, the executable `main` file is generated in `gpu_resnet50_inference_sample/out`.

### Performing Inference and Viewing the Result

After completing the preceding operations, you can learn how to perform inference.

Log in to the GPU environment, and create the `model` directory to store the `resnet50_imagenet.mindir` file, for example, `/home/mindspore_sample/gpu_resnet50_inference_sample/model`.

Set the environment variable base on the actual situation, where the TensorRT is an optional configuration item. It is recommended to add `TensorRT` path to `LD_LIBRARY_PATH` to improve mode inference performance.

```bash
export LD_PRELOAD=/home/miniconda3/lib/libpython37m.so
export LD_LIBRARY_PATH=/usr/local/TensorRT-7.2.2.3/lib/:$LD_LIBRARY_PATH
```

Then, perform inference.

```bash
cd out/
./main ../model/resnet50_imagenet.mindir 1000 10
```

In the current test script, we printed the inference delay and average delay for each step:

```text
Start to load model..
Load model successuflly
Start to warmup..
Warmup finished
Start to infer..
step 0 cost 1.54004ms
step 1 cost 1.5271ms
... ...
step 998 cost 1.30688ms
step 999 cost 1.30493ms
infer finished.
=================Average inference time: 1.35195 ms
```

### Notices

- During the training process, some networks set operator precision to FP16 artificially. For example, the [Bert mode](https://gitee.com/mindspore/models/blob/r1.9/official/nlp/bert/src/bert_model.py) set the Dense and LayerNorm to FP16:

```python
class BertOutput(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 initializer_range=0.02,
                 dropout_prob=0.1,
                 compute_type=mstype.float32):
        super(BertOutput, self).__init__()
        # Set the nn.Dense to fp16.
        self.dense = nn.Dense(in_channels, out_channels,
                              weight_init=TruncatedNormal(initializer_range)).to_float(compute_type)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.dropout_prob = dropout_prob
        self.add = P.Add()
        # Set the nn.LayerNorm to fp16.
        self.layernorm = nn.LayerNorm((out_channels,)).to_float(compute_type)
        self.cast = P.Cast()
        ... ...
```

It is recommended that export the MindIR model with fp32 precision mode before deploying inference. If you want to further improve the inference performance, the inference precision can be set to FP16 through `mindspore::GPUDeviceInfo::SetPrecisionMode ("fp16")`,and the framework automatically selects the operator inference with the better performance.

- Some inference scripts may introduce some unique network structures in the training process. For example, the model requires the image label, which are transmitted to the network output directly. It is suggested to delete this part of operators and then export MindIR model to improve inference performance.

## Inference by Using an ONNX File

1. Generate a model in ONNX format on the training platform. For details, see [Export ONNX Model](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore/mindspore.export.html#mindspore.export).

2. Perform inference on a GPU by referring to the runtime or SDK document. For example, use TensorRT to perform inference on the Nvidia GPU. For details, see [TensorRT backend for ONNX](https://github.com/onnx/onnx-tensorrt).
