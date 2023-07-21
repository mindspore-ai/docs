# Inference on the Ascend 910 AI processor

`Linux` `Ascend` `Inference Application` `Beginner` `Intermediate` `Expert`

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/programming_guide/source_en/multi_platform_inference_ascend_910.md)

## Inference Using a Checkpoint File with Single Device

1. Use the `model.eval` interface for model validation.

   1.1 Local Storage

     When the pre-trained models are saved in local, the steps of performing inference on validation dataset are as follows: firstly creating a model, then loading the model and parameters using `load_checkpoint` and `load_param_into_net` in `mindspore.train.serialization` module, and finally performing inference on the validation dataset once being created. The method of processing the validation dataset is the same as that of the training dataset.

    ```python
    network = LeNet5(cfg.num_classes)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)
    dataset = create_dataset(os.path.join(args.data_path, "test"),
                             cfg.batch_size,
                             1)
    acc = model.eval(dataset, dataset_sink_mode=args.dataset_sink_mode)
    print("============== {} ==============".format(acc))
    ```

    In the preceding information:  
    `model.eval` is an API for model validation. For details about the API, see <https://www.mindspore.cn/docs/api/en/r1.3/api_python/mindspore.html#mindspore.Model.eval>.
    > Inference sample code: <https://gitee.com/mindspore/mindspore/blob/r1.3/model_zoo/official/cv/lenet/eval.py>.

    1.2 Remote Storage

    When the pre-trained models are saved remotely, the steps of performing inference on the validation dataset are as follows: firstly determining which model to be used, then loading the model and parameters using `mindspore_hub.load`, and finally performing inference on the validation dataset once being created. The method of processing the validation dataset is the same as that of the training dataset.

    ```python
    model_uid = "mindspore/ascend/0.7/googlenet_v1_cifar10"  # using GoogleNet as an example.
    network = mindspore_hub.load(model_uid, num_classes=10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Testing ==============")
    dataset = create_dataset(os.path.join(args.data_path, "test"),
                             cfg.batch_size,
                             1)
    acc = model.eval(dataset, dataset_sink_mode=args.dataset_sink_mode)
    print("============== {} ==============".format(acc))
    ```

    In the preceding information:

    `mindspore_hub.load` is an API for loading model parameters. Please check the details in <https://www.mindspore.cn/hub/api/en/r1.3/index.html#module-mindspore_hub>.

2. Use the `model.predict` API to perform inference.

   ```python
   model.predict(input_data)
   ```

   In the preceding information:  
   `model.predict` is an API for inference. For details about the API, see <https://www.mindspore.cn/docs/api/en/r1.3/api_python/mindspore.html#mindspore.Model.predict>.

## Use C++ Interface to Load a MindIR File for Inferencing

Users can create C++ applications and call MindSpore C++ interface to inference MindIR models.

### Inference Directory Structure

Create a directory to store the inference code project, for example, `/home/HwHiAiUser/mindspore_sample/ascend910_resnet50_preprocess_sample`. The directory code can be obtained from the [official website](https://gitee.com/mindspore/docs/tree/r1.3/docs/sample_code/ascend910_resnet50_preprocess_sample). The `model` directory stores the exported `MindIR` model files and the `test_data` directory stores the images to be classified. The directory structure of the inference code project is as follows:

```text
└─ascend910_resnet50_preprocess_sample
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

### Inference Code

Inference sample code: <https://gitee.com/mindspore/docs/blob/r1.3/docs/sample_code/ascend310_resnet50_preprocess_sample/main.cc> .

Using namespace of `mindspore` and `mindspore::dataset`.

```c++
namespace ms = mindspore;
namespace ds = mindspore::dataset;
```

Set global context, device target is `Ascend910` and evice id is `0`:

```c++
auto context = std::make_shared<ms::Context>();
auto ascend910_info = std::make_shared<ms::Ascend910DeviceInfo>();
ascend910_info->SetDeviceID(0);
context->MutableDeviceInfo().push_back(ascend910_info);
```

Load mindir file:

```c++
// Load MindIR model
ms::Graph graph;
ms::Status ret = ms::Serialization::Load(resnet_file, ms::ModelType::kMindIR, &graph);
// Build model with graph object
ms::Model resnet50;
ret = resnet50.Build(ms::GraphCell(graph), context);
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

### Introduce to Building Script

The building script is used to building applications: <https://gitee.com/mindspore/docs/blob/r1.3/docs/sample_code/ascend910_resnet50_preprocess_sample/CMakeLists.txt>.

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

### Building Inference Code

Go to the project directory `ascend910_resnet50_preprocess_sample` and set the following environment variables:

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64/common:${LOCAL_ASCEND}/driver/lib64/driver:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# lib libraries that the mindspore depends on, modify "pip3" according to the actual situation
export LD_LIBRARY_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore/lib"}' | xargs realpath`:${LD_LIBRARY_PATH}

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}                 # TBE operator compilation tool path
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

After building, the executable `main` file is generated in `ascend910_resnet50_preprocess_sample`.

### Performing Inference and Viewing the Result

Log in to the Ascend 910 server, and create the `model` directory for storing the MindIR file `resnet50_imagenet.mindir`, for example, `/home/HwHiAiUser/mindspore_sample/ascend910_resnet50_preprocess_sample/model`.
Create the `test_data` directory to store images, for example, `/home/HwHiAiUser/mindspore_sample/ascend910_resnet50_preprocess_sample/test_data`.
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
