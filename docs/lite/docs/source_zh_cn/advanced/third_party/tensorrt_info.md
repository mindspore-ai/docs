# 集成TensorRT使用说明

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/advanced/third_party/tensorrt_info.md)

## 使用步骤

### 环境准备

在基本的[环境准备](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html)之外，使用TensorRT需要集成CUDA、TensorRT。当前版本适配[CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) 和[TensorRT 6.0.1.5](https://developer.nvidia.com/nvidia-tensorrt-6x-download) 以及 [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.1-download-archive) 和 [TensorRT 8.5.1](https://developer.nvidia.com/nvidia-tensorrt-8x-download).

安装相应版本的CUDA，并将安装后的目录设置为环境变量`${CUDA_HOME}`。构建脚本将使用这个环境变量寻找CUDA。

下载对应版本的TensorRT压缩包，并将压缩包解压后的目录设置为环境变量`${TENSORRT_PATH}`。构建脚本将使用这个环境变量寻找TensorRT。

### 编译构建

在Linux环境下，使用MindSpore[源代码](https://gitee.com/mindspore/mindspore)根目录下的build.sh脚本可以构建集成TensorRT的MindSpore Lite包，先配置环境变量`MSLITE_GPU_BACKEND=tensorrt`，再执行编译命令如下，它将在MindSpore源代码根目录下的output目录下构建出MindSpore Lite的包，其中包含`libmindspore-lite.so`以及测试工具Benchmark。

```bash
bash build.sh -I x86_64
```

有关编译详情见[Linux环境编译](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html#linux环境编译)。

### 集成使用

- 集成说明

    开发者需要集成使用TensorRT功能时，需要注意：
    - 在代码中[配置TensorRT后端](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/infer/runtime_cpp.html#配置使用gpu后端)，有关使用Runtime执行推理详情见[使用Runtime执行推理（C++）](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/infer/runtime_cpp.html)。
    - 编译执行可执行程序。如采用动态加载方式，参考[编译输出](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html)中编译选项为`-I x86_64`时的内容，需要配置的环境变量如下，将会动态加载相关的so。

    ```bash
    export LD_LIBRARY_PATH=mindspore-lite-{version}-{os}-{arch}/runtime/lib/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=user-installed-tensorrt-path/lib/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=user-installed-cuda-path/lib/:$LD_LIBRARY_PATH
    ```

- Benchmark测试TensorRT推理

    用户也可以使用MindSpore Lite的Benchmark工具测试TensorRT推理。编译出的Benchmark位置见[编译输出](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html)。将构建包传到具有TensorRT环境（TensorRT 6.0.1.5）的设备上，使用Benchmark工具测试TensorRT推理，示例如下：

    - 测性能

    ```bash
    ./benchmark --device=GPU --modelFile=./models/test_benchmark.ms --timeProfiling=true
    ```

    - 测精度

    ```bash
    ./benchmark --device=GPU --modelFile=./models/test_benchmark.ms --inDataFile=./input/test_benchmark.bin --inputShapes=1,32,32,1 --accuracyThreshold=3 --benchmarkDataFile=./output/test_benchmark.out
    ```

    有关Benchmark使用详情，见[Benchmark使用](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/tools/benchmark.html)。

    有关环境变量设置，需要根据[编译输出](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html)中编译选项为`-I x86_64`时的目录结构，将`libmindspore-lite.so`（目录为`mindspore-lite-{version}-{os}-{arch}/runtime/lib`）、CUDA的`so`库所在的目录和TensorRT的`so`库所在的目录加入`${LD_LIBRARY_PATH}`。

- 模型序列化

    TensorRT推理支持将已构建的TensorRT模型（Engine）序列化为二进制文件保存在本地，下次使用时即可从本地反序列化加载模型，避免重新构建，降低开销。支持此功能，用户需要在代码中使用[LoadConfig](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#loadconfig)接口加载配置文件，配置文件中须指定序列化文件保存路径：

    ```
    [ms_cache]
    serialize_path=/path/to/config
    ```

- 模型动态输入

    默认情况下，TensorRT根据定义模型的输入形状(批大小、图像大小等)优化模型。但是，可以通过配置profile在运行时调整输入维度，在profile中可以设置每个动态输入的最小、动态以及最优形状，TensorRT会根据用户设置的profile创建一个优化引擎，并选择最优最快的内核，并且在profile中支持一个输入配置多个输入维度。支持此功能，用户需要在代码中使用[LoadConfig](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#loadconfig)接口加载配置文件。

    如果min、opt 和 max 是最小、最优和最大维度，并且real_shape是输入张量的形状，则以下条件必须成立：

    1. `len(min)` == `len(opt)` == `len(max)` == `len(real_shape)`
    2. 0 < `min[i]` <= `opt[i]` <= `max[i]` for all `i`
    3. if `real_shape[i]` != -1, then `min[i]` == `opt[i]` == `max[i]` == `real_shape[i]`
    4. 在使用没有动态维度的张量输入时，所有形状必须等于real_shape。

    比如模型输入1的名字为"input_name1"，其输入维度为[3,-1,-1]（-1代表该维度支持动态变化），最小尺寸为[3,100,200]，最大尺寸为[3,200,300]，优化尺寸为[3,150,250]；模型输入2的名字为"input_name2"，其输入维度为[-1,-1，1]，最小尺寸为[700,800,1]，最大尺寸为[800,900,1]，优化尺寸为[750,850,1]，则配置文件中需配置为：

    ```
    [gpu_context]
    input_shape=input_name1:[3,-1,-1];input_name2:[-1,-1,1]
    dynamic_dims=[100~200,200~300];[700~800,800~900]
    opt_dims=[150,250];[750,850]
    ```

    同时可支持多profile的配置，若配置多个profile，根据上述的例子，增加一个profile的配置。增加输入1的profile的最小尺寸为[3,201,200]，最大尺寸为[3,150,300]，优化尺寸为[3,220,250]；增加输入2的profile，其最小尺寸为[801,800,1]，最大尺寸为[850,900,1]，优化尺寸为[810,850,1]，配置文件样例如下：

    ```
    [gpu_context]
    input_shape=input_name1:[3,-1,-1];input_name2:[-1,-1,1]
    dynamic_dims=[100~200,200~300],[201~250,200~300];[700~800,800~900],[801~850,800~900]
    opt_dims=[150,250],[220,250];[750,850],[810,850]
    ```

## 算子支持

TensorRT算子支持见[Lite 算子支持](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/reference/operator_list_lite.html)。
