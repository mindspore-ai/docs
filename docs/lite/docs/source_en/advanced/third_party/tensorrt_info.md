# TensorRT Integration Information

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/advanced/third_party/tensorrt_info.md)

## Steps

### Environment Preparation

Besides basic [Environment Preparation](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html), CUDA and TensorRT is required as well. Current version supports [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) and [TensorRT 6.0.1.5](https://developer.nvidia.com/nvidia-tensorrt-6x-download), and [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.1-download-archive) and [TensorRT 8.5.1](https://developer.nvidia.com/nvidia-tensorrt-8x-download).

Install the appropriate version of CUDA and set the installed directory as environment variable `${CUDA_HOME}`. Our build script uses this environment variable to seek CUDA.

Install TensorRT of the corresponding CUDA version, and set the installed directory as environment viriable `${TENSORRT_PATH}`. Our build script uses this environment viriable to seek TensorRT.

### Build

In the Linux environment, use the build.sh script in the root directory of MindSpore [Source Code](https://gitee.com/mindspore/mindspore) to build the MindSpore Lite package integrated with TensorRT. First configure the environment variable `MSLITE_GPU_BACKEND=tensorrt`, and then execute the compilation command as follows.

```bash
bash build.sh -I x86_64
```

For more information about compilation, see [Linux Environment Compilation](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html#linux-environment-compilation).

### Integration

- Integration instructions

    When developers need to integrate the use of TensorRT features, it is important to note:
    - [Configure the TensorRT backend](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/infer/runtime_cpp.html#configuring-the-gpu-backend),
    For more information about using Runtime to perform inference, see [Using Runtime to Perform Inference (C++)](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/infer/runtime_cpp.html).

    - Compile and execute the binary. If you use dynamic linking, please refer to [Compilation Output](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html#directory-structure) with compilation option `-I x86_64`.
    Please set environment variables to dynamically link related libs.

    ```bash
    export LD_LIBRARY_PATH=mindspore-lite-{version}-{os}-{arch}/runtime/lib/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=user-installed-tensorrt-path/lib/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=user-installed-cuda-path/lib/:$LD_LIBRARY_PATH
    ```

- Using Benchmark testing TensorRT inference

    Pass the build package to a device with a TensorRT environment(TensorRT 6.0.1.5) and use the Benchmark tool to test TensorRT inference. Examples are as follows:

    - Test performance

    ```bash
    ./benchmark --device=GPU --modelFile=./models/test_benchmark.ms --timeProfiling=true
    ```

    - Test precision

    ```bash
    ./benchmark --device=GPU --modelFile=./models/test_benchmark.ms --inDataFile=./input/test_benchmark.bin --inputShapes=1,32,32,1 --accuracyThreshold=3 --benchmarkDataFile=./output/test_benchmark.out
    ```

    For more information about the use of Benchmark, see [Benchmark Use](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/tools/benchmark.html).

    For environment variable settings, you need to set the directory where the `libmindspore-lite.so`
    (under the directory `mindspore-lite-{version}-{os}-{arch}/runtime/lib`), TensorRT and CUDA `so` libraries are located, to `${LD_LIBRARY_PATH}`.

- Using TensorRT engine serialization

    TensorRT backend inference supports serializing the built TensorRT model (Engine) into a binary file and saves it locally. When it is used the next time, the model can be deserialized and loaded from the local, avoiding rebuilding and reducing overhead. To support this function, users need to use the [LoadConfig](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html) interface to load the configuration file in the code, you need to specify the saving path of serialization file in the configuration file:

    ```
    [ms_cache]
    serialize_path=/path/to/config
    ```

- Using TensorRT dynamic shapes

    By default, TensorRT optimizes the model based on the input shapes (batch size, image size, and so on) at which it was defined. However, the input dimension can be adjusted at runtime by configuring the profile. In the profile, the minimum, dynamic and optimal shape of each input can be set.

    TensorRT creates an optimized engine for each profile, choosing CUDA kernels that work for all shapes within the [minimum ~ maximum] range. And in the profile, multiple input dimensions can be configured for a single input. To support this function, users need to use the [LoadConfig](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html) interface to load the configuration file in the code.

    If min, opt, and Max are the minimum, optimal, and maximum dimensions, and real_shape is the shape of the input tensor, the following conditions must hold:

    1. `len(min)` == `len(opt)` == `len(max)` == `len(real_shape)`
    2. 0 < `min[i]` <= `opt[i]` <= `max[i]` for all `i`
    3. if `real_shape[i]` != -1, then `min[i]` == `opt[i]` == `max[i]` == `real_shape[i]`
    4. When using tensor input without dynamic dimensions, all shapes must be equal to real_shape.

    For example, if the model input1's name is "input_name1", its input shape is [3,-1,-1] (-1 means that this dimension supports dynamic shape), the minimum dimension is [3,100,200], the maximum dimension is [3,200,300], and the optimized dimension is [3,150,250]. The name of model input2 is "input_name2", the input dimension is [-1,-1,1], the minimum size is [700,800,1], the maximum size is [800,900,1], and the optimized size is [750,850,1]. The following configuration file needs to be configured:

    ```
    [gpu_context]
    input_shape=input_name1:[3,-1,-1];input_name2:[-1,-1,1]
    dynamic_dims=[100~200,200~300];[700~800,800~900]
    opt_dims=[150,250];[750,850]
    ```

    It also supports configuring multiple profiles at the same time. According to the above example, if we add a profile configuration for each model input, for the input1, the minimum size of the added profile is [3,201,200], the maximum size is [3,150,300], and the optimized size is [3,220,250]. Add a profile for input2, whose minimum size is [801,800,1], maximum size is [850,900,1], and optimized size is [810,850,1]. The following is an example of the profile:

    ```
    [gpu_context]
    input_shape=input_name1:[3,-1,-1];input_name2:[-1,-1,1]
    dynamic_dims=[100~200,200~300],[201~250,200~300];[700~800,800~900],[801~850,800~900]
    opt_dims=[150,250],[220,250];[750,850],[810,850]
    ```

## Supported Operators

For supported TensorRT operators, see [Lite Operator List](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/reference/operator_list_lite.html).
