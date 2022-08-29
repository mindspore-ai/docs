# TensorRT Integration Information

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/use/tensorrt_info.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Steps

### Environment Preparation

Besides basic [Environment Preparation](https://www.mindspore.cn/lite/docs/en/master/use/build.html), CUDA and TensorRT is required as well. Current version only supports [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) and [TensorRT 6.0.1.5](https://developer.nvidia.com/nvidia-tensorrt-6x-download), and [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.1-download-archive) and [TensorRT 8.2.5](https://developer.nvidia.com/nvidia-tensorrt-8x-download).

Install the appropriate version of CUDA and set the installed directory as environment variable `${CUDA_HOME}`. Our build script uses this environment variable to seek CUDA.

Install TensorRT of the corresponding CUDA version, and set the installed directory as environment viriable `${TENSORRT_PATH}`. Our build script uses this environment viriable to seek TensorRT.

### Build

In the Linux environment, use the build.sh script in the root directory of MindSpore [Source Code](https://gitee.com/mindspore/mindspore) to build the MindSpore Lite package integrated with TensorRT. First configure the environment variable `MSLITE_GPU_BACKEND=tensorrt`, and then execute the compilation command as follows.

```bash
bash build.sh -I x86_64
```

For more information about compilation, see [Linux Environment Compilation](https://www.mindspore.cn/lite/docs/en/master/use/build.html#linux-environment-compilation).

### Integration

- Integration instructions

    When developers need to integrate the use of TensorRT features, it is important to note:
    - [Configure the TensorRT backend](https://www.mindspore.cn/lite/docs/en/master/use/runtime_cpp.html#configuring-the-gpu-backend),
    For more information about using Runtime to perform inference, see [Using Runtime to Perform Inference (C++)](https://www.mindspore.cn/lite/docs/en/master/use/runtime_cpp.html).

    - Compile and execute the binary. If you use dynamic linking, please refer to [Compilation Output](https://www.mindspore.cn/lite/docs/en/master/use/build.html#directory-structure) with compilation option `-I x86_64`.
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

    For more information about the use of Benchmark, see [Benchmark Use](https://www.mindspore.cn/lite/docs/en/master/use/benchmark.html).

    For environment variable settings, you need to set the directory where the `libmindspore-lite.so`
    (under the directory `mindspore-lite-{version}-{os}-{arch}/runtime/lib`), TensorRT and CUDA `so` libraries are located, to `${LD_LIBRARY_PATH}`.

- Using TensorRT engine serialization

    TensorRT backend inference supports serializing the built TensorRT model (Engine) into a binary file and saves it locally. When it is used the next time, the model can be deserialized and loaded from the local, avoiding rebuilding and reducing overhead. To support this function, users need to use the [LoadConfig](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html) interface to load the configuration file in the code, you need to specify the saving path of serialization file in the configuration file:

    ```
    [ms_cache]
    serialize_path=/path/to/config
    ```

- Using TensorRT dynamic shapes

    By default, TensorRT optimizes the model based on the input shapes (batch size, image size, and so on) at which it was defined. However, the builder can be configured to allow the input dimensions to be adjusted at runtime. In the profile, the maximum, minimum and optimal shape of each input can be set.

    TensorRT creates an optimized engine for each profile, choosing CUDA kernels that work for all shapes within the [minimum, maximum] range and are fastest for the optimization point. Multiple profiles should specify disjoint or overlapping ranges. To support this function, users need to use the [LoadConfig](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html) interface to load the configuration file in the code.

    If min, opt, and Max are the minimum, optimal, and maximum dimensions, and real_shape is the shape of the input tensor, then the following conditions must hold:

    1. `len(min)` == `len(opt)` == `len(max)` == `len(real_shape)`
    2. 0 <= `min[i]` <= `opt[i]` <= `max[i]` for all `i`
    3. if `real_shape[i]` != -1, then `min[i]` == `opt[i]` == `max[i]` == `real_shape[i]`
    4. When using tensor input without dynamic dimensions, all shapes must be equal to real_shape.

    For example, if the minimum dimension is [3,100,200], the maximum dimension is [3,200,300], and the optimized dimension is [3,150,250], the following configuration file needs to be configured:

    ```
    [input_ranges]
    min_dims:3,100,200
    opt_dims:3,150,250
    max_dims:3,200,300
    ```

## Supported Operators

For supported TensorRT operators, see [Lite Operator List](https://www.mindspore.cn/lite/docs/en/master/operator_list_lite.html).
