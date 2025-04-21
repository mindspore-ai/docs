# NPU Integration Information

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/advanced/third_party/npu_info.md)

## Steps

### Environment Preparation

Besides basic [Environment Preparation](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html), HUAWEI HiAI DDK, which contains
APIs (including building, loading models and calculation processes) and interfaces implemented to encapsulate dynamic libraries (namely libhiai*.so),
is required for the use of NPU. Download [DDK 100.510.010.010](https://developer.huawei.com/consumer/en/doc/development/hiai-Library/ddk-download-0000001053590180),
and set the directory of extracted files as `${HWHIAI_DDK}`. Our build script uses this environment viriable to seek DDK.

### Build

Under the Linux operating system, one can easily build MindSpore Lite Package integrating NPU interfaces and libraries using build.sh under
the root directory of MindSpore [Source Code](https://gitee.com/mindspore/mindspore). The command is as follows.
It will build MindSpore Lite's package under the output directory under the MindSpore source code root directory,
which contains the NPU's dynamic library, the libmindspore-lite dynamic library, and the test tool Benchmark.

```bash
export MSLITE_ENABLE_NPU=ON
bash build.sh -I arm64 -j8
```

For more information about compilation, see [Linux Environment Compilation](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html#linux-environment-compilation).

### Integration

- Integration instructions

    When developers need to integrate the use of NPU features, it is important to note:

    - [Configure the NPU backend](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/infer/runtime_cpp.html#configuring-the-npu-backend).
     For more information about using Runtime to perform inference, see [Using Runtime to Perform Inference (C++)](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/infer/runtime_cpp.html).

    - Compile and execute the binary. If you use dynamic linking, refer to [compile output](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html) when the compile option is `-I arm64` or `-I arm32`.
    Configured environment variables will dynamically load libhiai.so, libhiai_ir.so, libhiai_ir_build.so, libhiai_hcl_model_runtime.so. For example,

        ```bash
        export LD_LIBRARY_PATH=mindspore-lite-{version}-android-{arch}/runtime/third_party/hiai_ddk/lib/:$LD_LIBRARY_PATH
        ```

- Using Benchmark testing NPU inference

    Users can also test NPU inference using MindSpore Lite's Benchmark tool. Pass the build package to the `/data/local/tmp/` directory of an Android phone equipped with NPU chips and test NPU inference using the Benchmark tool on the phone, as shown in the example below:

    - Test performance

    ```bash
    ./benchmark --device=NPU --modelFile=./models/test_benchmark.ms --timeProfiling=true
    ```

    - Test precision

    ```bash
    ./benchmark --device=NPU --modelFile=./models/test_benchmark.ms --inDataFile=./input/test_benchmark.bin --inputShapes=1,32,32,1 --accuracyThreshold=3 --benchmarkDataFile=./output/test_benchmark.out
    ```

For more information about the use of Benchmark, see [Benchmark Use](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/tools/benchmark_tool.html).

For environment variable settings, you need to set the directory where the libmindspore-lite.so
(under the directory `mindspore-lite-{version}-android-{arch}/runtime/lib`) and NPU libraries
(under the directory `mindspore-lite-{version}-android-{arch}/runtime/third_party/hiai_ddk/lib/`) are located, to `${LD_LIBRARY_PATH}`.

## Supported Chips

For supported NPU chips, see [Chipset Platforms and Supported HUAWEI HiAI Versions](https://developer.huawei.com/consumer/en/doc/development/hiai-Guides/supported-platforms-0000001052830507#section94427279718).

## Supported Operators

For supported NPU operators, see [Lite Operator List](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/reference/operator_list_lite.html).