# DSP Integration Information

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/advanced/third_party/dsp_info.md)

## Steps

### Environment Preparation

Besides basic [Environment Preparation](https://www.mindspore.cn/lite/docs/en/master/use/build.html), Using DSP requires the integration of dsp_sdk. Dsp_sdk includes heterogeneous programming interfaces using DSP and interface implementations encapsulated into static libraries (named libhthread_host.a). Set the dsp_sdk directory as the environment variable `${dsp_sdk_path}`, and the build script will use this environment variable to find dsp_sdk; In addition, a cross compilation tool is required. The installation command is as follows:

```bash
sudo apt-get update && apt-get install -y --no-install-recommends \
  g++-arm-linux-gnueabihf gcc-arm-linux-gnueabihf
```

### Build

Under the Linux operating system, one can easily build MindSpore Lite Package integrating DSP interfaces and libraries using build.sh under the root directory of MindSpore [Source Code](https://gitee.com/mindspore/mindspore-lite). The command is as follows.
It will build MindSpore Lite's package under the output directory under the MindSpore source code root directory, which contains the libmindspore-lite dynamic library, and the test tool Benchmark.

```bash
export MSLITE_REGISTRY_DEVICE=ft78
export DSP_SDK_PATH=${your path}/dsp_sdk
export MSLITE_ENABLE_TESTCASES=ON
export MSLITE_ENABLE_TOOLS=ON
bash build.sh -I arm32 -j8
```

Where `${your path}/dsp_sdk` is the path of dsp_sdk. `MSLITE_REGISTRY_DEVICE` has two options: `ft78` and `ft04`, which correspond to different DSP chips respectively.
For more information about compilation, see [Linux Environment Compilation](https://www.mindspore.cn/lite/docs/en/master/use/build.html#linux-environment-compilation).

### Integration

- Integration instructions

    When developers need to integrate the use of DSP features, it is important to note:

    - [Configure the DSP backend](https://www.mindspore.cn/lite/docs/en/master/infer/runtime_cpp.html#configuring-the-dsp-backend).
      For more information about using Runtime to perform inference, see [Using Runtime to Perform Inference (C++)](https://www.mindspore.cn/lite/docs/en/master/infer/runtime_cpp.html).

- Using Benchmark testing DSP inference

    Users can test DSP inference using MindSpore Lite's Benchmark tool. Copy the benchmark tool to the `ft78` or `ft04` device, An example of using the benchmark tool on a device is as follows:

    - Test performance

    ```bash
    ./benchmark --device=DSP --modelFile=./models/test_benchmark.ms --timeProfiling=true
    ```

For more information about the use of Benchmark, see [Benchmark Use](https://www.mindspore.cn/lite/docs/en/master/tools/benchmark_tool.html).

For environment variable settings, copy libmindspore-lite.so to the `/usr/lib` directory of `ft78` or `ft04`.

## Supported Chips

The DSP chip supports `ft04` and `ft78`.

## Supported Operators

For supported DSP operators, see [Lite Operator List](https://www.mindspore.cn/lite/docs/en/master/reference/operator_list_lite.html).