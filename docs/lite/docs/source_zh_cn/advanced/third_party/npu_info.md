# 集成NPU使用说明

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/advanced/third_party/npu_info.md)

## 使用步骤

### 环境准备

在基本的[环境准备](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html)之外，使用NPU需要集成HUAWEI HiAI DDK。
DDK包含了使用NPU的对外接口（包括模型构建、加载，计算等），以及封装成动态库的接口实现（名为libhiai*.so）。
下载[DDK 100.510.010.010版本](https://developer.huawei.com/consumer/cn/doc/development/hiai-Library/ddk-download-0000001053590180)，
并将压缩包解压后的目录设置为环境变量`${HWHIAI_DDK}`。构建脚本将使用这个环境变量寻找DDK。

### 编译构建

在Linux环境下，使用MindSpore[源代码](https://gitee.com/mindspore/mindspore)根目录下的build.sh脚本可以构建集成NPU的MindSpore Lite包，命令如下，它将在MindSpore源代码根目录下的output目录下构建出MindSpore Lite的包，其中包含NPU的动态库，libmindspore-lite动态库以及测试工具Benchmark。

```bash
export MSLITE_ENABLE_NPU=ON
bash build.sh -I arm64 -j8
```

有关编译详情见[Linux环境编译](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html#linux环境编译)。

### 集成使用

- 集成说明

    开发者需要集成使用NPU功能时，需要注意：
    - 在代码中[配置NPU后端](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/infer/runtime_cpp.html#配置使用npu后端)，有关使用Runtime执行推理详情见[使用Runtime执行推理（C++）](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/infer/runtime_cpp.html)。
    - 编译执行可执行程序。如采用动态加载方式，参考[编译输出](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html)中编译选项为`-I arm64`或`-I arm32`时的内容，配置好环境变量，将会动态加载libhiai.so、libhiai_ir.so、libhiai_ir_build.so、libhiai_hcl_model_runtime.so。例如：

    ```bash
    export LD_LIBRARY_PATH=mindspore-lite-{version}-android-{arch}/runtime/third_party/hiai_ddk/lib/:$LD_LIBRARY_PATH
    ```

- Benchmark测试NPU推理

    用户也可以使用MindSpore Lite的Benchmark工具测试NPU推理。将构建包传到具有NPU芯片的Android手机的`/data/local/tmp/`目录下，在手机上使用Benchmark工具测试NPU推理，示例如下：

    - 测性能

    ```bash
    ./benchmark --device=NPU --modelFile=./models/test_benchmark.ms --timeProfiling=true
    ```

    - 测精度

    ```bash
    ./benchmark --device=NPU --modelFile=./models/test_benchmark.ms --inDataFile=./input/test_benchmark.bin --inputShapes=1,32,32,1 --accuracyThreshold=3 --benchmarkDataFile=./output/test_benchmark.out
    ```

有关Benchmark使用详情，见[Benchmark使用](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/tools/benchmark_tool.html)。

有关环境变量设置，将libmindspore-lite.so（目录为`mindspore-lite-{version}-android-{arch}/runtime/lib`）和
NPU库（目录为`mindspore-lite-{version}-android-{arch}/runtime/third_party/hiai_ddk/lib/`）所在的目录加入`${LD_LIBRARY_PATH}`。

## 芯片支持

NPU芯片支持见[芯片与HUAWEI HiAI Version版本映射关系](https://developer.huawei.com/consumer/cn/doc/development/hiai-Guides/supported-platforms-0000001052830507#section94427279718)。

## 算子支持

NPU算子支持见[Lite 算子支持](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/reference/operator_list_lite.html)。
