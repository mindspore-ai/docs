# 集成TensorRT使用说明

`TensorRT` `NVIDIA` `Linux` `环境准备` `算子支持` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/lite/docs/source_zh_cn/use/tensorrt_info.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 使用步骤

### 环境准备

在基本的[环境准备](https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/build.html)之外，使用TensorRT需要集成CUDA、TensorRT。当前版本适配CUDA 10.1 和 TensorRT 6.0.1.5。

安装[CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base)，并将安装后的目录设置为环境变量`${CUDA_HOME}`。构建脚本将使用这个环境变量寻找CUDA。

下载[TensorRT 6.0.1.5](https://developer.nvidia.com/nvidia-tensorrt-6x-download)，并将压缩包解压后的目录设置为环境变量`${TENSORRT_PATH}`。构建脚本将使用这个环境变量寻找TensorRT。

### 编译构建

在Linux环境下，使用MindSpore[源代码](https://gitee.com/mindspore/mindspore)根目录下的build.sh脚本可以构建集成TensorRT的MindSpore Lite包，先配置环境变量`MSLITE_GPU_BACKEND=tensorrt`，再执行编译命令如下，它将在MindSpore源代码根目录下的output目录下构建出MindSpore Lite的包，其中包含`libmindspore-lite.so`以及测试工具Benchmark。

```bash
bash build.sh -I x86_64
```

有关编译详情见[Linux环境编译](https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/build.html#linux)。

### 集成使用

- 集成说明

    开发者需要集成使用TensorRT功能时，需要注意：
    - 在代码中[配置TensorRT后端](https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/runtime_cpp.html#gpu)，有关使用Runtime执行推理详情见[使用Runtime执行推理（C++）](https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/runtime_cpp.html)。
    - 编译执行可执行程序。如采用动态加载方式，参考[编译输出](https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/build.html#runtime)中编译选项为`-I x86_64`时的内容，需要配置的环境变量如下，将会动态加载相关的so。

    ```bash
    export LD_LIBRARY_PATH=mindspore-lite-{version}-{os}-{arch}/runtime/lib/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=user-installed-tensorrt-path/lib/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=user-installed-cuda-path/lib/:$LD_LIBRARY_PATH
    ```

- Benchmark测试TensorRT推理

    用户也可以使用MindSpore Lite的Benchmark工具测试TensorRT推理。编译出的Benchmark位置见[编译输出](https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/build.html#runtime)。将构建包传到具有TensorRT环境（TensorRT 6.0.1.5）的设备上使用Benchmark工具测试TensorRT推理，示例如下：

    - 测性能

    ```bash
    ./benchmark --device=GPU --modelFile=./models/test_benchmark.ms --timeProfiling=true
    ```

    - 测精度

    ```bash
    ./benchmark --device=GPU --modelFile=./models/test_benchmark.ms --inDataFile=./input/test_benchmark.bin --inputShapes=1,32,32,1 --accuracyThreshold=3 --benchmarkDataFile=./output/test_benchmark.out
    ```

    有关Benchmark使用详情，见[Benchmark使用](https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/benchmark.html)。

    有关环境变量设置，需要根据[编译输出](https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/build.html#runtime)中编译选项为`-I x86_64`时的目录结构，将`libmindspore-lite.so`（目录为`mindspore-lite-{version}-{os}-{arch}/runtime/lib`）、CUDA的`so`库所在的目录和TensorRT的`so`库所在的目录加入`${LD_LIBRARY_PATH}`。

## 算子支持

TensorRT算子支持见[Lite 算子支持](https://www.mindspore.cn/lite/docs/zh-CN/r1.7/operator_list_lite.html)。
