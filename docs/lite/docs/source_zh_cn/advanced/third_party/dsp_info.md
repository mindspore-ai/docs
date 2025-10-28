# 集成DSP使用说明

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/advanced/third_party/dsp_info.md)

## 使用步骤

### 环境准备

在基本的[环境准备](https://www.mindspore.cn/lite/docs/zh-CN/master/build/build.html)之外，使用DSP需要集成dsp_sdk。dsp_sdk包含了使用DSP的异构编程接口，以及封装成静态库的接口实现（名为libhthread_host.a）。将dsp_sdk目录设置为环境变量`${DSP_SDK_PATH}`，构建脚本将使用这个环境变量寻找dsp_sdk。此外还需要交叉编译工具，安装命令如下：

```bash
sudo apt-get update && apt-get install -y --no-install-recommends \
  g++-arm-linux-gnueabihf gcc-arm-linux-gnueabihf
```

### 编译构建

在Linux环境，执行MindSpore Lite[源代码](https://gitee.com/mindspore/mindspore-lite)根目录下的build.sh脚本，将在MindSpore Lite源代码根目录下的output文件夹构建出集成DSP的MindSpore Lite包，其中包含libmindspore-lite动态库以及测试工具Benchmark。命令如下：

```bash
export MSLITE_REGISTRY_DEVICE=ft78
export DSP_SDK_PATH=${your path}/dsp_sdk
export MSLITE_ENABLE_TESTCASES=ON
export MSLITE_ENABLE_TOOLS=ON
bash build.sh -I arm32 -j8
```

其中，`${your path}/dsp_sdk`为dsp_sdk的路径。`MSLITE_REGISTRY_DEVICE`有两个选项：`ft78`和`ft04`，分别对应不同的DSP芯片。有关编译详情见[Linux环境编译](https://www.mindspore.cn/lite/docs/zh-CN/master/build/build.html#linux环境编译)。

### 集成使用

- 集成说明

    开发者集成DSP功能时，需要在代码中[配置DSP后端](https://www.mindspore.cn/lite/docs/zh-CN/master/infer/runtime_cpp.html#配置使用dsp后端)，相关使用方法可以参考[使用Runtime执行推理（C++）](https://www.mindspore.cn/lite/docs/zh-CN/master/infer/runtime_cpp.html)。

- Benchmark测试DSP推理

    用户可以使用MindSpore Lite的Benchmark工具测试DSP推理性能。将Benchmark工具拷贝到`ft78`或者`ft04`设备上，执行如下命令：

    ```bash
    ./benchmark --device=DSP --modelFile=./models/test_benchmark.ms --timeProfiling=true
    ```

有关Benchmark使用详情，见[Benchmark使用](https://www.mindspore.cn/lite/docs/zh-CN/master/tools/benchmark_tool.html)。

有关环境变量设置，将libmindspore-lite.so拷贝到`ft78`或`ft04`设备的/usr/lib目录即可。

## 芯片支持

支持`ft04`和`ft78`两种设备类型。

## 算子支持

DSP算子支持见[Lite 算子支持](https://www.mindspore.cn/lite/docs/zh-CN/master/reference/operator_list_lite.html)。
