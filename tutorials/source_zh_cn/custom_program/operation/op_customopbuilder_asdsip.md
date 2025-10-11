# CustomOpBuilder通过AsdSipFFTOpRunner接入ASDSIP FFT算子

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.1/tutorials/source_zh_cn/custom_program/operation/op_customopbuilder_asdsip.md)

## 概述

[Ascend Sip Boost (ASDSIP) 算子加速库](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/acce/SiP/SIP_0000.html) 是基于华为Ascend AI处理器，专门为信号处理领域而设计的算子库。

当用户需要使用ASDSIP加速库的FFT类算子，而MindSpore未提供相应算子接口时，用户可以使用自定义算子的方法快速接入使用。

在 [基于CustomOpBuilder的自定义算子](https://www.mindspore.cn/tutorials/zh-CN/r2.7.1/custom_program/operation/op_customopbuilder.html) 中，MindSpore提供了 `PyboostRunner` 方便用户在动态图接入自定义算子。现在针对ASDSIP算子，MindSpore又额外提供了一套`AsdSipFFTOpRunner`用于把ASDSIP FFT算子的调用流程和动态图多级流水封装到一起。

用户基于 [AsdSipFFTOpRunner类](https://www.mindspore.cn/tutorials/zh-CN/r2.7.1/custom_program/operation/cpp_api_for_custom_ops.html#class-asdsipfftoprunner) 对接ASDSIP FFT算子时，仅需要提供`Param`，并调用`Init`接口初始化（即构造`Operation`），再调用`Run`接口即可执行ASDSIP算子。还可以直接调用 [RunAsdSipFFTOp](https://www.mindspore.cn/tutorials/zh-CN/r2.7.1/custom_program/operation/cpp_api_for_custom_ops.html#function-runasdsipfftop)函数一键执行（函数内包含了`Init`和`Run`接口的调用）。

本指南以一个`FftC2C`为例，展示ASDSIP FFT算子的接入流程。完整代码请参阅[代码仓库](https://gitee.com/mindspore/mindspore/blob/v2.7.1/tests/st/graph_kernel/custom/jit_test_files/asdsip_fftc2c.cpp)。

## 安装ASDSIP加速库

[点这里查看安装教程](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/acce/SiP/SIP_0001.html)

安装成功之后，需要激活ASDSIP加速库的环境变量：

```sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh &> /dev/null
```

## FftC2C算子接入

这里使用`ms::pynative::RunAsdSipFFTOp`接入算子，并通过`ms::pynative::PyboostRunner::Call`调用函数接口：

```cpp
#include "ms_extension/api.h"

ms::Tensor InferFFTForward(const ms::Tensor &input) {
  ShapeVector out_tensor_shape(input.shape());
  return ms::Tensor(input.data_type(), out_tensor_shape);
}

ms::Tensor npu_fft(const ms::Tensor &input, int64_t n, int64_t batch_size) {
  ms::pynative::FFTParam param;
  param.fftXSize = n;
  param.fftYSize = 0;
  param.fftType = ms::pynative::asdFftType::ASCEND_FFT_C2C;
  param.direction = ms::pynative::asdFftDirection::ASCEND_FFT_FORWARD;
  param.batchSize = batch_size;
  param.dimType = ms::pynative::asdFft1dDimType::ASCEND_FFT_HORIZONTAL;
  auto output = InferFFTForward(input);
  ms::pynative::RunAsdSipFFTOp("asdFftExecC2C", param, input, output);
  return output;
}

auto pyboost_npu_fft(const ms::Tensor &input, int64_t n, int64_t batch_size) {
  return ms::pynative::PyboostRunner::Call<1>(npu_fft, input, n, batch_size);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("fft", &pyboost_npu_fft, "FFT C2C", pybind11::arg("input"), pybind11::arg("n"),
        pybind11::arg("batch_size"));
}
```

### 1. 推导算子的输出信息

```cpp
ms::Tensor InferFFT1DForward(const ms::Tensor &input) {
  ShapeVector out_tensor_shape(input.shape());
  return ms::Tensor(input.data_type(), out_tensor_shape);
}
```

对于`FftC2C`算子，输出张量的数据类型和输入的一样。推导出输出形状之后，通过`ms::Tensor`构造函数构造一个空的张量。

### 2. 创建并设置算子属性结构体

```cpp
ms::pynative::FFTParam param;
param.fftXSize = n;
param.fftYSize = 0;
param.fftType = ms::pynative::asdFftType::ASCEND_FFT_C2C;
param.direction = ms::pynative::asdFftDirection::ASCEND_FFT_FORWARD;
param.batchSize = batch_size;
param.dimType = ms::pynative::asdFft1dDimType::ASCEND_FFT_HORIZONTAL;
```

### 3. 调用AsdSipFFTOpRunner接口执行算子

```cpp
ms::pynative::PyboostRunner::Call<1>(npu_fft, input, n, batch_size);
```

这是一个模板接口，其等效于：

```cpp
auto runner = std::make_shared<AsdSipFFTOpRunner>("FFTExecC2C");
runner->Init(fft_param);
runner->Run({input}, {output});
```

传入算子名、属性、输入张量、输出张量几个信息，即可调用相应的ASDSIP算子。此接口支持了动态图的多级流水执行流程。

### 4. 通过pybind11将C++函数绑定一个Python函数

```cpp
auto pyboost_npu_fft(const ms::Tensor &input, int64_t n, int64_t batch_size) {
  return ms::pynative::PyboostRunner::Call<1>(npu_fft, input, n, batch_size);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("fft", &pyboost_npu_fft, "FFT C2C", pybind11::arg("input"), pybind11::arg("n"),
        pybind11::arg("batch_size"));
}
```

### 5. 使用CustomOpBuilder编译自定义算子

将上述C++代码保存成文件`asdsip_fftc2c.cpp`，然后使用Python接口`CustomOpBuilder`编译。

```python
input_np = np.random.rand(2, 16)
real_np = input_np.astype(np.float32)
imag_np = input_np.astype(np.float32)
complex_np = real_np + 1j * imag_np
my_ops = CustomOpBuilder("asdsip_fftc2c", "jit_test_files/asdsip_fftc2c.cpp", enable_asdsip=True).load()
output_tensor = my_ops.fft(input_tensor, 16, 2)
print(output_tensor)
```

这里向`CustomOpBuilder`传入了`enable_asdsip=True`的参数，MindSpore会自动添加与ASDSIP加速库有关的编译和链接选项。用户需保证正确执行了ASDSIP库的`set_env.sh`脚本，环境中已配置`ASDSIP_HOME_PATH`环境变量。
