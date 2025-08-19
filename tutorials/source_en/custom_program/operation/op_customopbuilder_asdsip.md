# CustomOpBuilder: Integrating ASDSIP FFT Operators Using AsdSipFFTOpRunner

[![View Source File](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/custom_program/operation/op_customopbuilder.md)

## Overview

[Ascend Sip Boost (ASDSIP) Operator Acceleration Library](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/acce/SiP/SIP_0000.html) is an operator library specifically designed for signal process, based on Huawei's Ascend AI processors.

When users need to use operators from the ASDSIP acceleration library that are not provided by MindSpore, they can quickly integrate and use them through custom operators.

In [Custom Operators Based on CustomOpBuilder](https://www.mindspore.cn/tutorials/en/master/custom_program/operation/op_customopbuilder.html), MindSpore provides the `PyboostRunner` tool to allow users to integrate custom operators in dynamic graphs. Now, for ASDSIP FFT operators, MindSpore additionally provides the `AsdSipFFTOpRunner` tool to encapsulate the ASDSIP FFT operator's workflow and the dynamic graph's multi-stage pipeline.

When integrating ASDSIP FFT operators using the [AsdSipFFTOpRunner class](https://www.mindspore.cn/tutorials/en/master/custom_program/operation/cpp_api_for_custom_ops.html#class-asdsipfftoprunner), users only need to provide a `Param` (used as the key for caching `Operation`) and call the `Init` interface for initialization (constructing `Operation`), followed by the `Run` interface to execute the ASDSIP FFT operator. Additionally, users can directly call the [RunAsdSipFFTOp](https://www.mindspore.cn/tutorials/en/master/custom_program/operation/cpp_api_for_custom_ops.html#function-runasdsipfftop) function for one-click execution (the function internally includes calls to both `Init` and `Run` interfaces).

This guide uses `FftC2C` as an example to demonstrate the ASDSIP FFT operator integration process. The complete code can be found in the [code repository](https://gitee.com/mindspore/mindspore/blob/master/tests/st/graph_kernel/custom/jit_test_files/asdsip_fftc2c.cpp).

## Installing the ASDSIP Acceleration Library

[Click here for installation tutorial](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/acce/SiP/SIP_0001.html)

After successful installation, the users need to activate the environment variable for the ASDSIP acceleration library:

```sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh &> /dev/null
```

## Integrating the FftC2C Operator

Here we use `ms::pynative::RunLaunchAsdSipFFTAtbOp` to integrate the operator and call the function interface through `ms::pynative::PyboostRunner::Call`:

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

### 1. Infer the Output Information of the Operator

```cpp
ms::Tensor InferFFT1DForward(const ms::Tensor &input) {
  ShapeVector out_tensor_shape(input.shape());
  return ms::Tensor(input.data_type(), out_tensor_shape);
}
```

For the `FftC2C` operator, the output tensor has the same data type and shape as the input tensor, an empty tensor is constructed using the `ms::Tensor` constructor.

### 2. Create and Set the Operator Attribute Structure

```cpp
ms::pynative::FFTParam param;
param.fftXSize = n;
param.fftYSize = 0;
param.fftType = ms::pynative::asdFftType::ASCEND_FFT_C2C;
param.direction = ms::pynative::asdFftDirection::ASCEND_FFT_FORWARD;
param.batchSize = batch_size;
param.dimType = ms::pynative::asdFft1dDimType::ASCEND_FFT_HORIZONTAL;
```

### 3. Execute the Operator via the RunAtbOp Interface

```cpp
ms::pynative::PyboostRunner::Call<1>(npu_fft, input, n, batch_size);
```

This is a template interface, equivalent to:

```cpp
auto runner = std::make_shared<AsdSipFFTOpRunner>("FFTExecC2C");
runner->Init(fft_param);
runner->Run({input}, {output});
```

By passing in the operator name, attributes, input tensor list, and output tensor list, the corresponding ASDSIP operator can be invoked. This interface supports multi-stage pipeline execution in dynamic graphs.

### 4. Bind the C++ Function to a Python Function via pybind11

```cpp
auto pyboost_npu_fft(const ms::Tensor &input, int64_t n, int64_t batch_size) {
  return ms::pynative::PyboostRunner::Call<1>(npu_fft, input, n, batch_size);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("fft", &pyboost_npu_fft, "FFT C2C", pybind11::arg("input"), pybind11::arg("n"),
        pybind11::arg("batch_size"));
}
```

### 5. Compile the Custom Operator Using CustomOpBuilder

Save the above C++ code as a file named `asdsip_fftc2c.cpp`, and then compile it using the Python interface `CustomOpBuilder`.

```python
input_np = np.random.rand(2, 16)
real_np = input_np.astype(np.float32)
imag_np = input_np.astype(np.float32)
complex_np = real_np + 1j * imag_np
my_ops = CustomOpBuilder("asdsip_fftc2c", "jit_test_files/asdsip_fftc2c.cpp", enable_asdsip=True).load()
output_tensor = my_ops.fft(input_tensor, 16, 2)
print(output_tensor)
```

Here, the parameter `enable_asdsip=True` is passed into `CustomOpBuilder`, and MindSpore will automatically add compilation and linking options related to the ASDSIP acceleration library. Users only need to ensure that the `set_env.sh` script for the ASDSIP library has been correctly executed, and the environment contains the `ASDSIP_HOME_PATH` variable.