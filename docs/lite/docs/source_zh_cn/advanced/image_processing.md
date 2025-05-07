# 数据预处理

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/advanced/image_processing.md)

## 概述

图像预处理的主要目的是消除图像中无关的信息，恢复有用的真实信息，增强有关信息的可检测性和最大限度地简化数据，从而改进特征抽取、图像分割、匹配和识别的可靠性。此处是通过创建LiteMat对象，在推理前对图像数据进行处理，达到模型推理所需要的数据格式要求。

## 导入图像预处理函数的库

```cpp
#include "lite_cv/lite_mat.h"
#include "lite_cv/image_process.h"
```

## 对图像进行初始化

这边使用的是`image_process.h`文件中的[InitFromPixel](https://www.mindspore.cn/lite/api/zh-CN/master/generate/function_mindspore_dataset_InitFromPixel-1.html)函数对图像进行初始化操作。

```cpp
bool InitFromPixel(const unsigned char *data, LPixelType pixel_type, LDataType data_type, int w, int h, LiteMat &m)
```

### 使用示例

```cpp
// Create the data object of the LiteMat object.
LiteMat lite_mat_bgr;

// Initialize the lite_mat_bgr object.
// The image data pointer passed in by the user (The data in the Bitmap corresponding to the Android platform).
InitFromPixel(pixel_ptr, LPixelType::RGBA2GRAY, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);
```

## 可选的图像预处理操作

此处的图像处理操作，用户可以根据实际情况任意搭配使用。

### 对图像进行缩放操作

这边利用的是`image_process.h`中的[ResizeBilinear](https://www.mindspore.cn/lite/api/zh-CN/master/generate/function_mindspore_dataset_ResizeBilinear-1.html)函数通过双线性算法调整图像大小，当前仅支持的数据类型为uint8，当前支持的通道为3和1。

```cpp
bool ResizeBilinear(const LiteMat &src, LiteMat &dst, int dst_w, int dst_h)
```

#### 使用示例

```cpp
// Initialize the image data.
LiteMat lite_mat_bgr;
InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);

// Create a resize image data object.
LiteMat lite_mat_resize;

// Resize the image.
ResizeBilinear(lite_mat_bgr, lite_mat_resize, 256, 256);
```

### 对图像数据类型进行转换

这边利用的是`image_process.h`中的[ConvertTo](https://www.mindspore.cn/lite/api/zh-CN/master/generate/function_mindspore_dataset_ConvertTo-1.html)函数对图像数据类型进行转换，目前支持的转换是将uint8转换为float。

```cpp
bool ConvertTo(const LiteMat &src, LiteMat &dst, double scale = 1.0)
```

#### 使用示例

```cpp
// Initialize the image data.
LiteMat lite_mat_bgr;
InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);

// Create the converted data type object.
LiteMat lite_mat_convert_float;

// Perform conversion type operations on the object. The currently supported conversion is to convert uint8 to float.
ConvertTo(lite_mat_bgr, lite_mat_convert_float);
```

### 对图像数据进行裁剪

这边利用的是`image_process.h`中的[Crop](https://www.mindspore.cn/lite/api/zh-CN/master/generate/function_mindspore_dataset_Crop-1.html)函数对图像进行裁剪，目前支持通道3和1。

```cpp
bool Crop(const LiteMat &src, LiteMat &dst, int x, int y, int w, int h)
```

#### 使用示例

```cpp
// Initialize the image data.
LiteMat lite_mat_bgr;
InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);

// Create the cropped object.
LiteMat lite_mat_cut;

// The image is cropped by the values of x, y, w, h.
Crop(lite_mat_bgr, lite_mat_cut, 16, 16, 224, 224);
```

### 对图像数据进行归一化处理

为了消除数据指标之间的量纲影响，通过标准化处理来解决数据指标之间的可比性问题，这边利用的是`image_process.h`中的[SubStractMeanNormalize](https://www.mindspore.cn/lite/api/zh-CN/master/generate/function_mindspore_dataset_SubStractMeanNormalize-1.html)函数对图像数据进行归一化处理。

```cpp
bool SubStractMeanNormalize(const LiteMat &src, LiteMat &dst, const std::vector<float> &mean, const std::vector<float> &std)
```

#### 使用示例

```cpp
// Initialize the image data.
LiteMat lite_mat_bgr;
InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);

// The mean value of the image data.
// The variance of the image data.
std::vector<float> means = {0.485, 0.456, 0.406};
std::vector<float> stds = {0.229, 0.224, 0.225};

// Create a normalized image object.
LiteMat lite_mat_bgr_norm;

// The image data is normalized by the mean value and variance of the image data.
SubStractMeanNormalize(lite_mat_bgr, lite_mat_bgr_norm, means, stds);
```
