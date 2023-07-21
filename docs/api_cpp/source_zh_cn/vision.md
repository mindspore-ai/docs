# mindspore::dataset::vision

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/api_cpp/source_zh_cn/vision.md)

## CenterCrop

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

```cpp
std::shared_ptr<CenterCropOperation> CenterCrop(std::vector<int32_t> size)
```

从输入的图像的中心区域裁剪出给定尺寸的区域。

- 参数

    - `size`: 输出裁剪区域的尺寸。如果size为单个值，则会生成一个正方形的裁剪区域，如果size具有2个值，则分别对应裁剪区域的高度、宽度。

- 返回值

    返回一个CenterCrop的算子。

## Crop

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

```cpp
std::shared_ptr<CropOperation> Crop(std::vector<int32_t> coordinates, std::vector<int32_t> size)
```

根据起始位置和裁剪尺寸，从输入图像中裁切出指定区域。

- 参数

    - `coordinates`: 裁剪区域在图像中的起始位置。
    - `size`: 输出裁剪区域的尺寸。如果size为单个值，则会生成一个正方形的裁剪区域；如果size具有2个值，则分别对应裁剪区域的高度、宽度。

- 返回值

    返回一个Crop的算子。

## Decode

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

```cpp
std::shared_ptr<DecodeOperation> Decode(bool rgb = true)
```

对输入的图像进行解码。

- 参数

    - `rgb`: 表示是否执行RGB模式解码。

- 返回值

    返回一个Decode的算子。

## Normalize

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

```cpp
std::shared_ptr<NormalizeOperation> Normalize(std::vector<float> mean, std::vector<float> std)
```

通过给定的均值和标准差对输入的图像进行标准化。

- 参数

    - `mean`: 表示进行标准化的均值。
    - `std`: 表示进行标准化的标准差。

- 返回值

    返回一个Normalize的算子。

## Resize

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

```cpp
std::shared_ptr<ResizeOperation> Resize(std::vector<int32_t> size, InterpolationMode interpolation = InterpolationMode::kLinear)
```

对输入的图像的长、宽尺寸进行调整。

- 参数

    - `size`: 表示调整后的图像的输出尺寸大小。如果size为单个值，图像的短边会调整到此值，另一边则将以相同的纵横比进行调整；如果size具有2个值，则对应输出图像的高度、宽度。
    - `interpolation`: 插值模式的枚举。

- 返回值

    返回一个Resize的算子。

## HWC2CHW

\#include &lt;[vision.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/include/vision.h)&gt;

```cpp
std::shared_ptr<HwcToChwOperation> HWC2CHW()
```

将输入图像的通道顺序从（H，W，C）转换成（C，H，W）。

- 返回值

    返回一个HwcToChw的算子。

## Pad

\#include &lt;[vision.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/include/vision.h)&gt;

```cpp
std::shared_ptr<PadOperation> Pad(std::vector<int32_t> padding, std::vector<uint8_t> fill_value = {0}, BorderType padding_mode = BorderType::kConstant)
```

根据给定的填充参数对图像进行填充。

- 参数

    - `padding`: 图像的上、下、左、右边需要填充的像素个数。如果padding为单个值P，则四个边都填充P个像素；如果padding为两个值(P, Q)，则左、右边填充P个像素，上、下边填充Q个像素；如果padding为四个值，则分别对应左、上、右、下四个边的填充像素个数。
    - `fill_value`: 需要填充的像素值。
    - `padding_mode`: 填充的模式，可以为常量模式、边界模式、反射模式、对称模式。

- 返回值

    返回一个Pad的算子。

## Rescale

\#include &lt;[vision.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/include/vision.h)&gt;

```cpp
std::shared_ptr<RescaleOperation> Rescale(float rescale, float shift)
```

对输入图像的像素进行`y = αx + β`变换。

- 参数

    - `rescale`: 变换的α参数。
    - `shift`: 变换的β参数。

- 返回值

    返回一个Rescale的算子。