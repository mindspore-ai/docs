# mindspore::dataset::vision

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/api_cpp/source_zh_cn/vision.md)

## HWC2CHW

\#include &lt;[vision.h](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/ccsrc/minddata/dataset/include/vision.h)&gt;

```cpp
std::shared_ptr<HwcToChwOperation> HWC2CHW()
```

将输入图像的通道顺序从（H，W，C）转换成（C，H，W）。

- 返回值

    返回一个HwcToChw的算子。

## CenterCrop

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

```cpp
std::shared_ptr<CenterCropOperation> CenterCrop(std::vector<int32_t> size)
```

将输入的PIL图像的中心区域裁剪到给定的大小。

- 参数

    - `size`: 表示调整大小后的图像的输出大小。如果size为单个值，则将以相同的图像纵横比将图像调整为该值， 如果size具有2个值，则应为（高度，宽度）。

- 返回值

    返回一个CenterCrop的算子。

## Crop

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

```cpp
std::shared_ptr<CropOperation> Crop(std::vector<int32_t> coordinates, std::vector<int32_t> size)
```

根据位置和尺寸裁切图像。

- 参数

    - `coordinates`: 裁剪的起始位置。
    - `size`: 裁剪区域的大小。

- 返回值

    返回一个Crop的算子。

## Decode

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

```cpp
std::shared_ptr<DecodeOperation> Decode(bool rgb = true)
```

对输入的图像进行解码。

- 参数

    - `rgb`: 表示是否执行RGB模式解码。  

- 返回值

    返回一个Decode的算子。

## Normalize

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

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

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

```cpp
std::shared_ptr<ResizeOperation> Resize(std::vector<int32_t> size, InterpolationMode interpolation = InterpolationMode::kLinear)
```

通过给定的大小对输入的PIL图像进行调整。

- 参数

    - `size`: 表示调整大小后的图像的输出大小。如果size为单个值，则将以相同的图像纵横比将图像调整为该值，如果size具有2个值，则应为（高度，宽度）。
    - `interpolation`: 插值模式的枚举。
        - kLinear，线性差值。
        - kNearestNeighbour，最近邻插值。
        - kCubic，双三次插值。
        - kArea，区域插值。

- 返回值

    返回一个Resize的算子。
