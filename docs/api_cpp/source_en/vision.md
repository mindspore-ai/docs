# mindspore::dataset::vision

<a href="https://gitee.com/mindspore/docs/blob/master/docs/api_cpp/source_en/vision.md" target="_blank"><img src="./_static/logo_source.png"></a>

## HWC2CHW

```cpp
std::shared_ptr<HwcToChwOperation> HWC2CHW()
```

\#include &lt;[vision.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/minddata/dataset/include/vision.h)&gt;

Convert the channel of the input image from (H, W, C) to (C, H, W).

- Returns

    Return a HwcToChw operator.

## CenterCrop

```cpp
std::shared_ptr<CenterCropOperation> CenterCrop(std::vector<int32_t> size)
```

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

Crop the center area of the input image to the given size.

- Parameters

    - `size`: The output size of the resized image. If the size is a single value, the image will be resized to this value with the same image aspect ratio. If the size has 2 values, it should be (height, width).

- Returns

    Return a CenterCrop operator.

## Decode

```cpp
std::shared_ptr<DecodeOperation> Decode(bool rgb = true)
```

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

Decode the input image.

- Parameters

    - `rgb`: Whether to decode in RGB mode.

- Returns

    Return a Decode operator.

## Normalize

```cpp
std::shared_ptr<NormalizeOperation> Normalize(std::vector<float> mean, std::vector<float> std)
```

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

Normalize the input image with the given mean and standard deviation.

- Parameters

    - `mean`: The mean value to do normalization.
    - `std`: The standard deviation value to do normalization.

- Returns

    Return a Normalize operator.

## Resize

```cpp
std::shared_ptr<ResizeOperation> Resize(std::vector<int32_t> size, InterpolationMode interpolation = InterpolationMode::kLinear)
```

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

Resize the input image to the given size.

- Parameters

    - `size`: The output size of the resized image. If the size is a single value, the image will be resized to this value with the same image aspect ratio. If the size has 2 values, it should be (height, width).
    - `interpolation`: An enumeration for the mode of interpolation.

- Returns

    Return a Resize operator.
