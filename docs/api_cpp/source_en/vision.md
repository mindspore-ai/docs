# mindspore::dataset::vision

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/api_cpp/source_en/vision.md)

## HWC2CHW

\#include &lt;[vision.h](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/ccsrc/minddata/dataset/include/vision.h)&gt;

```cpp
std::shared_ptr<HwcToChwOperation> HWC2CHW()
```

Convert the channel of the input image from (H, W, C) to (C, H, W).

- Returns

    Return a HwcToChw operator.

## CenterCrop

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

```cpp
std::shared_ptr<CenterCropOperation> CenterCrop(std::vector<int32_t> size)
```

Crop the center area of the input image to the given size.

- Parameters

    - `size`: The output size of the resized image. If the size is a single value, the image will be resized to this value with the same image aspect ratio. If the size has 2 values, it should be (height, width).

- Returns

    Return a CenterCrop operator.

## Crop

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

```cpp
std::shared_ptr<CropOperation> Crop(std::vector<int32_t> coordinates, std::vector<int32_t> size)
```

Crop an image based on the location and crop size.

- Parameters

    - `coordinates`: The starting location of the crop.
    - `size`: Size of the cropped area.

- Returns

    Return a Crop operator.

## Decode

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

```cpp
std::shared_ptr<DecodeOperation> Decode(bool rgb = true)
```

Decode the input image.

- Parameters

    - `rgb`: Whether to decode in RGB mode.

- Returns

    Return a Decode operator.

## Normalize

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

```cpp
std::shared_ptr<NormalizeOperation> Normalize(std::vector<float> mean, std::vector<float> std)
```

Normalize the input image with the given mean and standard deviation.

- Parameters

    - `mean`: The mean value to do normalization.
    - `std`: The standard deviation value to do normalization.

- Returns

    Return a Normalize operator.

## Resize

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

```cpp
std::shared_ptr<ResizeOperation> Resize(std::vector<int32_t> size, InterpolationMode interpolation = InterpolationMode::kLinear)
```

Resize the input image to the given size.

- Parameters

    - `size`: The output size of the resized image. If the size is a single value, the image will be resized to this value with the same image aspect ratio. If the size has 2 values, it should be (height, width).
    - `interpolation`: An enumeration for the mode of interpolation.
        - kLinear, Linear interpolation.
        - kNearestNeighbour, Nearest Interpolation.
        - kCubic, Bicubic interpolation.
        - kArea, Area interpolation.

- Returns

    Return a Resize operator.
