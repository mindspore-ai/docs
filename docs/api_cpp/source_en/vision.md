# mindspore::dataset::vision

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/api_cpp/source_en/vision.md)

## CenterCrop

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

```cpp
std::shared_ptr<CenterCropOperation> CenterCrop(std::vector<int32_t> size)
```

Crop the center area of the input image to the given size.

- Parameters

    - `size`: The output size of the cropped image. If the size is a single value, a square crop of size (size, size) is returned. If the size has 2 values, it should be (height, width).

- Returns

    Return a CenterCrop operator.

## Crop

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

```cpp
std::shared_ptr<CropOperation> Crop(std::vector<int32_t> coordinates, std::vector<int32_t> size)
```

Crop an image based on the location and crop size.

- Parameters

    - `coordinates`: Starting location of crop.
    - `size`: The output size of the cropped image. If the size is a single value, a square crop of size (size, size) is returned. If the size has 2 values, it should be (height, width).

- Returns

    Return a Crop operator.

## Decode

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

```cpp
std::shared_ptr<DecodeOperation> Decode(bool rgb = true)
```

Decode the input image.

- Parameters

    - `rgb`: Whether to decode in RGB mode.

- Returns

    Return a Decode operator.

## Normalize

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

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

\#include &lt;[vision_lite.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/include/vision_lite.h)&gt;

```cpp
std::shared_ptr<ResizeOperation> Resize(std::vector<int32_t> size, InterpolationMode interpolation = InterpolationMode::kLinear)
```

Resize the input image to the given size.

- Parameters

    - `size`: The output size of the resized image. If the size is a single value, the image will be resized to this value with the same image aspect ratio. If the size has 2 values, it should be (height, width).
    - `interpolation`: An enumeration for the mode of interpolation.

- Returns

    Return a Resize operator.

## HWC2CHW

\#include &lt;[vision.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/include/vision.h)&gt;

```cpp
std::shared_ptr<HwcToChwOperation> HWC2CHW()
```

Convert the channel of the input image from (H, W, C) to (C, H, W).

- Returns

    Return a HwcToChw operator.

## Pad

\#include &lt;[vision.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/include/vision.h)&gt;

```cpp
std::shared_ptr<PadOperation> Pad(std::vector<int32_t> padding, std::vector<uint8_t> fill_value = {0}, BorderType padding_mode = BorderType::kConstant)
```

Pad the image according to padding parameters.

- Parameters

    - `padding`: A vector representing the number of pixels to pad the image. If the vector has a single value, it pads all sides of the image with that value. If the vector has two values, it pads left and right with the first value, and pads top and bottom with the second value. If the vector has four values, it pads left, top, right, and bottom with those values respectively.
    - `fill_value`: A vector representing the pixel intensity of the borders if the padding_mode is BorderType.kConstant. If 3 values are provided, it is used to fill R, G, B channels respectively.
    - `padding_mode`: padding_mode The method of padding. Can be any of BorderType.kConstant, BorderType.kEdge, BorderType.kReflect, BorderType.kSymmetric.
        - BorderType.kConstant, means it fills the border with constant values.
        - BorderType.kEdge, means it pads with the last value on the edge.
        - BorderType.kReflect, means it reflects the values on the edge omitting the last value of edge.
        - BorderType.kSymmetric, means it reflects the values on the edge repeating the last value of edge.

- Returns

    Return a Pad operator.

## Rescale

\#include &lt;[vision.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/include/vision.h)&gt;

```cpp
std::shared_ptr<RescaleOperation> Rescale(float rescale, float shift)
```

Apply `y = αx + β` transform on pixels of input image.

- Parameters

    - `rescale`: paramter α.
    - `shift`: paramter β.

- Returns

    Return a Rescale operator.
