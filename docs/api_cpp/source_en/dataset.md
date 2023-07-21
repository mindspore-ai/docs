# mindspore::dataset

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/api_cpp/source_en/dataset.md)

## Execute

\#include &lt;[execute.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/include/execute.h)&gt;

```cpp
Execute(std::shared_ptr<TensorOperation> op);

Execute(std::vector<std::shared_ptr<TensorOperation>> ops);
```

Class to run Tensor operations(cv, text) in the eager mode.

- Parameters

    - `op`: Single transform operation to be used.
    - `ops`: A list of transform operations to be used.

```cpp
Status operator()(const mindspore::MSTensor &input, mindspore::MSTensor *output);
```

Callable function to execute the TensorOperation in the eager mode.

- Parameters

    - `input`: Tensor to be transformed.
    - `output`: Transformed tensor.

- Returns

    Return Status code to indicate transform successful or not.

## ResizeBilinear

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

```cpp
bool ResizeBilinear(LiteMat &src, LiteMat &dst, int dst_w, int dst_h)
```

Resize image by bilinear algorithm, currently the data type only supports uint8, the channel only supports 3 and 1.

- Parameters

    - `src`: Input image data.
    - `dst`: Output image data.
    - `dst_w`: The width of the output image data.
    - `dst_h`: The height of the output image data.

- Returns

    Return true if the execution is successful, otherwise return false if the condition is not met.

## InitFromPixel

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

```cpp
bool InitFromPixel(const unsigned char *data, LPixelType pixel_type, LDataType data_type, int w, int h, LiteMat &m)
```

Initialize LiteMat from pixel, providing data in RGB or BGR format does not need to be converted. Currently the conversion supports RGB_TO_BGR, RGBA_To_RGB, RGBA_To_BGR, NV21_To_BGR and NV12_To_BGR.

- Parameters

    - `data`: Input data.
    - `pixel_type`: The type of pixel.
    - `data_type`: The type of data.
    - `w`: The width of the output data.
    - `h`: The height of the output data.
    - `mat`: Used to store image data.

- Returns

    Return true if the initialization is successful, otherwise return false.

## ConvertTo

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

```cpp
bool ConvertTo(LiteMat &src, LiteMat &dst, double scale = 1.0)
```

Convert the data type, currently it supports converting the data type from uint8 to float.

- Parameters

    - `src`: Input image data.
    - `dst`: Output image data.
    - `scale`: Scale pixel values (default: 1.0).

- Returns

    Return true if the data type is converted successfully, otherwise return false.

## Crop

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

```cpp
bool Crop(LiteMat &src, LiteMat &dst, int x, int y, int w, int h)
```

Crop image, the channel supports 3 and 1.

- Parameters

    - `src`: Input image data.
    - `dst`: Output image data.
    - `x`: The x coordinate value of the starting point of the screenshot.
    - `y`: The y coordinate value of the starting point of the screenshot.
    - `w`: The width of the screenshot.
    - `h`: The height of the screenshot.

- Returns

    Return true if the image is cropped successfully, otherwise return false.

## SubStractMeanNormalize

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

```cpp
bool SubStractMeanNormalize(const LiteMat &src, LiteMat &dst, const std::vector<float> &mean, const std::vector<float> &std)
```

Normalize image, currently the supports data type is float.

- Parameters

    - `src`: Input image data.
    - `dst`: Output image data.
    - `mean`: Mean of the data set.
    - `std`: Norm of the data set.

- Returns

    Return true if the normalization is successful, otherwise return false.

## Pad

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

```cpp
bool Pad(const LiteMat &src, LiteMat &dst, int top, int bottom, int left, int right, PaddBorderType pad_type, uint8_t fill_b_or_gray, uint8_t fill_g, uint8_t fill_r)
```

Pad image, the channel supports 3 and 1.

- Parameters

    - `src`: Input image data.
    - `dst`: Output image data.
    - `top`: The length of top.
    - `bottom`: The length of bottom.
    - `left`: The length of left.
    - `right`: The length of right.
    - `pad_type`: The type of pad.
    - `fill_b_or_gray`: B or GRAY.
    - `fill_g`: G.
    - `fill_r`: R.

- Returns

    Return true if the image is filled successfully, otherwise return false.

## ExtractChannel

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

```cpp
bool ExtractChannel(const LiteMat &src, LiteMat &dst, int col)
```

Extract image channel by index.

- Parameters

    - `src`: Input image data.
    - `col`: The serial number of the channel.

- Returns

    Return true if the image channel is extracted successfully, otherwise return false.

## Split

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

```cpp
bool Split(const LiteMat &src, std::vector<LiteMat> &mv)
```

Split image channels to single channel.

- Parameters

    - `src`: Input image data.
    - `mv`: Single channel data.

- Returns

    Return true if the image channel is split successfully, otherwise return false.

## Merge

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

```cpp
bool Merge(const std::vector<LiteMat> &mv, LiteMat &dst)
```

Create a multi-channel image out of several single-channel arrays.

- Parameters

    - `mv`: Single channel data.
    - `dst`: Output image data.

- Returns

    Return true if the multi-channel image is created successfully, otherwise returns false.

## Affine

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

```cpp
void Affine(LiteMat &src, LiteMat &out_img, double M[6], std::vector<size_t> dsize, UINT8_C1 borderValue)
```

Apply affine transformation to the 1-channel image.

- Parameters

    - `src`: Input image data.
    - `out_img`: Output image data.
    - `M[6]`: Affine transformation matrix.
    - `dsize`: The size of the output image.
    - `borderValue`: The pixel value is used for filing after the image is captured.

```cpp
void Affine(LiteMat &src, LiteMat &out_img, double M[6], std::vector<size_t> dsize, UINT8_C3 borderValue)
```

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

Apply affine transformation to the 3-channel image.

- Parameters

    - `src`: Input image data.
    - `out_img`: Output image data.
    - `M[6]`: Affine transformation matrix.
    - `dsize`: The size of the output image.
    - `borderValue`: The pixel value is used for filing after the image is captured.

## GetDefaultBoxes

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

```cpp
std::vector<std::vector<float>> GetDefaultBoxes(BoxesConfig config)
```

Get default anchor boxes for Faster R-CNN, SSD, YOLO, etc.

- Parameters

    - `config`: Objects of BoxesConfig structure.

- Returns

    Return the default boxes.

## ConvertBoxes

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

```cpp
void ConvertBoxes(std::vector<std::vector<float>> &boxes, std::vector<std::vector<float>> &default_boxes, BoxesConfig config)
```

Convert the prediction boxes to the actual boxes with (y, x, h, w).

- Parameters

    - `boxes`: Actual size box.
    - `default_boxes`: Default box.
    - `config`: Objects of BoxesConfig structure.

## ApplyNms

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

```cpp
std::vector<int> ApplyNms(std::vector<std::vector<float>> &all_boxes, std::vector<float> &all_scores, float thres, int max_boxes)
```

Real-size box non-maximum suppression.

- Parameters

    - `all_boxes`: All input boxes.
    - `all_scores`: Score after all boxes are executed through the network.
    - `thres`: Pre-value of IOU.
    - `max_boxes`: Maximum value of output box.

- Returns

    Return the id of the boxes.

## LiteMat

\#include &lt;[lite_mat.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/lite_mat.h)&gt;

LiteMat is a class that processes images.

### Constructors & Destructors

#### LiteMat

```cpp
LiteMat()

LiteMat(int width, LDataType data_type = LDataType::UINT8)

LiteMat(int width, int height, LDataType data_type = LDataType::UINT8)

LiteMat(int width, int height, int channel, LDataType data_type = LDataType::UINT8)
```

Constructor of MindSpore dataset LiteMat using default value of parameters.

#### ~LiteMat

```cpp
~LiteMat()
```

Destructor of MindSpore dataset LiteMat.

### Public Member Functions

#### Init

```cpp
void Init(int width, LDataType data_type = LDataType::UINT8)

void Init(int width, int height, LDataType data_type = LDataType::UINT8)

void Init(int width, int height, int channel, LDataType data_type = LDataType::UINT8)
```

The function to initialize the channel, width and height of the image, but the parameters are different.

#### IsEmpty

```cpp
bool IsEmpty() const
```

A function to determine whether the object is empty.

- Returns

    Return true or false.

#### Release

```cpp
void Release()
```

A function to release memory.

### Public Attributes

#### data_ptr_

```cpp
data_ptr_
```

A **pointer** to the address of the image.

#### elem_size_

```cpp
elem_size_
```

An **int** value. Bytes of the element.

#### width_

```cpp
width_
```

An **int** value. The width of the image.

#### height_

```cpp
height_
```

An **int** value. The height of the image.

#### channel_

```cpp
channel_
```

An **int** value. The number of channels of the image.

#### c_step_

```cpp
c_step_
```

An **int** value. The product of width and height after alignment.

#### dims_

```cpp
dims_
```

An **int** value. The dimensions of the image.

#### size_

```cpp
size_
```

The memory size of the image.

#### data_type_

```cpp
data_type_
```

The data type of the image.

#### ref_count_

```cpp
ref_count_
```

A **pointer** to the address of the reference counter.

## Subtract

\#include &lt;[lite_mat.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/lite_mat.h)&gt;

```cpp
bool Subtract(const LiteMat &src_a, const LiteMat &src_b, LiteMat *dst)
```

Calculates the difference between the two images for each element.

- Parameters

    - `src_a`: Input image_a data.
    - `src_b`: Input image_b data.
    - `dst`: Output image data.

- Returns

    Return true if the calculation satisfies the conditions, otherwise return false.

## Divide

\#include &lt;[lite_mat.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/lite_mat.h)&gt;

```cpp
bool Divide(const LiteMat &src_a, const LiteMat &src_b, LiteMat *dst)
```

Calculates the division between the two images for each element.

- Parameters

    - `src_a`: Input image_a data.
    - `src_b`: Input image_b data.
    - `dst`: Output image data.

- Returns

    Return true if the calculation satisfies the conditions, otherwise return false.

## Multiply

\#include &lt;[lite_mat.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/lite_mat.h)&gt;

```cpp
bool Multiply(const LiteMat &src_a, const LiteMat &src_b, LiteMat *dst)
```

Calculates the multiply between the two images for each element.

- Parameters

    - `src_a`: Input image_a data.
    - `src_b`: Input image_b data.
    - `dst`: Output image data.

- Returns

    Return true if the calculation satisfies the conditions, otherwise return false.
