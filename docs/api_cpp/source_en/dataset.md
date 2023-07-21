# mindspore::dataset

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/docs/api_cpp/source_en/dataset.md)

\#include &lt;[lite_mat.h](https://gitee.com/mindspore/mindspore/blob/r1.0/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/lite_mat.h)&gt;  
\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.0/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

## Functions of image_process.h

### ResizeBilinear

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

    Return True or False.

### InitFromPixel

```cpp
bool InitFromPixel(const unsigned char *data, LPixelType pixel_type, LDataType data_type, int w, int h, LiteMat &m)
```

Initialize LiteMat from pixel, currently the conversion supports rbgaTorgb and rgbaTobgr.

- Parameters

    - `data`: Input data.
    - `pixel_type`: The type of pixel.
    - `data_type`: The type of data.
    - `w`: The width of the output data.
    - `h`: The height of the output data.
    - `mat`: Used to store image data.
- Returns

    Return True or False.

### ConvertTo

```cpp
bool ConvertTo(LiteMat &src, LiteMat &dst, double scale = 1.0)
```

Convert the data type, currently it supports converting the data type from uint8 to float.

- Parameters

    - `src`: Input image data.
    - `dst`: Output image data.
    - `scale`: Scale pixel values(default=1.0).

- Returns

    Return True or False.

### Crop

```cpp
bool Crop(LiteMat &src, LiteMat &dst, int x, int y, int w, int h)
```

Crop image, the channel supports is 3 and 1.

- Parameters

    - `src`: Input image data.
    - `dst`: Output image data.
    - `x`: The x coordinate value of the starting point of the screenshot.
    - `y`: The y coordinate value of the starting point of the screenshot.
    - `w`: The width of the screenshot.
    - `h`: The height of the screenshot.
- Returns

    Return True or False.

### SubStractMeanNormalize

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

    Return True or False.

### Pad

```cpp
bool Pad(const LiteMat &src, LiteMat &dst, int top, int bottom, int left, int right, PaddBorderType pad_type, uint8_t fill_b_or_gray, uint8_t fill_g, uint8_t fill_r)
```

Pad image, the channel supports is 3 and 1.

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

    Return True or False.

### Affine

```cpp
void Affine(LiteMat &src, LiteMat &out_img, double M[6], std::vector<size_t> dsize, UINT8_C1 borderValue)
```

Apply affine transformation for 1 channel image.

- Parameters

    - `src`: Input image data.
    - `out_img`: Output image data.
    - `M[6]`: Affine transformation matrix.
    - `dsize`: The size of the output image.
    - `borderValue`: The pixel value is used for filing after the image is captured.

```cpp
void Affine(LiteMat &src, LiteMat &out_img, double M[6], std::vector<size_t> dsize, UINT8_C3 borderValue)
```

Apply affine transformation for 3 channel image.

- Parameters

    - `src`: Input image data.
    - `out_img`: Output image data.
    - `M[6]`: Affine transformation matrix.
    - `dsize`: The size of the output image.
    - `borderValue`: The pixel value is used for filing after the image is captured.

### GetDefaultBoxes

```cpp
std::vector<std::vector<float>> GetDefaultBoxes(BoxesConfig config)
```

Get default anchor boxes for Faster R-CNN, SSD, YOLO etc.

- Parameters

    - `config`: Objects of BoxesConfig structure.

- Returns

    Return the default boxes.

### ConvertBoxes

```cpp
void ConvertBoxes(std::vector<std::vector<float>> &boxes, std::vector<std::vector<float>> &default_boxes, BoxesConfig config)
```

Convert the prediction boxes to the actual boxes with (y, x, h, w).

- Parameters

    - `boxes`: Actual size box.
    - `default_boxes`: Default box.
    - `config`: Objects of BoxesConfig structure.

### ApplyNms

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

&emsp;

## LiteMat

Class that represents a lite Mat of a Image.

### Constructors & Destructors

### LiteMat

```cpp
LiteMat()

LiteMat(int width, LDataType data_type = LDataType::UINT8)

LiteMat(int width, int height, LDataType data_type = LDataType::UINT8)

LiteMat(int width, int height, int channel, LDataType data_type = LDataType::UINT8)
```

Constructor of MindSpore dataset LiteMat using default value of parameters.

```cpp
~LiteMat();
```

Destructor of MindSpore dataset LiteMat.

### Public Member Functions

### Init

```cpp
void Init(int width, LDataType data_type = LDataType::UINT8)

void Init(int width, int height, LDataType data_type = LDataType::UINT8)

void Init(int width, int height, int channel, LDataType data_type = LDataType::UINT8)
```

The function to initialize the channel, width and height of the image, but the parameters are different.

### IsEmpty

```cpp
bool IsEmpty() const
```

A function to determine whether the object is empty.

- Returns

    Return True or False.

### Release

```cpp
void Release()
```

A function to release memory.

### Private Member Functions

### AlignMalloc

```cpp
void *AlignMalloc(unsigned int size)
```

Apply for memory alignment.

- Parameters

    - `size`: Memory size.

- Returns

   Return the size of a pointer.

### AlignFree

```cpp
void AlignFree(void *ptr)
```

A function to release pointer memory.

```cpp
void InitElemSize(LDataType data_type)
```

Initialize the value of elem_size_ by data_type.

- Parameters

    - `data_type`: Type of data.

### addRef

```cpp
 int addRef(int *p, int value)
```

A function to count the number of times the function is referenced.

- Parameters

    - `p`: Point to the referenced object.
    - `value`: Value added when quoted.
