# mindspore::dataset

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/docs/api_cpp/source_zh_cn/dataset.md)

\#include &lt;[lite_mat.h](https://gitee.com/mindspore/mindspore/blob/r1.0/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/lite_mat.h)&gt;  
\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.0/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

## image_process.h文件的函数

### ResizeBilinear

```cpp
bool ResizeBilinear(LiteMat &src, LiteMat &dst, int dst_w, int dst_h)
```

通过双线性算法调整图像大小，当前仅支持的数据类型为uint8，当前支持的通道为3和1。

- 参数

    - `src`: 输入的图片数据。
    - `dst`: 输出的图片数据。
    - `dst_w`: 输出图片数据的宽度。
    - `dst_h`: 输出图片数据的高度。
- 返回值

    返回True或者False。

### InitFromPixel

```cpp
bool InitFromPixel(const unsigned char *data, LPixelType pixel_type, LDataType data_type, int w, int h, LiteMat &m)
```

从像素初始化LiteMat，当前支持的转换是rbgaTorgb和rgbaTobgr。

- 参数

    - `data`: 输入的数据。
    - `pixel_type`: 像素点的类型。
    - `data_type`: 数据的类型。
    - `w`: 输出数据的宽度。
    - `h`: 输出数据的高度。
    - `mat`: 用于存储图像数据。
- 返回值

    返回True或者False。

### ConvertTo

```cpp
bool ConvertTo(LiteMat &src, LiteMat &dst, double scale = 1.0)
```

转换数据类型，当前支持的转换是将uint8转换为float。

- 参数

    - `src`: 输入的图片数据。
    - `dst`: 输出图像数据。
    - `scale`: 对像素做尺度(默认值为1.0)。

- 返回值

    返回True或者False。

### Crop

```cpp
bool Crop(LiteMat &src, LiteMat &dst, int x, int y, int w, int h)
```

裁剪图像，通道支持为3和1。

- 参数

    - `src`: 输入的图片数据。
    - `dst`: 输出图像数据。
    - `x`: 屏幕截图起点的x坐标值。
    - `y`: 屏幕截图起点的y坐标值。
    - `w`: 截图的宽度。
    - `h`: 截图的高度。
- 返回值

    返回True或者False。

### SubStractMeanNormalize

```cpp
bool SubStractMeanNormalize(const LiteMat &src, LiteMat &dst, const std::vector<float> &mean, const std::vector<float> &std)
```

规一化图像，当前支持的数据类型为float。

- 参数

    - `src`: 输入的图片数据。
    - `dst`: 输出图像数据。
    - `mean`: 数据集的均值。
    - `std`: 数据集的方差。
- 返回值

    返回True或者False。

### Pad

```cpp
bool Pad(const LiteMat &src, LiteMat &dst, int top, int bottom, int left, int right, PaddBorderType pad_type, uint8_t fill_b_or_gray, uint8_t fill_g, uint8_t fill_r)
```

填充图像，通道支持为3和1。

- 参数

    - `src`: 输入的图片数据。
    - `dst`: 输出图像数据。
    - `top`: 图片顶部长度。
    - `bottom`: 图片底部长度。
    - `left`: 图片左边长度。
    - `right`: 图片右边长度。
    - `pad_type`: padding的类型。
    - `fill_b_or_gray`: R或者GRAY。
    - `fill_g`: G.
    - `fill_r`: R.
- 返回值

    返回True或者False。

### Affine

```cpp
void Affine(LiteMat &src, LiteMat &out_img, double M[6], std::vector<size_t> dsize, UINT8_C1 borderValue)
```

对1通道图像应用仿射变换。

- 参数

    - `src`: 输入图片数据。
    - `out_img`: 输出图片数据。
    - `M[6]`: 仿射变换矩阵。
    - `dsize`: 输出图像的大小。
    - `borderValue`: 采图之后用于填充的像素值。

```cpp
void Affine(LiteMat &src, LiteMat &out_img, double M[6], std::vector<size_t> dsize, UINT8_C3 borderValue)
```

对3通道图像应用仿射变换。

- 参数

    - `src`: 输入图片数据。
    - `out_img`: 输出图片数据。
    - `M[6]`: 仿射变换矩阵。
    - `dsize`: 输出图像的大小。
    - `borderValue`: 采图之后用于填充的像素值。

### GetDefaultBoxes

```cpp
std::vector<std::vector<float>> GetDefaultBoxes(BoxesConfig config)
```

获取Faster R-CNN，SSD，YOLO等的默认框。

- 参数

    - `config`: BoxesConfig结构体对象。

- 返回值

    返回默认框。

### ConvertBoxes

```cpp
void ConvertBoxes(std::vector<std::vector<float>> &boxes, std::vector<std::vector<float>> &default_boxes, BoxesConfig config)
```

将预测框转换为（y，x，h，w）的实际框。

- 参数

    - `boxes`: 实际框的大小。
    - `default_boxes`: 默认框。
    - `config`: BoxesConfig结构体对象。

### ApplyNms

```cpp
std::vector<int> ApplyNms(std::vector<std::vector<float>> &all_boxes, std::vector<float> &all_scores, float thres, int max_boxes)
```

对实际框的非极大值抑制。

- 参数

    - `all_boxes`: 所有输入的框。
    - `all_scores`: 通过网络执行后所有框的得分。
    - `thres`: IOU的预值。
    - `max_boxes`: 输出框的最大值。
- 返回值

    返回框的id。

&emsp;

## LiteMat

LiteMat是一个处理图像的类。

### 构造函数和析构函数

### LiteMat

```cpp
LiteMat()

LiteMat(int width, LDataType data_type = LDataType::UINT8)

LiteMat(int width, int height, LDataType data_type = LDataType::UINT8)

LiteMat(int width, int height, int channel, LDataType data_type = LDataType::UINT8)
```

MindSpore中dataset模块下LiteMat的构造方法，使用参数的默认值。

```cpp
~LiteMat();
```

MindSpore dataset LiteMat的析构函数。

### 公有成员函数

### Init

```cpp
void Init(int width, LDataType data_type = LDataType::UINT8)

void Init(int width, int height, LDataType data_type = LDataType::UINT8)

void Init(int width, int height, int channel, LDataType data_type = LDataType::UINT8)
```

该函数用于初始化图像的通道，宽度和高度，参数不同。

### IsEmpty

```cpp
bool IsEmpty() const
```

确定对象是否为空的函数。

- 返回值

    返回True或者False。

### Release

```cpp
void Release()
```

释放内存的函数。

### 私有成员函数

### AlignMalloc

```cpp
void *AlignMalloc(unsigned int size)
```

申请内存对齐的函数。

- 参数

    - `size`: 内存大小。

- 返回值

   返回指针的大小。

### AlignFree

```cpp
void AlignFree(void *ptr)
```

释放指针内存大小的方法。

### InitElemSize

```cpp
void InitElemSize(LDataType data_type)
```

通过data_type初始化元素字节数的值。

- 参数

    - `data_type`: 数据的类型。

```cpp
 int addRef(int *p, int value)
```

用于计算引用该函数次数的函数。

- 参数

    - `p`: 指向引用的对象。
    - `value`: 引用时所加的值。
