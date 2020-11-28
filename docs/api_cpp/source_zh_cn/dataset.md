# mindspore::dataset

<a href="https://gitee.com/mindspore/docs/blob/master/docs/api_cpp/source_zh_cn/dataset.md" target="_blank"><img src="./_static/logo_source.png"></a>

\#include &lt;[lite_mat.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/lite_mat.h)&gt;

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

\#include &lt;[vision.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ccsrc/minddata/dataset/include/vision.h)&gt;

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

    执行成功返回True，否则不满足条件返回False。

### InitFromPixel

```cpp
bool InitFromPixel(const unsigned char *data, LPixelType pixel_type, LDataType data_type, int w, int h, LiteMat &m)
```

从像素初始化LiteMat，提供数据为RGB或者BGR格式，不用进行格式转换，当前支持的转换是RGB_TO_BGR、RGBA_To_RGB、RGBA_To_BGR、NV21_To_BGR和NV12_To_BGR。

- 参数

    - `data`: 输入的数据。
    - `pixel_type`: 像素点的类型。
    - `data_type`: 数据的类型。
    - `w`: 输出数据的宽度。
    - `h`: 输出数据的高度。
    - `mat`: 用于存储图像数据。

- 返回值

    初始化成功返回True，否则返回False。

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

    转换数据类型成功返回True，否则返回False。

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

    裁剪图像成功返回True，否则返回False。

### SubStractMeanNormalize

```cpp
bool SubStractMeanNormalize(const LiteMat &src, LiteMat &dst, const std::vector<float> &mean, const std::vector<float> &std)
```

归一化图像，当前支持的数据类型为float。

- 参数

    - `src`: 输入的图片数据。
    - `dst`: 输出图像数据。
    - `mean`: 数据集的均值。
    - `std`: 数据集的方差。

- 返回值

    归一化成功返回True，否则返回False。

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

    填充图像成功返回True，否则返回False。

### ExtractChannel

```cpp
bool ExtractChannel(const LiteMat &src, LiteMat &dst, int col)
```

按索引提取图像通道。

- 参数

    - `src`: 输入的图片数据。
    - `col`: 通道的序号。

- 返回值

    提取图像通道成功返回True，否则返回False。

### Split

```cpp
bool Split(const LiteMat &src, std::vector<LiteMat> &mv)
```

将图像通道拆分为单通道。

- 参数

    - `src`: 输入的图片数据。
    - `mv`: 单个通道数据。

- 返回值

    图像通道拆分成功返回True，否则返回False。

### Merge

```cpp
bool Merge(const std::vector<LiteMat> &mv, LiteMat &dst)
```

用几个单通道阵列创建一个多通道图像。

- 参数

    - `mv`: 单个通道数据。
    - `dst`: 输出图像数据。

- 返回值

    创建多通道图像成功返回True，否则返回False。

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

### Subtract

```cpp
bool Subtract(const LiteMat &src1, const LiteMat &src2, LiteMat &dst)
```

计算每个元素的两个图像之间的差异。

- 参数

    - `src1`: 输入的图像1的数据。
    - `src2`: 输入的图像2的数据。
    - `dst`: 输出图像的数据。

- 返回值

    满足条件的计算返回True，否则返回False。

### Divide

```cpp
bool Divide(const LiteMat &src1, const LiteMat &src2, LiteMat &dst);
```

计算每个元素在两个图像之间的划分。

- 参数

    - `src1`: 输入的图像1的数据。
    - `src2`: 输入的图像2的数据。
    - `dst`: 输出图像的数据。

- 返回值

    满足条件的计算返回True，否则返回False。

&emsp;

## LiteMat

LiteMat是一个处理图像的类。

### 构造函数和析构函数

#### LiteMat

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

#### Init

```cpp
void Init(int width, LDataType data_type = LDataType::UINT8)

void Init(int width, int height, LDataType data_type = LDataType::UINT8)

void Init(int width, int height, int channel, LDataType data_type = LDataType::UINT8)
```

该函数用于初始化图像的通道，宽度和高度，参数不同。

#### IsEmpty

```cpp
bool IsEmpty() const
```

确定对象是否为空的函数。

- 返回值

    返回True或者False。

#### Release

```cpp
void Release()
```

释放内存的函数。

### 私有成员函数

#### AlignMalloc

```cpp
void *AlignMalloc(unsigned int size)
```

申请内存对齐的函数。

- 参数

    - `size`: 内存大小。

- 返回值

   返回指针的大小。

#### AlignFree

```cpp
void AlignFree(void *ptr)
```

释放指针内存大小的方法。

#### InitElemSize

```cpp
void InitElemSize(LDataType data_type)
```

通过data_type初始化元素字节数的值。

- 参数

    - `data_type`: 数据的类型。

#### addRef

```cpp
 int addRef(int *p, int value)
```

用于计算引用该函数次数的函数。

- 参数

    - `p`: 指向引用的对象。
    - `value`: 引用时所加的值。

## 端侧训练相关算子

### Resize

```cpp
std::shared_ptr<ResizeOperation> Resize(std::vector<int32_t> size, InterpolationMode interpolation = InterpolationMode::kLinear);
```

通过给定的大小对输入的PIL图像进行调整。

- 参数

    - `size`: 表示调整大小后的图像的输出大小。如果size为单个值，则将以相同的图像纵横比将图像调整为该值，如果size具有2个值，则应为（高度，宽度）。
    - `interpolation`: 插值模式的枚举。

- 返回值

    返回一个Resize的算子。

### CenterCrop

```cpp
std::shared_ptr<CenterCropOperation> CenterCrop(std::vector<int32_t> size);
```

将输入的PIL图像的中心区域裁剪到给定的大小。

- 参数

    - `size`: 表示调整大小后的图像的输出大小。如果size为单个值，则将以相同的图像纵横比将图像调整为该值， 如果size具有2个值，则应为（高度，宽度）。

- 返回值

    返回一个CenterCrop的算子。