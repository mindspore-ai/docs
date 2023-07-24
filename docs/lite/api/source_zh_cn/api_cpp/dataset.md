# mindspore::dataset

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/lite/api/source_zh_cn/api_cpp/dataset.md)

## Execute

\#include &lt;[execute.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/minddata/dataset/include/dataset/execute.h)&gt;

```cpp
// shared_ptr
Execute::Execute(std::shared_ptr<TensorTransform> op, MapTargetDevice deviceType, uint32_t device_id);
Execute::Execute(std::vector<std::shared_ptr<TensorTransform>> ops, MapTargetDevice deviceType, uint32_t device_id);

// normal pointer
Execute::Execute(std::reference_wrapper<TensorTransform> op, MapTargetDevice deviceType, uint32_t device_id);
Execute::Execute(std::vector<std::reference_wrapper<TensorTransform>> ops, MapTargetDevice deviceType, uint32_t device_id);

// reference_wrapper
Execute::Execute(TensorTransform *op, MapTargetDevice deviceType, uint32_t device_id);
Execute::Execute(std::vector<TensorTransform *> ops, MapTargetDevice deviceType, uint32_t device_id);
```

Transform（图像、文本）变换算子Eager模式执行类。支持多种构造函数形式，包括智能指针，普通指针以及引用封装。

- 参数

    - `op`: 指定单个使用的变换算子。
    - `ops`: 指定一个列表，包含多个使用的变换算子。
    - `deviceType`:  指定运行硬件设备，选项为CPU，GPU以及Ascend 310。
    - `device_id`: 指定运行硬件设备的设备号，仅当`deviceType = MapTargetDevice::kAscend310`时生效。

```cpp
Status operator()(const mindspore::MSTensor &input, mindspore::MSTensor *output);
```

Eager模式执行接口。

- 参数

    - `input`: 待变换的Tensor张量。
    - `output`: 变换后的Tensor张量。

- 返回值

    返回一个状态码指示执行变换是否成功。

```cpp
std::string Execute::AippCfgGenerator();
```

与Dvpp相关的Aipp配置文件生成器。
该接口在`deviceType = kAscend310`时生效，依据数据预处理算子的参数自动生成Ascend 310内置Aipp模块推理配置文件。

- 参数

    无。

- 返回值

    返回一个`string`表示Aipp配置文件的系统路径。

## Dvpp模块

Dvpp模块为Ascend 310芯片内置硬件解码器，相较于CPU拥有对图形处理更强劲的性能。支持JPEG图片的解码缩放等基础操作。

- 定义`execute`对象时，若设置`deviceType = kAscend310`则会调用Dvpp模块执行数据预处理算子。
- 当前Dvpp模块支持`Decode()`， `Resize()`， `CenterCrop()`， `Normalize()`。
- 上述Dvpp算子与同功能CPU算子共用统一API，仅以`deviceType`区分。

示例代码：

```cpp
// Define dvpp transforms
std::vector<int32_t> crop_paras = {224, 224};
std::vector<int32_t> resize_paras = {256};
std::vector<float> mean = {0.485 * 255, 0.456 * 255, 0.406 * 255};
std::vector<float> std = {0.229 * 255, 0.224 * 255, 0.225 * 255};

std::shared_ptr<TensorTransform> decode(new vision::Decode());
std::shared_ptr<TensorTransform> resize(new vision::Resize(resize_paras));
std::shared_ptr<TensorTransform> centercrop(new vision::CenterCrop(crop_paras));
std::shared_ptr<TensorTransform> normalize(new vision::Normalize(mean, std));

std::vector<std::shared_ptr<TensorTransform>> trans_list = {decode, resize, centercrop, normalize};
mindspore::dataset::Execute Transform(trans_list, MapTargetDevice::kAscend310);
```

## ResizeBilinear

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

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

    执行成功返回true，执行失败返回false。

## InitFromPixel

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

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

    初始化成功返回true，否则返回false。

## ConvertTo

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

```cpp
bool ConvertTo(LiteMat &src, LiteMat &dst, double scale = 1.0)
```

转换数据类型，当前支持的转换是将uint8转换为float。

- 参数

    - `src`: 输入的图片数据。
    - `dst`: 输出图像数据。
    - `scale`: 对像素进行缩放(默认值为1.0)。

- 返回值

    转换数据类型成功返回true，否则返回false。

## Crop

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

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

    裁剪图像成功返回true，否则返回false。

## SubStractMeanNormalize

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

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

    归一化成功返回true，否则返回false。

## Pad

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

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
    - `fill_g`: G。
    - `fill_r`: R。

- 返回值

    填充图像成功返回true，否则返回false。

## ExtractChannel

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

```cpp
bool ExtractChannel(const LiteMat &src, LiteMat &dst, int col)
```

按索引提取图像通道。

- 参数

    - `src`: 输入的图片数据。
    - `col`: 通道的序号。

- 返回值

    提取图像通道成功返回true，否则返回false。

## Split

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

```cpp
bool Split(const LiteMat &src, std::vector<LiteMat> &mv)
```

将图像通道拆分为单通道。

- 参数

    - `src`: 输入的图片数据。
    - `mv`: 单个通道数据。

- 返回值

    图像通道拆分成功返回true，否则返回false。

## Merge

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

```cpp
bool Merge(const std::vector<LiteMat> &mv, LiteMat &dst)
```

用几个单通道阵列创建一个多通道图像。

- 参数

    - `mv`: 单个通道数据。
    - `dst`: 输出图像数据。

- 返回值

    创建多通道图像成功返回true，否则返回false。

## Affine

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

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

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

对3通道图像应用仿射变换。

- 参数

    - `src`: 输入图片数据。
    - `out_img`: 输出图片数据。
    - `M[6]`: 仿射变换矩阵。
    - `dsize`: 输出图像的大小。
    - `borderValue`: 采图之后用于填充的像素值。

## GetDefaultBoxes

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

```cpp
std::vector<std::vector<float>> GetDefaultBoxes(BoxesConfig config)
```

获取Faster R-CNN，SSD，YOLO等的默认框。

- 参数

    - `config`: BoxesConfig结构体对象。

- 返回值

    返回默认框。

## ConvertBoxes

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

```cpp
void ConvertBoxes(std::vector<std::vector<float>> &boxes, std::vector<std::vector<float>> &default_boxes, BoxesConfig config)
```

将预测框转换为（y，x，h，w）的实际框。

- 参数

    - `boxes`: 实际框的大小。
    - `default_boxes`: 默认框。
    - `config`: BoxesConfig结构体对象。

## ApplyNms

\#include &lt;[image_process.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/image_process.h)&gt;

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

## LiteMat

\#include &lt;[lite_mat.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/lite_mat.h)&gt;

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

#### ~LiteMat

```cpp
~LiteMat()
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

    返回true或者false。

#### Release

```cpp
void Release()
```

释放内存的函数。

### 公有属性

#### data_ptr_

```cpp
data_ptr_
```

**pointer**类型，表示存放图像数据的地址。

#### elem_size_

```cpp
elem_size_
```

**int**类型，表示元素的字节数。

#### width_

```cpp
width_
```

**int**类型，表示图像的宽度。

#### height_

```cpp
height_
```

**int**类型，表示图像的高度。

#### channel_

```cpp
channel_
```

**int**类型，表示图像的通道数。

#### c_step_

```cpp
c_step_
```

**int**类型，表示经过对齐后的图像宽高之积。

#### dims_

```cpp
dims_
```

**int**类型，表示图像的维数。

#### size_

```cpp
size_
```

**size_t**类型，表示图像占用内存的大小。

#### data_type_

```cpp
data_type_
```

**LDataType**类型，表示图像的数据类型。

#### ref_count_

```cpp
ref_count_
```

**pointer**类型，表示引用计数器的地址。

## Subtract

\#include &lt;[lite_mat.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/lite_mat.h)&gt;

```cpp
bool Subtract(const LiteMat &src_a, const LiteMat &src_b, LiteMat *dst)
```

对两个图像间的元素进行减法运算。

- 参数

    - `src_a`: 输入的图像a的数据。
    - `src_b`: 输入的图像b的数据。
    - `dst`: 输出图像的数据。

- 返回值

    满足条件的计算返回true，否则返回false。

## Divide

\#include &lt;[lite_mat.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/lite_mat.h)&gt;

```cpp
bool Divide(const LiteMat &src_a, const LiteMat &src_b, LiteMat *dst)
```

对两个图像间的元素进行除法运算。

- 参数

    - `src_a`: 输入的图像a的数据。
    - `src_b`: 输入的图像b的数据。
    - `dst`: 输出图像的数据。

- 返回值

    满足条件的计算返回true，否则返回false。

## Multiply

\#include &lt;[lite_mat.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/lite_mat.h)&gt;

```cpp
bool Multiply(const LiteMat &src_a, const LiteMat &src_b, LiteMat *dst)
```

对两个图像间的元素进行乘法运算。

- 参数

    - `src_a`: 输入的图像a的数据。
    - `src_b`: 输入的图像b的数据。
    - `dst`: 输出图像的数据。

- 返回值

    满足条件的计算返回true，否则返回false。
