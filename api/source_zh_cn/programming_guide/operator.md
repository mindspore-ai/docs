# 算子组件
<!-- TOC -->
算子组件指常用的算子及其操作，按功能大致可分为张量操作，网络操作，数组操作，图像操作，编码操作，调试操作，量化操作等七个模块。所有的算子在Ascend芯片或者CPU, GPU的支持情况，参见[这里](https://www.mindspore.cn/docs/zh-CN/master/operator_list.html "list")


这七类算子操作的相互关系见下:

<!-- TOC -->

- [算子组件](#算子组件)
    - [张量操作](#张量操作)
        - [标量运算](#标量运算)
            - [加法](#加法)
            - [Element-wise 除法](#element-wise-除法)
            - [Element-wise 乘](#element-wise-乘)
            - [三角函数](#求三角函数)
        - [向量运算](#向量运算)
            - [Concat](#concat-算子)
            - [Squeeze](#squeeze)
            - [Sparse2Dense](#求sparse2dense改变tensor维度使其变稠密)
            - [ScalarCast](#scalarcast)
        - [矩阵运算](#矩阵运算)
            - [矩阵乘法](#矩阵乘法)
            - [常见范数](#常见范数)
            - [广播机制](#广播机制)
    - [网络操作](#网络操作)
        - [特征提取](#特征提取)
            - [卷积操作](#卷积操作)
            - [卷积的反向传播操作](#卷积的反向传播算子操作)
        - [激活函数](#激活函数)
        - [LossFunction](#lossfunction)
            - [L1 Loss](#l1loss)
        - [优化算法](#优化算法)
            - [SGD](#sgd)
    - [数组操作](#数组操作)
        - [DType](#dtype)
        - [Cast](#cast)
        - [Shape](#shape)
    - [图像操作](#图像操作)
    - [编码运算](#编码运算)
        - [BoundingBoxEncode](#boundingboxencode)
        - [BoundingBoxDecode](#boundingboxdecode)
        - [IOU](#iou-计算)
    - [调试操作](#调试操作)
        - [Debug](#debug)
        - [HookBackward](#hookbackward)
    - [量化操作](#量化操作)
        - [MinMaxUpdatePerLayer](#minmaxupdateperlayer)

<!-- /TOC -->



## 张量操作

<!-- /TOC -->
主要包括张量的结构操作和张量的数学运算。
张量结构操作诸如：张量创建，索引切片，维度变换，合并分割。
张量数学运算主要有：标量运算，向量运算，矩阵运算。另外我们会介绍张量运算的广播机制。
本篇我们介绍张量的数学运算。
<!-- /TOC -->
<a href="https://gitee.com/mindspore/docs/blob/master/api/source_zh_cn/programming_guide/operator.md" target="_blank"><img src="../_static/logo_source.png"></a>

### 标量运算
张量的数学运算符可以分为标量运算符、向量运算符、以及矩阵运算符。
加减乘除乘方，以及三角函数，指数，对数等常见函数，逻辑比较运算符等都是标量运算符。
标量运算符的特点是对张量实施逐元素运算。
有些标量运算符对常用的数学运算符进行了重载。并且支持类似numpy的广播特性。

举例说明:
```python
import numpy as np            
import mindspore             # 导入mindspore包
from mindspore import Tensor # 导入mindspore下的Tensor包
import mindspore.ops.operations as P
input_x = mindspore.Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
input_y = 3.0
input_x**input_y
```

真实输入为：
```python
print(input_x)
[ 1.  8. 64.]
```

真实输出为：
```python
print(input_x**input_y)
[ 1.  8. 64.]
```

#### 加法
```python
input_x + input_y
[4.0 5.0 7.0]
```

除普通加外，还有element-wise加法:
```python
net = NetAddN()
input_x = Tensor(np.array([1, 2, 3]), mindspore.float32)
input_y = Tensor(np.array([4, 5, 6]), mindspore.float32)
net(input_x, input_y, input_x, input_y)[10.0, 14.0, 18.0]
```

#### Element-wise 除法
```python
input_x = Tensor(np.array([-4.0, 5.0, 6.0]), mindspore.float32)
input_y = Tensor(np.array([3.0, 2.0, 3.0]), mindspore.float32)
div = P.Div()
div(input_x, input_y)
```

求FloorDiv:
```python
input_x = Tensor(np.array([2, 4, -1]), mindspore.int32))
input_y = Tensor(np.array([3, 3, 3]), mindspore.int32)
floor_div = P.FloorDiv()
floor_div(input_x, input_y)[0, 1, -1]
```

#### Element-wise 乘
```python
input_x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
input_y = Tensor(np.array([4.0, 5.0, 6.0]), mindspore.float32)
mul = P.Mul()
mul(input_x, input_y)
```

真实输出:
```python
[4, 10, 18]
```

#### 求三角函数:
```python
acos = P.ACos()
input_x = Tensor(np.array([0.74, 0.04, 0.30, 0.56]), mindspore.float32)
output = acos(input_x)
```

### 向量运算
向量运算符只在一个特定轴上运算，将一个向量映射到一个标量或者另外一个向量。

#### Concat 算子:
```python
data1 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))
data2 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))
op = P.Concat()
output = op((data1, data2))
```

#### Squeeze
```python
input_tensor = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
squeeze = P.Squeeze(2)
output = squeeze(input_tensor)
```

#### 求Sparse2Dense(改变tensor维度使其变稠密):
```python
indices = Tensor([[0, 1], [1, 2]])
values = Tensor([1, 2], dtype=ms.float32)
dense_shape = (3, 4)
out = P.SparseToDense()(indices, values, dense_shape)
```

#### ScalarCast:
```python
scalar_cast = P.ScalarCast()
output = scalar_cast(255.0, mindspore.int32)
```

### 矩阵运算
矩阵运算包括: 矩阵乘法，矩阵范数，矩阵行列式，矩阵求特征值，矩阵分解等运算。

#### 矩阵乘法:
```python
input_x = Tensor(np.ones(shape=[1, 3]), mindspore.float32)
input_y = Tensor(np.ones(shape=[3, 4]), mindspore.float32)
matmul = P.MatMul()
output = matmul(input_x, input_y)
```

#### 常见范数:

```python
input_x = Tensor(np.ones([128, 64, 32, 64]), mindspore.float32)
scale = Tensor(np.ones([64]), mindspore.float32)
bias = Tensor(np.ones([64]), mindspore.float32)
mean = Tensor(np.ones([64]), mindspore.float32)
variance = Tensor(np.ones([64]), mindspore.float32)
batch_norm = P.BatchNorm()
output = batch_norm(input_x, scale, bias, mean, variance)
```

#### 广播机制

Broadcast 广播一个tensor到整个group
举例说明:
```python
from mindspore import Tensor
from mindspore.communication import init
import mindspore.nn as nn
import mindspore.ops.operations as P
init()
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.broadcast = P.Broadcast(1)

    def construct(self, x):
        return self.broadcast((x,))

input_ = Tensor(np.ones([2, 8]).astype(np.float32))
net = Net()
output = net(input_)
```

## 网络操作

<!-- /TOC -->
网络操作包括特征提取, 激活函数， LossFunction,  优化算法等：

### 特征提取

#### 卷积操作
举例说明:
```python
input = Tensor(np.ones([10, 32, 32, 32]), mindspore.float32)
weight = Tensor(np.ones([32, 32, 3, 3]), mindspore.float32))
conv2d = P.Conv2D(out_channel=32, kernel_size=3)
conv2d(input, weight)
```

#### 卷积的反向传播算子操作：
输出结果:
```python
dout = Tensor(np.ones([10, 32, 30, 30]), mindspore.float32)
weight = Tensor(np.ones([32, 32, 3, 3]), mindspore.float32)
x = Tensor(np.ones([10, 32, 32, 32]))
conv2d_backprop_input = P.Conv2DBackpropInput(out_channel=32, kernel_size=3)
conv2d_backprop_input(dout, weight, F.shape(x))
```

### 激活函数
举例说明:
```python
input_x = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
softmax = P.Softmax()
softmax(input_x)
```

输出结果:
```python
[0.01165623, 0.03168492, 0.08612854, 0.23412167, 0.6364086]
```

### LossFunction

#### L1Loss:
举例说明：
```python
loss = P.SmoothL1Loss()
input_data = Tensor(np.array([1, 2, 3]), mindspore.float32)
target_data = Tensor(np.array([1, 2, 2]), mindspore.float32)
loss(input_data, target_data)
```

输出结果:
```python
[0, 0, 0.5]
```

### 优化算法
#### SGD:
```python
sgd = P.SGD()
parameters = Tensor(np.array([2, -0.5, 1.7, 4]), mindspore.float32)
gradient = Tensor(np.array([1, -1, 0.5, 2]), mindspore.float32)
learning_rate = Tensor(0.01, mindspore.float32)
accum = Tensor(np.array([0.1, 0.3, -0.2, -0.1]), mindspore.float32)
momentum = Tensor(0.1, mindspore.float32)
stat = Tensor(np.array([1.5, -0.3, 0.2, -0.7]), mindspore.float32)
result = sgd(parameters, gradient, learning_rate, accum, momentum, stat)
```

## 数组操作

<!-- /TOC -->

数组操作指操作对象是一些数组的操作。

### DType 
返回跟输入的数据类型一致的并且适配Mindspore的tensor变量， 常用于Mindspore 工程内。
举例说明:
```python
input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
type = P.DType()(input_tensor)
```

### Cast
转换输入的数据类型并且输出与目标数据类型相同的变量
举例说明:
```python
input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
input_x = Tensor(input_np)
type_dst = mindspore.float16
cast = P.Cast()
result = cast(input_x, type_dst)
```

### Shape 
返回输入数据的形状
举例说明:
```python
input_tensor = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
shape = P.Shape()
output = shape(input_tensor)
```

## 图像操作

<!-- /TOC -->
图像操作包括图像预处理操作， 如图像剪切（Crop，便于得到大量训练样本）和大小变化（Reise,用于构建图像金子塔等）：

举例说明:
```python
class CropAndResizeNet(nn.Cell):
    def __init__(self, crop_size):
        super(CropAndResizeNet, self).__init__()
        self.crop_and_resize = P.CropAndResize()
        self.crop_size = crop_size
    @ms_function
    def construct(self, x, boxes, box_index):
        return self.crop_and_resize(x, boxes, box_index, self.crop_size)

BATCH_SIZE = 1
NUM_BOXES = 5
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHANNELS = 3
image = np.random.normal(size=[BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS]).astype(np.float32)
boxes = np.random.uniform(size=[NUM_BOXES, 4]).astype(np.float32)
box_index = np.random.uniform(size=[NUM_BOXES], low=0, high=BATCH_SIZE).astype(np.int32)
crop_size = (24, 24)
crop_and_resize = CropAndResizeNet(crop_size=crop_size)
output = crop_and_resize(Tensor(image), Tensor(boxes), Tensor(box_index))
print(output.asnumpy())
```

## 编码运算

<!-- /TOC -->
编码运算包括 BoundingBox Encoding和 BoundingBox Decoding， IOU计算等。

### BoundingBoxEncode
对物体所在区域方框进行编码，得到类似PCA的更精简信息，以便做后续类似特征提取，物体检测，图像恢复等任务。

举例说明:
```python
anchor_box = Tensor([[4,1,2,1],[2,2,2,3]],mindspore.float32)
groundtruth_box = Tensor([[3,1,2,2],[1,2,1,4]],mindspore.float32)
boundingbox_encode = P.BoundingBoxEncode(means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0))
boundingbox_encode(anchor_box, groundtruth_box)
```
输出结果为:
```python
[[5.0000000e-01  5.0000000e-01  -6.5504000e+04  6.9335938e-01]
 [-1.0000000e+00  2.5000000e-01  0.0000000e+00  4.0551758e-01]]
```

### BoundingBoxDecode  
编码器对区域位置信息解码之后，用此算子进行解码。

举例说明:
```python
anchor_box = Tensor([[4,1,2,1],[2,2,2,3]],mindspore.float32)
deltas = Tensor([[3,1,2,2],[1,s2,1,4]],mindspore.float32)
boundingbox_decode = P.BoundingBoxDecode(means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0), max_shape=(768, 1280), wh_ratio_clip=0.016)
boundingbox_decode(anchor_box, deltas)
```
输出结果:
```python
[[4.1953125  0.  0.  5.1953125]
 [2.140625  0.  3.859375  60.59375]]
```

### IOU 计算：
计算预测的物体所在方框和真实物体所在方框的交集区域与并集区域的占比大小。其常作为一种损失函数，用以优化模型。

举例说明:
```python
iou = P.IOU()
anchor_boxes = Tensor(np.random.randint(1.0, 5.0, [3, 4]), mindspore.float16)
gt_boxes = Tensor(np.random.randint(1.0, 5.0, [3, 4]), mindspore.float16)
```

## 调试操作
调试操作指的是用于调试网络的一些常用算子及其操作， 例如Debug等

### Debug
输出tensor变量的数值， 方便用户随时随地打印想了解或者debug必需的某变量数值。

参考示例:
```python
class DebugNN(nn.Cell):
    def __init__(self,):
        self.debug = nn.Debug()

    def construct(self, x, y):
        x = self.add(x, y)
        self.debug(x)
        return x
```

### HookBackward
打印中间变量的梯度，这一算子特别常用，遂举例在此，虽目前仅支持Pynative 形式
参考示例:
```python
def hook_fn(grad_out):
    print(grad_out)

grad_all = GradOperation(get_all=True)
hook = P.HookBackward(hook_fn)

def hook_test(x, y):
    z = x * y
    z = hook(z)
    z = z * y
    return z

def backward(x, y):
    return grad_all(hook_test)(x, y)

backward(1, 2)
```

## 量化操作

<!-- /TOC -->
量化操作指对tensor做量化或者反量化操作。 量化操作指将浮点数用整数的加和表示，利用整数加和并行加速时速度快的优点， 实
现在可接受精度损失下的性能提升。反量化指其反过程，其在精度要求高的地方常被用到。

### MinMaxUpdatePerLayer 
完成在训练时的量化和反量化操作
举例说明:
```python
input_tensor = Tensor(np.random.rand(3, 16, 5, 5), mstype.float32)
min_tensor = Tensor(np.array([-6]), mstype.float32)
max_tensor = Tensor(np.array([6]), mstype.float32)
output_tensor = FakeQuantPerLayer(num_bits=8)(input_tensor, min_tensor, max_tensor)
```
