# 基于gRPC接口访问MindSpore Serving服务

`Linux` `Ascend` `Serving` `初级` `中级` `高级`

[![查看源文件](_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/tutorials/inference/source_zh_cn/serving_grpc.md)

## 概述

MindSpore Serving提供gRPC接口访问Serving服务。在Python环境下，我们提供[mindspore_serving.client](https://gitee.com/mindspore/serving/blob/r1.1/mindspore_serving/client/python/client.py) 模块用于填写请求、解析回复。gRPC服务端（worker节点）当前仅支持Ascend平台，客户端运行不依赖特定硬件环境。接下来我们通过`add`和`ResNet-50`样例来详细说明gRPC Python客户端接口的使用。

## add样例

样例来源于[add example](https://gitee.com/mindspore/serving/blob/r1.1/example/add/client.py) ，`add` Servable提供的`add_common`方法提供两个2x2 Tensor相加功能。其中gRPC Python客户端代码如下所示，一次gRPC请求包括了三对独立的2x2 Tensor：

```python
from mindspore_serving.client import Client
import numpy as np


def run_add_common():
    """invoke Servable add method add_common"""
    client = Client("localhost", 5500, "add", "add_common")
    instances = []

    # instance 1
    x1 = np.asarray([[1, 1], [1, 1]]).astype(np.float32)
    x2 = np.asarray([[1, 1], [1, 1]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2})

    # instance 2
    x1 = np.asarray([[2, 2], [2, 2]]).astype(np.float32)
    x2 = np.asarray([[2, 2], [2, 2]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2})

    # instance 3
    x1 = np.asarray([[3, 3], [3, 3]]).astype(np.float32)
    x2 = np.asarray([[3, 3], [3, 3]]).astype(np.float32)
    instances.append({"x1": x1, "x2": x2})

    result = client.infer(instances)
    print(result)


if __name__ == '__main__':
    run_add_common()
```

按照[入门流程](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.1/serving_example.html) 导出模型、启动Serving服务器，并执行上述客户端代码。当运行正常后，将打印以下结果，为了展示方便，格式作了调整：

```python
[{'y': array([[2., 2.], [2., 2.]], dtype=float32)},
 {'y': array([[4., 4.], [4., 4.]], dtype=float32)},
 {'y': array([[6., 6.], [6., 6.]], dtype=float32)}]
```

以下将对其中的细节进行说明。

1. 构造`Client`。

   构造`Client`时，指示Serving的ip和端口号，并给定Servable名称和它提供的方法。这里的Servable可以是单个模型，也可以是多个模型的组合，多个模型组合提供Servable当前尚未支持，一个Servable可以通过提供多种方法来提供不同的服务。

   上面的`add`样例， Serving运行在本地（`localhost`），指定的gRPC端口号为`5500`，运行了`add` Servable，`add` Servable提供了`add_common`方法。

2. 添加实例。

   每次gRPC请求可包括一个或多个实例，每个实例之间相互独立，结果互不影响。

   比如：`add` Servable提供的`add_common`方法提供两个2x2 Tensor相加功能，即一个实例包含两个2x2 Tensor输入，一个2x2 Tensor输出。一次请求可包括一个、两个或者多个这样的实例，针对每个实例返回一个结果。上述`add`样例提供了三个实例，预期将返回三个实例的结果。

   `Client.infer`接口入参可为一个或多个实例输入组成的list、tuple或者单个实例输入。每个实例输入由输入的名称和输入的值组成python字典，值可以是以下格式：

   - `numpy array`：用以表示Tensor。例如：np.ones((3,224), np.float32)。
   - `numpy number`：用以表示Scalar。例如：np.int8(5)。
   - `python bool int float`：用以表示Scalar, 当前int将作为int32, float将作为float32。例如：32.0。
   - `python str`：用以表示字符串。例如："this is a text"。
   - `python bytes`：用以表示二进制数据。例如：图片数据。

   上面的add样例，`add` Servable提供的`add_common`方法入参名为`x1`和`x2`，添加每个实例时指定每个输入的值。

3. 获取推理结果。

   通过`Client.infer`填入一个或多个实例。
   返回可能有以下形式：

   - 所有实例推理正确：

        ```shell
        [{'y': array([[2., 2.], [2., 2.]], dtype=float32)},
         {'y': array([[4., 4.], [4., 4.]], dtype=float32)},
         {'y': array([[6., 6.], [6., 6.]], dtype=float32)}]
        ```

   - 针对所有实例共同的错误，返回一个包含`error`的dict。将例子中Client构造时填入的`add_common`改为`add_common2`，将返回结果：

        ```shell
        {'error', 'Request Servable(add) method(add_common2), method is not available'}
        ```

   - 部分实例推理错误，出错的推理实例将返回包含`error`的dict。将instance2一个输入的`dtype`改为`np.int32`，将返回结果：

        ```shell
        [{'y': array([[2., 2.], [2., 2.]], dtype=float32)},
         {'error': 'Given model input 1 data type kMSI_Int32 not match ...'},
         {'y': array([[6., 6.], [6., 6.]], dtype=float32)}]
        ```

   每个实例返回一个dict，key的值来自于Servable的方法定义，例如本例子中，`add` Servable提供的`add_common`方法输出仅有一个，为`y`。value为以下格式：

   |  Serving输出类型 | Client返回类型   | 说明  |  举例  |
   |  ----  | ----  |  ---- | ---- |
   | Tensor | numpy array | tensor array | np.ones((3,224), np.float32) |
   | Scalar: <br>int8, int16, int32, int64, <br>uint8, uint16, uint32, uint64, <br>bool, float16, float32, float64 | numpy scalar | Scalar格式的数据转为numpy scalar | np.int8(5)  |
   | String | python str | 字符串格式输出转为python str | "news_car"  |
   | Bytes | python bytes | 二进制格式输出转为python bytes | 图片数据  |

## ResNet-50样例

样例来源于[ResNet-50 example](https://gitee.com/mindspore/serving/blob/r1.1/example/resnet/client.py)，`ResNet-50` Servable提供的`classify_top1`方法提供对图像进行识别的服务。`classify_top1`方法输入为图像数据，输出为字符串，方法中预处理对图像进行解码、Resize等操作，接着进行推理，并通过后处理返回得分最大的分类标签。

```python
import os
from mindspore_serving.client import Client


def run_classify_top1():
    client = Client("localhost", 5500, "resnet50", "classify_top1")
    instances = []
    for path, _, file_list in os.walk("./test_image/"):
        for file_name in file_list:
            image_file = os.path.join(path, file_name)
            print(image_file)
            with open(image_file, "rb") as fp:
                instances.append({"image": fp.read()})
    result = client.infer(instances)
    print(result)


if __name__ == '__main__':
    run_classify_top1()
```

`ResNet-50` Servable提供的`classify_top1`方法需要用户提供输入`image`，上面例子中，每个实例的输入`image`为图像的二进制数据。
正常结束执行后，预期将会有以下打印：

```shell
[{'label': 'tabby, tabby cat'}, {'label': 'ox'}]
```

如果Resnet50模型未训练，可能有其他未知分类结果。
