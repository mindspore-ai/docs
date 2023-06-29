# gRPC-based MindSpore Serving Access

`Linux` `Ascend` `GPU` `Serving` `Beginner` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r1.2/tutorials/inference/source_en/serving_grpc.md" target="_blank"><img src="_static/logo_source.png"></a>

## Overview

The gRPC API is provided to access the MindSpore Serving. In the Python environment, the [mindspore_serving.client](https://gitee.com/mindspore/serving/blob/r1.2/mindspore_serving/client/python/client.py) module is provided to fill in requests and parse responses. The gRPC server (a worker node) supports only the Ascend platform. The client running does not depend on a specific hardware environment. The following uses `add` and `ResNet-50` as examples to describe how to use the gRPC Python API on clients.

## add

This example comes from [add example](https://gitee.com/mindspore/serving/blob/r1.2/example/add/client.py). The `add` Servable provides the `add_common` method to add up two 2x2 tensors. The code of the gRPC Python client is as follows. One gRPC request includes three pairs of independent 2x2 tensors.

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

Export the model, start the Serving server, and run the preceding client code by referring to the [MindSpore Serving-based Inference Service Deployment](https://www.mindspore.cn/tutorial/inference/en/r1.2/serving_example.html). After the code runs properly, the following information is displayed. To facilitate display, the format is adjusted.

```python
[{'y': array([[2., 2.], [2., 2.]], dtype=float32)},
 {'y': array([[4., 4.], [4., 4.]], dtype=float32)},
 {'y': array([[6., 6.], [6., 6.]], dtype=float32)}]
```

Details are described as follows:

1. Build `Client`.

   When `Client` is built, the IP address and port number of Serving are indicated, and the Servable's name and method it provides are given. Servable indicates a single model or a combination of multiple models (not supported yet) and provides different services in various methods.

   In the preceding `add` example, Serving runs on the `localhost`, the gRPC port number is set to `5500`, and `add` Servable is run to provide the `add_common` method.

2. Add instances.

   Each gRPC request includes one or more independent instances which do not affect each other's result.

   For example, the `add_common` method provided by `add` Servable provides the function of adding two 2x2 tensors. That is, an instance contains two 2x2 tensor inputs and one 2x2 tensor output. A request may include one or more such instances, and one result is returned for each instance. The preceding `add` example provides three instances, so it is expected that three results will be returned.

   The input parameters of the `Client.infer` API can be a list or tuple consisting of one or more instance inputs, or a single instance input. Each instance input consists of the input name and value. The value can be in the following format:

   - `numpy array`: represents a tensor. For example, np.ones((3,224), np.float32).
   - `numpy number`: represents a scalar. For example, np.int8(5).
   - `python bool int float`: represents a scalar. Currently, int is regarded as int64, and float is regarded as float64. For example, 32.0.
   - `python str`: represents a character string. For example, "this is a text".
   - `python bytes`: represents binary data. For example, image data.

   In the preceding example, `x1` and `x2` are the input parameters of the `add_common` method provided by `add` Servable. Each input value is specified when an instance is added.

3. Obtain the inference result.

   Use `Client.infer` to enter one or more instances.
   The return results may be in the following format:

   - If all instances are correctly inferred, the following result is returned:

        ```shell
        [{'y': array([[2., 2.], [2., 2.]], dtype=float32)},
         {'y': array([[4., 4.], [4., 4.]], dtype=float32)},
         {'y': array([[6., 6.], [6., 6.]], dtype=float32)}]
        ```

   - If certain errors occur in all instances , a dict containing `error` is returned. In the example, `add_common` is changed to `add_common2`, and the returned result is as follows:

        ```shell
        {'error', 'Request Servable(add) method(add_common2), method is not available'}
        ```

   - If inference errors occur in certain instances, the error instances return a dict containing `error`. In the example, an input `dtype` of instance2 is changed to `np.int32`, and the returned result is as follows:

        ```shell
        [{'y': array([[2., 2.], [2., 2.]], dtype=float32)},
         {'error': 'Given model input 1 data type kMSI_Int32 not match ...'},
         {'y': array([[6., 6.], [6., 6.]], dtype=float32)}]
        ```

   Each instance returns a dict. The key value comes from the Servable method definition. In this example, the `add_common` method provided by `add` Servable has only one output, which is `y`. The value is in the following format:

   | Serving Output Type | Client Return Type | Description | Example |
   |  ----  | ----  |  ---- | ---- |
   | Tensor | numpy array | Tensor array | np.ones((3,224), np.float32) |
   | Scalar: <br>int8, int16, int32, int64, <br>uint8, uint16, uint32, uint64, <br>bool, float16, float32, float64 | numpy scalar | Converts data format from scalar to numpy scalar. | np.int8(5)  |
   | String | python str | Converts output format from character string to python str. | "news_car"  |
   | Bytes | python bytes | Converts output format from binary to python bytes. | Image data  |

## ResNet-50

This example comes from [ResNet-50 example](https://gitee.com/mindspore/serving/blob/r1.2/example/resnet/client.py). `ResNet-50` Servable provides the `classify_top1` method to recognize images. In the `classify_top1` method, input the image data to obtain the output character string, perform operations such as decoding and resizing on images, and then perform inference. The classification label with the highest score is returned through post-processing.

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

The preceding `classify_top1` method requires users to input `image` (the image binary data) in each instance.
If the execution is properly completed, the following information is displayed:

```shell
[{'label': 'tabby, tabby cat'}, {'label': 'ox'}]
```

If the ResNet-50 model is not trained, there may be other unknown classification results.
