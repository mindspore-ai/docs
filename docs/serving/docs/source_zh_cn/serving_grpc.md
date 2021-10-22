# 基于gRPC接口访问MindSpore Serving服务

<!-- TOC -->

- [基于gRPC接口访问MindSpore Serving服务](#基于grpc接口访问mindspore-serving服务)
    - [概述](#概述)
    - [add样例](#add样例)
    - [ResNet-50样例](#resnet-50样例)
    - [通过Unix domain socket访问Serving服务器](#通过unix-domain-socket访问serving服务器)
    - [访问开启SSL/TLS的Serving服务](#访问开启ssltls的serving服务)
        - [单向认证](#单向认证)
        - [双向认证](#双向认证)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/serving/docs/source_zh_cn/serving_grpc.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## 概述

MindSpore Serving提供gRPC接口访问Serving服务。在Python环境下，我们提供[mindspore_serving.client](https://gitee.com/mindspore/serving/blob/r1.5/mindspore_serving/client/python/client.py) 模块用于填写请求、解析回复。gRPC服务端当前支持Ascend和Nvidia GPU平台，客户端运行不依赖特定硬件环境。接下来我们通过`add`和`ResNet-50`样例来详细说明gRPC Python客户端接口的使用。

## add样例

样例来源于[add example](https://gitee.com/mindspore/serving/blob/r1.5/example/tensor_add/serving_client.py) ，`add` Servable提供的`add_common`方法提供两个2x2 Tensor相加功能。其中gRPC Python客户端代码如下所示，一次gRPC请求包括了三对独立的2x2 Tensor：

```python
from mindspore_serving.client import Client
import numpy as np


def run_add_common():
    """invoke Servable add method add_common"""
    client = Client("localhost:5500", "add", "add_common")
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

按照[入门流程](https://www.mindspore.cn/serving/docs/zh-CN/r1.5/serving_example.html) 导出模型、启动Serving服务器，并执行上述客户端代码。当运行正常后，将打印以下结果，为了展示方便，格式作了调整：

```text
[{'y': array([[2., 2.], [2., 2.]], dtype=float32)},
 {'y': array([[4., 4.], [4., 4.]], dtype=float32)},
 {'y': array([[6., 6.], [6., 6.]], dtype=float32)}]
```

以下将对其中的细节进行说明。

1. 构造`Client`。

   构造`Client`时，指示Serving的网络地址，并给定Servable名称和它提供的方法。这里的Servable可以是单个模型，也可以是多个模型的组合，多个模型组合提供Servable当前尚未支持，一个Servable可以通过提供多种方法来提供不同的服务。

   上面的`add`样例， Serving运行在本地（`localhost`），指定的gRPC端口号为`5500`，运行了`add` Servable，`add` Servable提供了`add_common`方法。

2. 添加实例。

   每次gRPC请求可包括一个或多个实例，每个实例之间相互独立，结果互不影响。

   比如：`add` Servable提供的`add_common`方法提供两个2x2 Tensor相加功能，即一个实例包含两个2x2 Tensor输入，一个2x2 Tensor输出。一次请求可包括一个、两个或者多个这样的实例，针对每个实例返回一个结果。上述`add`样例提供了三个实例，预期将返回三个实例的结果。

   `Client.infer`接口入参可为一个或多个实例输入组成的list、tuple或者单个实例输入。每个实例输入由输入的名称和输入的值组成python字典，值可以是以下格式：

   - `numpy array`：用以表示Tensor。例如：np.ones((3,224), np.float32)。
   - `numpy number`：用以表示Scalar。例如：np.int8(5)。
   - `python bool int float`：用以表示Scalar, 当前int将作为int64, float将作为float64。例如：32.0。
   - `python str`：用以表示字符串。例如："this is a text"。
   - `python bytes`：用以表示二进制数据。例如：图片数据。

   上面的add样例，`add` Servable提供的`add_common`方法入参名为`x1`和`x2`，添加每个实例时指定每个输入的值。

3. 获取推理结果。

   通过`Client.infer`填入一个或多个实例。
   返回可能有以下形式：

   - 所有实例推理正确：

        ```text
        [{'y': array([[2., 2.], [2., 2.]], dtype=float32)},
         {'y': array([[4., 4.], [4., 4.]], dtype=float32)},
         {'y': array([[6., 6.], [6., 6.]], dtype=float32)}]
        ```

   - 针对所有实例共同的错误，返回一个包含`error`的dict。将例子中Client构造时填入的`add_common`改为`add_common2`，将返回结果：

        ```text
        {'error', 'Request Servable(add) method(add_common2), method is not available'}
        ```

   - 部分实例推理错误，出错的推理实例将返回包含`error`的dict。将instance2一个输入的`dtype`改为`np.int32`，将返回结果：

        ```text
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

样例来源于[ResNet-50 example](https://gitee.com/mindspore/serving/blob/r1.5/example/resnet/serving_client.py)，`ResNet-50` Servable提供的`classify_top1`方法提供对图像进行识别的服务。`classify_top1`方法输入为图像数据，输出为字符串，方法中预处理对图像进行解码、Resize等操作，接着进行推理，并通过后处理返回得分最大的分类标签。

```python
import os
from mindspore_serving.client import Client


def run_classify_top1():
    client = Client("localhost:5500", "resnet50", "classify_top1")
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

```text
[{'label': 'tabby, tabby cat'}, {'label': 'ox'}]
```

如果Resnet50模型未训练，可能有其他未知分类结果。

## 通过Unix domain socket访问Serving服务器

MindSpore Serving服务器和客户端可通过TCP/IP进行通信，当它们在一个机器内部时，也可通过Unix domain socket的方式通信，提升通讯性能。

Serving服务器启动gRPC服务时，`mindspore_serving.server.start_grpc_server`的`address`参数填写为`'unix:{some_file_path}'`作为gRPC服务的访问地址，其中`{some_file_path}`为相对或者绝对的文件路径，文件所在目录需要已经存在，接口成功调用后，文件将被复写。同时`mindspore_serving.client.Client`的`address`参数填写为上述的地址。比如：

服务器：

```python
import os
import sys
from mindspore_serving import server


def start():
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

    servable_config = server.ServableStartConfig(servable_directory=servable_dir, servable_name="resnet50",
                                                 device_ids=(0, 1))
    server.start_servables(servable_configs=servable_config)

    server.start_grpc_server(address="unix:/tmp/serving_resnet50_test_temp_file")


if __name__ == "__main__":
    start()
```

客户端：

```python
import os
from mindspore_serving.client import Client


def run_classify_top1():
    client = Client("unix:/tmp/serving_resnet50_test_temp_file", "resnet50", "classify_top1")
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

## 访问开启SSL/TLS的Serving服务

Mindspore Serving的服务器和客户端可以通过`SSL/TLS`协议进行通信。

`SSL/TLS`是一个安全通信协议，可以用来验证客户端或服务器的身份，加密所有的数据，保证通信的安全。数字证书用来标识服务器或客户端的身份，私钥用来解密数据和对信息摘要进行签名。我们可以用openssl来生成服务器与客户端相关的私钥和证书。

下面举个例子展示如何生成证书并进行单双向认证：

### 单向认证

仅客户端验证服务器的身份，所以我们需要服务器的证书和私钥。可以执行下面的openssl命令来生成相关证书。

```shell
# 生成根证书 用来签发服务器或客户端的证书
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ca.key -out ca.crt -subj "/C=CN/ST=xx/L=xx/OU=gRPC/CN=Root"

# 生成服务器的私钥
openssl genrsa -out server.key 2048
# 生成服务器证书签名请求
# 参数CN可以自定义证书上服务器名，这里我们可以配置成localhost，gRPC客户端访问时地址需要设置为localhost
openssl req -new -key server.key -out server.csr -subj "/C=XX/ST=MyST/L=XX/O=HW/OU=gRPC/CN=localhost"
# 使用根证书签发服务器证书
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365 -sha256
```

我们得到了`server.key`，`server.crt`和`ca.crt`三个文件。将他们传入对应的`SSLConfig`。

- 服务器：

  ```python
  import os
  import sys
  from mindspore_serving import server


  def start():
      servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

      servable_config = server.ServableStartConfig(servable_directory=servable_dir, servable_name="add",
                                                   device_ids=(0, 1))
      server.start_servables(servable_configs=servable_config)

      ssl_config = server.SSLConfig(certificate="server.crt", private_key="server.key", custom_ca=None, verify_client=False)

      server.start_grpc_server(address="127.0.0.1:5500", ssl_config=ssl_config)


  if __name__ == "__main__":
      start()
  ```

    - `ssl_config`表示服务器的`SSL`配置。该参数默认为`None`，表示不开启`SSL/TLS`。开启`SSL/TLS`则需要传入`mindspore_serving.server.SSLConfig`对象。
    - `certificate`为服务器证书文件的路径。
    - `private_key`为服务器私钥文件的路径。
    - `custom_ca`为服务器的根证书文件的路径，用来验证客户端的身份。当`verify_client` 的为`True`时，需要验证客户端的证书，所以该参数不能为`None`，必须传入对应的路径。
    - `verify_client`表示是否验证客户端的身份。

  将`verify_client`设为`False`表示单向认证。我们分别传入服务器的证书`server.crt`和私钥`server.key`，由于服务器不需要验证客户端的证书，此时服务器的`custom_ca`参数会被忽略。

- 客户端：

  ```python
  from mindspore_serving.client import Client
  from mindspore_serving.client import SSLConfig
  import numpy as np


  def run_add_common():
      """invoke Servable add method add_common"""
      ssl_config = SSLConfig(custom_ca="ca.crt")
      client = Client("localhost:5500", "add", "add_common", ssl_config=ssl_config)
      instances = []

      # instance 1
      x1 = np.asarray([[1, 1], [1, 1]]).astype(np.float32)
      x2 = np.asarray([[1, 1], [1, 1]]).astype(np.float32)
      instances.append({"x1": x1, "x2": x2})

      result = client.infer(instances)
      print(result)


  if __name__ == '__main__':
      run_add_common()
  ```

    - `ssl_config`表示客户端的`SSL`配置。该参数默认为`None`，表示不开启`SSL/TLS`。开启`SSL/TLS`则需要传入`mindspore_serving.client.SSLConfig`对象。
    - `certificate`为客户端证书文件的路径。
    - `private_key`为客户端私钥文件的路径。
    - `custom_ca`为客户端的根证书文件的路径，用来验证服务器的身份。该参数可以为`None`，这个时候gRPC会通过gRPC安装路径下的`grpc/_cython/_credentials/roots.pem`文件或`GRPC_DEFAULT_SSL_ROOTS_FILE_PATH`环境变量找到对应的根证书。

  由于仅客户端验证服务器证书，所以只需要将`custom_ca`设置为签发服务器证书的`ca.crt`。

### 双向认证

客户端和服务器都需要验证对方的身份，所以除了服务器的证书，我们还需要执行下面的命令生成客户端的证书。

```shell
# 生成客户端的私钥
openssl genrsa -out client.key 2048
# 生成客户端证书签名请求
openssl req -new -key client.key -out client.csr -subj "/C=XX/ST=MyST/L=XX/O=HW/OU=gRPC/CN=client"
# 使用根证书签发客户端证书
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt -days 365 -sha256
```

我们得到了`client.key`和`cleint.crt`。

- 服务器：

  ```python
  import os
  import sys
  from mindspore_serving import server


  def start():
      servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

      servable_config = server.ServableStartConfig(servable_directory=servable_dir, servable_name="add",
                                                   device_ids=(0, 1))
      server.start_servables(servable_configs=servable_config)

      ssl_config = server.SSLConfig(certificate="server.crt", private_key="server.key", custom_ca="ca.crt", verify_client=True)

      server.start_grpc_server(address="127.0.0.1:5500", ssl_config=ssl_config)


  if __name__ == "__main__":
      start()
  ```

  将`verify_client`设为`True`表示双向认证。同时将`custom_ca`设置为`ca.crt`来验证客户端证书。

- 客户端：

  ```python
  from mindspore_serving.client import Client
  from mindspore_serving.client import SSLConfig
  import numpy as np


  def run_add_common():
      """invoke Servable add method add_common"""
      ssl_config = SSLConfig(certificate="client.crt", private_key="client.key", custom_ca="ca.crt")
      client = Client("localhost:5500", "add", "add_common", ssl_config=ssl_config)
      instances = []

      # instance 1
      x1 = np.asarray([[1, 1], [1, 1]]).astype(np.float32)
      x2 = np.asarray([[1, 1], [1, 1]]).astype(np.float32)
      instances.append({"x1": x1, "x2": x2})

      result = client.infer(instances)
      print(result)


  if __name__ == '__main__':
      run_add_common()
  ```

  客户端需要提供自己的证书给服务器验证，我们分别传入客户端的证书`client.crt`和私钥`client.key`。

当gRPC服务器与客户端`SSL/TLS`开启状态不一致的时候，服务器或客户端会出现`ssl3_get_record:wrong version number`的错误，这时需要确认服务器与客户端是否都开启了`SSL/TLS`。
