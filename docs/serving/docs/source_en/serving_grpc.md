# gRPC-based MindSpore Serving Access

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/serving/docs/source_en/serving_grpc.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

## Overview

The gRPC API is provided to access the MindSpore Serving. In the Python environment, the [mindspore_serving.client](https://gitee.com/mindspore/serving/blob/r1.6/mindspore_serving/client/python/client.py) module is provided to fill in requests and parse responses. The gRPC server supports the Ascend and Nvidia GPU platform. The client running does not depend on a specific hardware environment. The following uses `add` and `ResNet-50` as examples to describe how to use the gRPC Python API on clients.

## add

This example comes from [add example](https://gitee.com/mindspore/serving/blob/r1.6/example/tensor_add/serving_client.py). The `add` Servable provides the `add_common` method to add up two 2x2 tensors. The code of the gRPC Python client is as follows. One gRPC request includes three pairs of independent 2x2 tensors.

```python
from mindspore_serving.client import Client
import numpy as np


def run_add_common():
    """invoke Servable add method add_common"""
    client = Client("127.0.0.1:5500", "add", "add_common")
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

Export the model, start the Serving server, and run the preceding client code by referring to the [MindSpore Serving-based Inference Service Deployment](https://www.mindspore.cn/serving/docs/en/r1.6/serving_example.html). After the code runs properly, the following information is displayed. To facilitate display, the format is adjusted.

```text
[{'y': array([[2., 2.], [2., 2.]], dtype=float32)},
 {'y': array([[4., 4.], [4., 4.]], dtype=float32)},
 {'y': array([[6., 6.], [6., 6.]], dtype=float32)}]
```

Details are described as follows:

1. Build `Client`.

   When `Client` is built, the network address of Serving are indicated, and the Servable's name and method it provides are given. Servable indicates a single model or a combination of multiple models (not supported yet) and provides different services in various methods.

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

        ```text
        [{'y': array([[2., 2.], [2., 2.]], dtype=float32)},
         {'y': array([[4., 4.], [4., 4.]], dtype=float32)},
         {'y': array([[6., 6.], [6., 6.]], dtype=float32)}]
        ```

   - If certain errors occur in all instances , a dict containing `error` is returned. In the example, `add_common` is changed to `add_common2`, and the returned result is as follows:

        ```text
        {'error', 'Request Servable(add) method(add_common2), method is not available'}
        ```

   - If inference errors occur in certain instances, the error instances return a dict containing `error`. In the example, an input `dtype` of instance2 is changed to `np.int32`, and the returned result is as follows:

        ```text
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

This example comes from [ResNet-50 example](https://gitee.com/mindspore/serving/blob/r1.6/example/resnet/serving_client.py). `ResNet-50` Servable provides the `classify_top1` method to recognize images. In the `classify_top1` method, input the image data to obtain the output character string, perform operations such as decoding and resizing on images, and then perform inference. The classification label with the highest score is returned through post-processing.

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

The preceding `classify_top1` method requires users to input `image` (the image binary data) in each instance.
If the execution is properly completed, the following information is displayed:

```text
[{'label': 'tabby, tabby cat'}, {'label': 'ox'}]
```

If the ResNet-50 model is not trained, there may be other unknown classification results.

## Accessing Serving Server through Unix Domain Socket

MindSpore Serving server and client can communicate through TCP/IP. When they are inside one machine, they can also communicate through Unix domain socket to improve the communication performance.

When the serving server starts the grpc service, the `address` parameter of `mindspore_serving.server.start_grpc_server` should be filled with `'unix:{some_file_path}'` as the access address of the gRPC service, where `{some_file_path}` is a relative or absolute file path, and the directory where the file is located must already exist. After the interface is successfully called, the file will be overwrited. At the same time, the 'address' parameter of `mindspore_serving.client.Client` should be filled with the above address. For example:

The Server:

```python
import os
import sys
from mindspore_serving import server


def start():
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

    servable_config = server.ServableStartConfig(servable_directory=servable_dir, servable_name="resnet50",
                                                 device_ids=(0, 1))
    server.start_servables(servable_configs=servable_config)

    server.start_grpc_server(address="unix:/tmp/resnet50_test_temp_file")


if __name__ == "__main__":
    start()
```

The Client:

```python
import os
from mindspore_serving.client import Client


def run_classify_top1():
    client = Client("unix:/tmp/resnet50_test_temp_file", "resnet50", "classify_top1")
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

## Accessing SSL/TLS enabled Serving Service

Mindspore Serving supports server and client communicating based on `SSL/TLS`.

`SSL/TLS`is a secure communication protocol that can be used to verify the identity of a client or server, encrypt all data, and secure communication.
Digital certificates are used to identify the server or client, and private keys are used to decrypt data and sign information digests.
We can use openssl to generate the private keys and certificates related to server and client.

Here's an example of how to generate a certificate and perform single-bidirectional authentication:

### One-way authentication

Only the client verifies the identity of the server, so we need the server's certificate and private key.
You can execute the following openssl command to generate the relevant certificate.

```shell
# Generate the root certificate used to issue the certificate of server or client
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ca.key -out ca.crt -subj "/C=CN/ST=xx/L=xx/OU=gRPC/CN=Root"

# Generate server's private key
openssl genrsa -out server.key 2048
# Generate server's certificate sign request
# You can customize the server name on the certificate by setting CN (Common Name). In this case we can set CN to localhost.
# When the gRPC client accesses the server with this certificate, address needs to be localhost.
openssl req -new -key server.key -out server.csr -subj "/C=XX/ST=MyST/L=XX/O=HW/OU=gRPC/CN=localhost"
# Use the root certificate to issue a server certificate
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365 -sha256
```

We got `server.key`, `server.crt` and `ca.crt` files. Pass them to the corresponding `SSLConfig`.

- Server:

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

    - `ssl_config` represents the server's `SSL` configuration. This parameter defaults to `None`, which means `SSL/TLS` is not enabled.
      Enabling `SSL/TLS` requires `mindspore serving.server.SSLConfig` object passed to this parameter.
    - `certificate` is the path to the server's certificate file.
    - `private_key` is the path to the server's private key file.
    - `custom_ca` is the path to the server's root certificate file which is for verifying client certificate. When `verify_client` is `True`,
      the client's certificate needs to be verified, so this parameter can't be `None`, the corresponding path must be passed in.
    - `verify_client` indicates whether to verify the identity of the client.

  Setting `verify_client` to `False` represents one-way authentication. We pass in the certificate `server.crt` and the private key `server.key`, respectively.
  Due to the server does not need to verify the client so `custom_ca` is ignored.

- Client:

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

    - `ssl_config` represents the client's `SSL` configuration. This parameter defaults to `None`, which means `SSL/TLS` is not enabled.
      Enabling `SSL/TLS` requires passing `mindspore_serving.client.SSLConfig` object to `ssl_config`.
    - `certificate` is the path to the client's certificate file.
    - `private_key` is the path to the client's private key file.
    - `custom_ca` is the path to the client's root certificate file, which is used to verify the identity of the server.
      This parameter can be `None`, at which point gRPC finds the corresponding root certificate through the `grpc/_cython/_credentials/roots.pem` file under the gRPC installation path or
      the `GRPC_DEFAULT_SSL_ROOTS_FILE_PATH` environment variable.

  Because only the client verifies the server certificate, you only need to set `custom_ca` to `ca.crt` which issues the server's certificate.

### Mutual authentication

Both the client and the server need to verify each other's identity, so in addition to the server's certificate,
we need to execute the following command to generate the client's certificate.

```shell
# Generate client's private key
openssl genrsa -out client.key 2048
# Generate client's certificate sign request
openssl req -new -key client.key -out client.csr -subj "/C=XX/ST=MyST/L=XX/O=HW/OU=gRPC/CN=client"
# Use root certificate to issue client's certificate
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt -days 365 -sha256
```

We got `client.key`and`cleint.crt`.

- Server:

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

  Setting `verify_client` to `True` represents two-way authentication. Also set `custom_ca` to `ca.crt` to verify the client certificate.

- Client:

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

  The client needs providing its own certificate to the server for authentication, and we pass in the client's certificate `client.crt` and the private key `client.key`, respectively.

When the gRPC server and client are not enabling `SSL/TLS` at the same time, the server side or client side will get `ssl3_get_record:wrong version number` error,
and you will need to confirm that both the server and the client have enabled `SSL/TLS`.