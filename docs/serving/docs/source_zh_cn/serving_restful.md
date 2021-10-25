# 基于RESTful接口访问MindSpore Serving服务

<!-- TOC -->

- [基于RESTful接口访问MindSpore Serving服务](#基于restful接口访问mindspore-serving服务)
    - [概述](#概述)
    - [请求方式](#请求方式)
    - [请求输入格式](#请求输入格式)
        - [base64数据编码](#base64数据编码)
    - [请求应答格式](#请求应答格式)
    - [访问开启SSL/TLS的RESTful服务](#访问开启ssltls的restful服务)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/serving/docs/source_zh_cn/serving_restful.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

MindSpore Serving支持`gPRC`和`RESTful`两种请求方式。本章节介绍`RESTful`类型请求。

`RESTful`是一种基于`HTTP`协议的网络应用程序的设计风格和开发方式，通过`URI`实现对资源的管理及访问，具有扩展性强、结构清晰的特点。基于其轻量级以及通过`HTTP`直接传输数据的特性，`RESTful`已经成为最常见的`Web`服务访问方式。用户通过`RESTful`方式，能够简单直接的与服务进行交互。

部署`Serving`参考[快速入门](https://www.mindspore.cn/serving/docs/zh-CN/master/serving_example.html) 章节。

我们可以通过`mindspore_serving.server.start_restful_server`接口启动`RESTful`服务。

## 请求方式

当前仅支持`POST`类型的RESTful请求，请求格式如下：

```text
POST http://${HOST}:${PORT}/model/${MODLE_NAME}[/version/${VERSION}]:${METHOD_NAME}
```

其中：

- `${HOST}`：指定访问的IP地址；
- `${PORT}`：指定访问的端口号；
- `${MODLE_NAME}`：请求的模型名称；
- `${VERSION}`：表示版本号。版本号是可选的，若未指定具体版本号，则默认使用模型的最新版本。
- `${METHOD_NAME}`：表示请求模型的具体方法名称。

如果使用`curl`工具，RESTful请求方式如下：

```text
curl -X POST -d '${REQ_JSON_MESSAGE}' http://${HOST}:${PORT}/model/${MODLE_NAME}[/version/${VERSION}]:${METHOD_NAME}
```

例子：请求`LeNet`模型的`predict`方法进行数字图片的推理，请求如下：

```text
curl -X POST -d '{"instances":{"image":{"b64":"babe64-encoded-string"}}}' http://127.0.0.1:1500/model/lenet/version/1:predict
```

其中：`babe64-encoded-string`表示数字图片经过`base64`编码之后的字符串。由于字符串比较长，不显式列出。

## 请求输入格式

RESTful支持`Json`请求格式，`key`固定为`instances`，`value`表示多个实例。

每个实例通过`key-value`格式的`Json`对象来表示。其中：

- `key`：表示输入名称，需要与请求模型提供的方法的输入参数名称一致，若不一致，则请求失败。

- `value`：表示具体的值。当前支持的`value`类型：

    - 标量：`str`、`bytes`、`int`、`float`、`bool`。

      `bytes`：通过`base64`编码方式支持。

    - 张量：`int`、`float`、`bool`组成的一级或多级数组。

      张量通过数组格式表示数据和维度信息。

`Json`中支持的`int`类型：是`int32`表示的范围，`float`类型：是`float32`表示的范围。

请求格式：

```text
{
    "instances":[
        {
            "input_name1":<value>|<list>|<object>,
            "input_name2":<value>|<list>|<object>,
            ...
        },
        {
            "input_name1":<value>|<list>|<object>,
            "input_name2":<value>|<list>|<object>,
            ...
        }
        ...
    ]
}
```

例子：

```text
{
    "instances":[
        {
            "tag":"one",
            "box":[[1,1],[2,3],[3,4]],
            "image":{"b64":"iVBOR...ggg==="}
        },
        {
            "tag":"two",
            "box":[[2,2],[5,5],[6,6]],
            "image":{"b64":"iVBOR...QmCC", "type":"bytes"}
        }
    ]
}
```

其中：`iVBOR...ggg===`是图片数字`0`经过`base64`编码之后的省略字符串。`iVBOR...QmCC`是图片数字`1`经过`base64`编码之后的省略字符串。不同图片编码出来的字符串可能不同，上述是示意说明。

### base64数据编码

`bytes`类型需要通过`base64`编码进行表示。`base64`除了可以表示`bytes`类型，也可以表示其他标量和张量数据，此时将标量和张量的二进制数据通过`base64`进行编码，并额外通过`type`指定数据类型，通过`shape`指定维度信息：

- `type`：可选，如果不指定，默认为`bytes`。

  支持`int8`、`int16`、`int32`、`int64`、`uint8`、`uint16`、`uint32`、`uint64`、`float16`(或`fp16`)、`float32`(或`fp32`)、`float64`(或`fp64`)、`bool`、`str`、`bytes`。

- `shape`：可选，如果不指定，默认为`[1]`。

例子：

如果要用`base64`编码表示：`int16`的数据类型，`shape`为3*2，值是`[[1,1],[2,3],[3,4]]`的张量，则表示如下：

```json
{
    "instances":[
        {
            "box":{"b64":"AQACAAIAAwADAAQA", "type":"int16", "shape":[3,2]}
        }
    ]
}
```

其中`AQACAAIAAwADAAQA`：是`[[1,1],[2,3],[3,4]]`的二进制数据格式经过`base64`编码后的字符串。

**请求支持的类型总结如下:**

| 支持的类型 |    例子   |     备注         |
| ------ | -------- | ---------------- |
|  `int`  | 1，[1,2,3,4]                   | 默认`int32`表示范围   |
| `float` | 1.0，[[1.2, 2.3], [3.0, 4.5]]  | 默认`float32`表示范围 |
|  `bool` | true，false，[[true],[false]]  | `bool`类型           |
| `string` | "hello"或者<br/>  {"b64":"aGVsbG8=", "type":"str"} | 直接表示或者指定`type`方式表示     |
| `bytes` | {"b64":"AQACAAIAAwADAAQA"} 或者 <br>{"b64":"AQACAAIAAwADAAQA", "type":"bytes"} | 如果不填`type`，默认为`bytes` |
| `int8`,`int16`,`int32`,`int64`,<br/>`uint8`,`uint16`,`uint32`,`uint64`,<br/>`float16`,`float32`,`float64`,`bool` | {"b64":"AQACAAIAAwADAAQA", "type":"int16", "shape":[3,2]}  | 利用base64编码，表示指定type的数据 |

## 请求应答格式

应答格式与请求格式保持一致。返回`Json`格式信息。应答格式如下：

```text
{
    "instances":[
        {
            "output_name1":<value>|<list>|<object>,
            "output_name2":<value>|<list>|<object>,
            ...
        },
        {
            "output_name1":<value>|<list>|<object>,
            "output_name2":<value>|<list>|<object>,
            ...
        }
        ...
    ]
}
```

1. 多实例请求后，如果多实例全部成功处理，则响应格式如下：

   例子：`LeNet`请求识别数字`0`和数字`1`。

   ```json
   {
       "instances":[
           {
               "result":0
           },
           {
               "result":1
           }
       ]
   }
   ```

2. 如果部分实例出错，则响应格式如下：

   例子：`lenet`请求识别数字`0`和一个错误数字图片。

   ```json
   {
       "instances":[
           {
               "result":0
           },
           {
               "error_msg":"Preprocess Failed"
           }
       ]
   }
   ```

3. 如果请求全部失败，则响应格式如下：

   例子：`lenet`请求识别两张错误数字图片为例。

   ```json
   {
       "instances":[
           {
               "error_msg":"Preprocess Failed"
           },
           {
               "error_msg":"Time out"
           }
       ]
   }
   ```

4. 出现系统性或者其他解析等错误，则返回格式：

   例子：`lenet`传入非法`Json`字符串。

   ```json
   {
       "error_msg":"Parse request failed"
   }
   ```

**应答数据表示如下:**

|  Serving输出类型 | RESTful json中数据类型   | 说明  |  举例  |
|  ----  | ----  |  ---- | ---- |
| `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64` | json integer | 整型格式的数据表示为json整型 | 1，[1,2,3,4]  |
| `float16`, `float32`, `float64` | json float | 浮点格式的数据表示为json浮点数 | 1.0，[[1.2, 2.3], [3.0, 4.5]]  |
| `bool` | json bool | bool类型数据表示为json bool | true，false，[[true],[false]]  |
| `string` | json str | 字符串格式输出表示为json str | "news_car"  |
| `bytes` | base64 object | 二进制格式输出转为base64对象 | {"b64":"AQACAAIAAwADAAQA"}  |

## 访问开启SSL/TLS的RESTful服务

MindSpore Serving支持开启`SSL/TLS`的`RESTful`服务，下面以单向认证为例展示如何启动并访问开启`SSL/TLS`的`Restful`服务。

`verify_client`设置为`False`表示单向认证，开启`SSL/TLS`需要把`mindspore_serving.server.SSLConfig`对象传入`start_restful_server`的`ssl_config`参数。其他内容可以参考[访问开启SSL/TLS的Serving服务](https://www.mindspore.cn/serving/docs/zh-CN/master/serving_grpc.html#ssl-tlsserving)。

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

    server.start_restful_server(address="127.0.0.1:5500", ssl_config=ssl_config)


if __name__ == "__main__":
    start()
```

我们可以使用`curl`工具或`python`的`requests`库访问`Serving`的开启`SSL/TLS`的`RESTful`服务。如果使用`curl`工具访问，可以尝试使用下面的请求方式：

```text
curl -X POST -d '${REQ_JSON_MESSAGE}' --cacert '${PATH_TO_CA_CERT_FILE}' https://${HOST}:${PORT}/model/${MODLE_NAME}/version/${VERSION}]:${METHOD_NAME}
```

例子：请求`add`模型的`add_common`方法，具体如下：

```text
curl -X POST -d '{"instances":[{"x1":[[1.0, 2.0], [3.0, 4.0]], "x2":[[1.0, 2.0], [3.0, 4.0]]}]}' --cacert ca.crt https://localhost:5500/model/add/version/1:add_common
```

我们这里需要将协议设置为`https`，设置选项`--cacert`的值为CA证书文件`ca.crt`的路径。

另外由于示例中使用了自签名的证书，也可以设置选项`--insecure`表示忽略对服务器证书的验证，具体如下：

```text
curl -X POST -d '{"instances":[{"x1":[[1.0, 2.0], [3.0, 4.0]], "x2":[[1.0, 2.0], [3.0, 4.0]]}]}' --insecure https://localhost:5500/model/add/version/1:add_common
```