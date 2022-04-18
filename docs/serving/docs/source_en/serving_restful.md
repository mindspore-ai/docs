# RESTful-based MindSpore Serving Access

<a href="https://gitee.com/mindspore/docs/blob/master/docs/serving/docs/source_en/serving_restful.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore Serving supports both `gPRC` and `RESTful` request modes. The following describes the `RESTful` request.

`RESTful` is an API designed and developed based on `HTTP`. It manages and accesses resources through `URI` and features high scalability and clear structure. The lightweight `RESTful` can directly transmit data through `HTTP`, and has become the most popular `Web` service access mode. Users can directly interact with services in `RESTful` mode.

For details about how to deploy `Serving`, see [MindSpore Serving-based Inference Service Deployment](https://www.mindspore.cn/serving/docs/en/master/serving_example.html).

We can use the `mindspore_serving.server.start_restful_server` API to start the `RESTful` service.

## Request Method

Currently, only RESTful request of the `POST` type is supported. The request format is as follows:

```text
POST http://${HOST}:${PORT}/model/${MODLE_NAME}[/version/${VERSION}]:${METHOD_NAME}
```

In the preceding information:

- `${HOST}`: specifies the IP address to be accessed.
- `${PORT}`: specifies the port number to be accessed.
- `${MODLE_NAME}`: specifies the name of a model in the request.
- `${VERSION}`: specifies the version number. The version number is optional. If it is not specified, the latest model version is used by default.
- `${METHOD_NAME}`: specifies the method name of the request model.

If the `curl` tool is used, the RESTful request method is as follows:

```text
curl -X POST -d '${REQ_JSON_MESSAGE}' http://${HOST}:${PORT}/model/${MODLE_NAME}[/version/${VERSION}]:${METHOD_NAME}
```

For example, request for the `predict` method of the `LeNet` model to perform digital image inference:

```text
curl -X POST -d '{"instances":{"image":{"b64":"babe64-encoded-string"}}}' http://127.0.0.1:1500/model/lenet/version/1:predict
```

In the preceding information, `babe64-encoded-string` indicates the character string generated after the digital image is encoded using `base64`. The character string is long and is not listed explicitly.

## Request Format

RESTful supports the `Json` request format. `key` is fixed at `instances`, and `value` indicates multiple instances.

Each instance is represented by a `Json` object in `key-value` format. In the preceding information:

- `key`: specifies the input name, which must be the same as the input parameter name of the method provided by the request model. If they are different, the request fails.

- `value`: a specific value. Currently supported `value` types:

    - Scalar: `str`, `bytes`, `int`, `float` and `bool`

      `bytes` is supported after `base64` encoding.

    - Tensor: a one-level or multi-level array consisting of `int`, `float`, and `bool`

      A tensor uses the array format to indicate data and dimension information.

The `int` type supported in `Json` is `int32`, indicating the range, and the supported `float` type is `float32`, indicating the range.

Request format:

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

Example:

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

In the preceding information, `iVBOR...ggg===` is the omitted character string of the image number `0` after `base64` encoding. `iVBOR...QmCC` is the omitted character string of the image number `1` after `base64` encoding. The character strings encoded in different images may be different. The preceding description is for reference only.

### Base64 Data Encoding

The `bytes` type needs to be encoded using `base64`. `base64` can indicate the `bytes` type as well as other scalar and tensor data. In this case, the binary data of scalar and tensor is encoded using `base64`, the data type is specified using `type`, and the dimension information is specified using `shape`.

- `type`: This parameter is optional. If it is not specified, the default value is `bytes`.

  The value can be `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float16`(or `fp16`), `float32`(or `fp32`), `float64`(or `fp64`), `bool`, `str`, or `bytes`.

- `shape`: This parameter is optional. If it is not specified, the default value is `[1]`.

Example:

If the `base64` encoding is used to indicate a tensor of `int16` type, with `shape` 3*2 and the value `[[1,1],[2,3],[3,4]]`, the expression is as follows:

```json
{
    "instances":[
        {
            "box":{"b64":"AQACAAIAAwADAAQA", "type":"int16", "shape":[3,2]}
        }
    ]
}
```

`AQACAAIAAwADAAQA` is a character string obtained after the binary data format of `[[1,1],[2,3],[3,4]]` is encoded using `base64`.

**The supported types in request are as follows:**

| Supported Type | Example | Remarks |
| ------ | -------- | ---------------- |
| `int` | 1, [1, 2, 3, 4] | The default value is `int32`, indicating the range. |
| `float` | 1.0, [[1.2, 2.3], [3.0, 4.5]] | The default value is `float32`, indicating the range. |
| `bool` | true, false, [[true], [false]] | `bool` type |
| `string` | "hello" or <br/> {"b64":"aGVsbG8=", "type":"str"} | Direct representation or representation specified by `type`. |
| `bytes` | {"b64":"AQACAAIAAwADAAQA"} or <br>{"b64":"AQACAAIAAwADAAQA", "type":"bytes"} | If `type` is not specified, the default value `bytes` is used. |
| `int8`,`int16`,`int32`,`int64`,<br/>`uint8`,`uint16`,`uint32`,`uint64`,<br/>`float16`,`float32`,`float64`,`bool` | {"b64":"AQACAAIAAwADAAQA", "type":"int16", "shape":[3,2]} | The base64 encoding is used to indicate the data specified by `type`. |

## Response Format

The response format is the same as the request format. The information in the `Json` format is returned. The response format is as follows:

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

1. If all instances in a request are successfully processed, the response format is as follows:

   Example: `LeNet` requests to recognize numbers `0` and `1`.

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

2. If certain instances are faulty, the response format is as follows:

   Example: `LeNet` requests to recognize the digit `0` and an incorrect digit image.

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

3. If all instances in a request fail, the response format is as follows:

   Example: `LeNet` requests to recognize two incorrect digital images.

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

4. If a system error or other parsing error occurs, the return value is in the following format:

   For example, the value of `LeNet` is an invalid JSON character string.

   ```json
   {
       "error_msg":"Parse request failed"
   }
   ```

**The response data is represented as follows:**

   |  Serving Output Type | RESTful json Data Type   | Description  |  Example  |
   |  ----  | ----  |  ---- | ---- |
   | `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64` | json integer | All types of integer data are represented as JSON integer | 1，[1,2,3,4]  |
   | `float16`, `float32`, `float64` | json float | All types of float data are represented as JSON float | 1.0，[[1.2, 2.3], [3.0, 4.5]]  |
   | `bool` | json bool | Bool data is represented as json bool | true，false，[[true],[false]]  |
   | `string` | json str | String data is represented as json string | "news_car"  |
   | `bytes` | base64 object | Bytes data is represented as a base64 object | {"b64":"AQACAAIAAwADAAQA"}  |

## Accessing SSL/TLS enabled Serving RESTful service

MindSpore Serving supports `SSL/TLS` enabled `RESTful` service. Here's an example of starting and accessing `RESTful` service with one-way authentication.

Setting `verify_client` to `False` indicates one-way authentication, in order to enable `SSL/TLS`, pass  `mindspore_serving.server.SSLConfig` object to`ssl_config`. You can refer to  [Accessing SSL/TLS enabled Serving service](https://www.mindspore.cn/serving/docs/en/master/serving_grpc.html#accessing-ssltls-enabled-serving-service) for other details.

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

We can use `curl` command line or `requests` library accessing `SSL/TLS` enabled `RESTful` service. If you use `curl`, you could try the following command:

```text
curl -X POST -d '${REQ_JSON_MESSAGE}' --cacert '${PATH_TO_CA_CERT_FILE}' https://${HOST}:${PORT}/model/${MODLE_NAME}/version/${VERSION}]:${METHOD_NAME}
```

The example of accessing the `add_common` method of the `add` model is as follows:

```text
curl -X POST -d '{"instances":[{"x1":[[1.0, 2.0], [3.0, 4.0]], "x2":[[1.0, 2.0], [3.0, 4.0]]}]}' --cacert ca.crt https://localhost:5500/model/add/version/1:add_common
```

The protocol needs to be set to `https`, and set value of the option `--cacert` to the path of `ca.crt`.

By the way, we can set `--insecure` option representing not verifying the server's certificate due to using self-signed server's certificate in this case.
And here's an example:

```text
curl -X POST -d '{"instances":[{"x1":[[1.0, 2.0], [3.0, 4.0]], "x2":[[1.0, 2.0], [3.0, 4.0]]}]}' --insecure https://localhost:5500/model/add/version/1:add_common
```