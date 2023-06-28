# 基于MindSpore部署推理服务

`Linux` `Ascend` `环境准备` `企业` `高级`
<a href="https://gitee.com/mindspore/docs/blob/r1.0/tutorials/inference/source_zh_cn/serving.md" target="_blank"><img src="./_static/logo_source.png"></a>


## 概述

MindSpore Serving是一个轻量级、高性能的服务模块，旨在帮助MindSpore开发者在生产环境中高效部署在线推理服务。当用户使用MindSpore完成模型训练后，导出MindSpore模型，即可使用MindSpore Serving创建该模型的推理服务。当前Serving仅支持Ascend 910。

## 启动Serving服务
通过pip安装MindSpore后，Serving可执行程序位于`/{your python path}/lib/python3.7/site-packages/mindspore/ms_serving`。
启动Serving服务命令如下
```bash 
ms_serving [--help] [--model_path=<MODEL_PATH>] [--model_name=<MODEL_NAME>] [--port=<PORT1>] 
                            [--rest_api_port=<PORT2>] [--device_id=<DEVICE_ID>]
```
参数含义如下

|参数名|属性|功能描述|参数类型|默认值|取值范围|
|---|---|---|---|---|---|
|`--help`|可选|显示启动命令的帮助信息。|-|-|-|
|`--model_path=<MODEL_PATH>`|必选|指定待加载模型的存放路径。|String|空|-|
|`--model_name=<MODEL_NAME>`|必选|指定待加载模型的文件名。|String|空|-|
|`--port=<PORT1>`|可选|指定Serving对外的gRPC端口号。|Integer|5500|1~65535|
|`--rest_api_port=<PORT2>`|可选|指定Serving对外的REST API端口号。|Integer|5501|1~65535|
|`--device_id=<DEVICE_ID>`|可选|指定使用的设备号|Integer|0|0~7|

 > 执行启动命令前，需将`/{your python path}/lib:/{your python path}/lib/python3.7/site-packages/mindspore/lib`对应的路径加入到环境变量LD_LIBRARY_PATH中 。
 > port与rest_api_port不可相同。

## 应用示例
下面以一个简单的网络为例，演示MindSpore Serving如何使用。

### 导出模型
 > 导出模型之前，需要配置MindSpore[基础环境](https://www.mindspore.cn/install)。

使用[add_model.py](https://gitee.com/mindspore/mindspore/blob/r1.0/serving/example/export_model/add_model.py)，构造一个只有Add算子的网络，并导出MindSpore推理部署模型。

```python 
python add_model.py
```

执行脚本，生成`tensor_add.mindir`文件，该模型的输入为两个shape为[2,2]的二维Tensor，输出结果是两个输入Tensor之和。

### 启动Serving推理服务
```bash 
ms_serving --model_path={model directory} --model_name=tensor_add.mindir
```

当服务端打印日志`MS Serving gRPC start success, listening on 0.0.0.0:5500`时，表示Serving gRPC服务已加载推理模型完毕。
当服务端打印日志`MS Serving RESTful start, listening on 0.0.0.0:5501`时，表示Serving REST服务已加载推理模型完毕。

### gRPC客户端示例
#### <span name="python客户端示例">Python客户端示例</span>
 > 执行客户端前，需将`/{your python path}/lib/python3.7/site-packages/mindspore`对应的路径添加到环境变量PYTHONPATH中。

获取[ms_client.py](https://gitee.com/mindspore/mindspore/blob/r1.0/serving/example/python_client/ms_client.py)，启动Python客户端。
```bash
python ms_client.py
```

显示如下返回值说明Serving服务已正确执行Add网络的推理。
```bash
ms client received:
[[2. 2.]
 [2. 2.]]
```

#### <span name="cpp客户端示例">C++客户端示例</span>
1. 获取客户端示例执行程序

    首先需要下载[MindSpore源码](https://gitee.com/mindspore/mindspore)。有两种方式编译并获取客户端示例程序：
    + 从源码编译MindSpore时候，将会编译产生Serving C++客户端示例程序，可在`build/mindspore/serving/example/cpp_client`目录下找到`ms_client`可执行程序。
    + 独立编译：

        需要先预装[gRPC](https://gRPC.io)。

        然后，在MindSpore源码路径中执行如下命令，编译一个客户端示例程序。
        ```bash
        cd mindspore/serving/example/cpp_client
        mkdir build && cd build
        cmake -D GRPC_PATH={grpc_install_dir} ..
        make
        ```
        其中`{grpc_install_dir}`为gRPC安装时的路径，请替换为实际gRPC安装路径。

2. 启动gRPC客户端

    执行ms_client，向Serving服务发送推理请求：
    ```bash
    ./ms_client --target=localhost:5500
    ```
    显示如下返回值说明Serving服务已正确执行Add网络的推理。
    ```
    Compute [[1, 2], [3, 4]] + [[1, 2], [3, 4]]
    Add result is 2 4 6 8
    client received: RPC OK
    ```

客户端代码主要包含以下几个部分：

1. 基于MSService::Stub实现Client，并创建Client实例。
    ```
    class MSClient {
     public:
      explicit MSClient(std::shared_ptr<Channel> channel) :  stub_(MSService::NewStub(channel)) {}
     private:
      std::unique_ptr<MSService::Stub> stub_;
    };
    
    MSClient client(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
    
    ```
2. 根据网络的实际输入构造请求的入参Request、出参Reply和gRPC的客户端Context。
    ```
    PredictRequest request;
    PredictReply reply;
    ClientContext context;
    
    //construct tensor
    Tensor data;
    
    //set shape
    TensorShape shape;
    shape.add_dims(2);
    shape.add_dims(2);
    *data.mutable_tensor_shape() = shape;
    
    //set type
    data.set_tensor_type(ms_serving::MS_FLOAT32);
    std::vector<float> input_data{1, 2, 3, 4};
    
    //set datas
    data.set_data(input_data.data(), input_data.size());
    
    //add tensor to request
    *request.add_data() = data;
    *request.add_data() = data;
    ```
3. 调用gRPC接口和已经启动的Serving服务通信，并取回返回值。
    ```Status status = stub_->Predict(&context, request, &reply);```

完整代码参考[ms_client](https://gitee.com/mindspore/mindspore/blob/r1.0/serving/example/cpp_client/ms_client.cc)。 

### REST API客户端示例
1. `data`形式发送数据：

    data字段：将网络模型每个输入数据展平成一维数据，假设网络模型有n个输入，最后data数据结构为1*n的二维list。
    
    如本例中，将模型输入数据`[[1.0, 2.0], [3.0, 4.0]]`和`[[1.0, 2.0], [3.0, 4.0]]`展平后组合成data形式的数据`[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]`
    
    ```
    curl -X POST -d '{"data": [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]}' http://127.0.0.1:5501
    ```
   
    显示如下返回值，说明Serving服务已正确执行Add网络的推理，输出数据结构同输入类似：
    ```
    {"data":[[2.0,4.0,6.0,8.0]]}
    ```

2. `tensor`形式发送数据：

    tensor字段：由网络模型每个输入组合而成，保持输入的原始shape。
    
    如本例中，将模型输入数据`[[1.0, 2.0], [3.0, 4.0]]`和`[[1.0, 2.0], [3.0, 4.0]]`组合成tensor形式的数据`[[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]`
    ```
    curl -X POST -d '{"tensor": [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]]}' http://127.0.0.1:5501
    ```
    显示如下返回值，说明Serving服务已正确执行Add网络的推理，输出数据结构同输入类似：
    ```
    {"tensor":[[2.0,4.0], [6.0,8.0]]}
    ```
 > REST API当前只支持int32和fp32数据输入。

