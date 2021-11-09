# 使用示例

<!-- TOC -->

- [使用示例](#使用示例)
    - [联邦学习启动接口flJobRun()](#联邦学习启动接口fljobrun)
    - [多条数据输入推理接口modelInference()](#多条数据输入推理接口modelinference)
    - [获取云侧最新模型接口getModel ()](#获取云侧最新模型接口getmodel-)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/federated/api/source_zh_cn/interface_description_federated_client.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

注意，在使用以下接口前，可先参照文档[端侧部署](https://www.mindspore.cn/federated/docs/zh-CN/r1.5/deploy_federated_client.html)进行相关环境的部署。

## 联邦学习启动接口flJobRun()

调用flJobRun()接口前，需先实例化参数类FLParameter，进行相关参数设置， 相关参数如下：

| 参数名称       | 参数类型 | 是否必须 | 描述信息                                                     | 备注                                                         |
| -------------- | -------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| trainDataset   | String   | Y        | 训练数据集路径                                               | 情感分类任务是训练数据txt文件格式；图片分类任务是训练data.bin文件与label.bin文件用逗号拼接 |
| vocabFile      | String   | Y        | 数据预处理的词典文件路径                                     | 情感分类任务必须设置；图片分类任务不需要设置该参数，默认为null |
| idsFile        | String   | Y        | 词典的映射id文件路径                                         | 情感分类任务必须设置；图片分类任务不需要设置该参数，默认为null |
| testDataset    | String   | N        | 测试数据集路径                                               | 1. 图片分类任务不需要设置该参数，默认为null；情感分类任务不设置该参数代表训练过程中不进行验证<br />2.情感分类任务是测试数据txt文件格式；图片分类任务是测试data.bin文件与label.bin文件用逗号拼接 |
| flName         | String   | Y        | 联邦学习使用的模型名称                                       | 情感分类任务需设置为”albert“; lenet场景需设置为”lenet“       |
| trainModelPath | String   | Y        | 联邦学习使用的训练模型路径，为.ms文件的绝对路径              |                                                              |
| inferModelPath | String   | Y        | 联邦学习使用的推理模型路径，为.ms文件的绝对路径              | 有监督情感分类任务与图片分类任务均需设置为与trainModelPath相同 |
| useSSL         | boolean  | N        | 端云通信是否进行ssl证书认证，ssl证书认证只在https通信中使用  | 设置为false, 不进行ssl证书认证；设置为true，进行ssl证书认证；默认为false |
| certPath       | String   | N        | 端侧https通信所使用的SSL根证书路径                           | 当端云进行https通信时，且useSSL设置为true时必须设置该参数    |
| domainName     | String   | Y        | 端云通信url                                                  | 目前，https和http通信均支持，对应格式分别为：https://......、http://......，当`useElb`设置为true时，格式必须为：https://127.0.0.0:6666 或者http://127.0.0.0:6666 ，其中`127.0.0.0`对应提供云侧服务的机器ip（即云侧参数`--scheduler_ip`），`6666`对应云侧参数`--fl_server_port` |
| ifUseElb       | boolean  | N        | 用于设置是否模拟弹性负载均衡，true代表客户端会将请求随机发给一定范围内的server地址， false客户端的请求会发给固定的server地址 | 默认值为false                                                |
| serverNum      | int      | N        | 用于设置模拟弹性负载均衡时可发送请求的server数量             | 当ifUseElb设置为true时，可设置为与云侧启动server端时的`server_num`参数保持一致，默认值为1 |

注意useSSL设置为true时只支持https通信，还需对以下参数进行设置：

```java
FLParameter flParameter = FLParameter.getInstance();
String certPath  =  "client.crt";             //  端云通信SSL根证书绝对路径
flParameter.setCertPath(certPath);
```

创建SyncFLJob对象，并通过SyncFLJob类的flJobRun()方法启动同步联邦学习任务。

示例代码如下：

1. 情感分类任务示例代码

   ```java
   // set parameters
   String trainDataset = "SyncFLClient/data/albert/client/0.txt";                        //绝对路径
   String vocal_file = "SyncFLClient/data/albert/vocab.txt";                           //绝对路径
   String idsFile = "SyncFLClient/data/albert/vocab_map_ids.txt";                      //绝对路径
   String testDataset = "SyncFLClient/data/albert/eval/eval.txt";    //绝对路径, 若不包含单独的测试集, 可使用训练数据作为测试集， 或不进行测试（不设置该参数）
   String flName = "albert";  
   String trainModelPath = "SyncFLClient/ms/albert/albert_train.mindir.ms";                      //绝对路径
   String inferModelPath = "SyncFLClient/ms/albert/albert_train.mindir.ms";                      //绝对路径
   boolean useSSL = false;
   String domainName = "http://10.113.216.106:6668";

   FLParameter flParameter = FLParameter.getInstance();
   flParameter.setTrainDataset(trainDataset);
   flParameter.setVocabFile(vocabFile);
   flParameter.setIdsFile(idsFile);
   flParameter.setTestDataset(testDataset);
   flParameter.setFlName(flName);
   flParameter.setTrainModelPath(trainModelPath);
   flParameter.setInferModelPath(inferModelPath);
   flParameter.setUseSSL(useSSL);
   flParameter.setDomainName(domainName);

   // start FLJob
   SyncFLJob syncFLJob = new SyncFLJob();
   syncFLJob.flJobRun();
   ```

2. Lenet图片分类任务示例代码

   ```java
   // set parameters
   String trainDataset = "SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_9_train_data.bin,SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_9_train_label.bin";                        //绝对路径
   String testDataset = "SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_data.bin,SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_label.bin";    //绝对路径, 若不包含单独的测试集, 可使用训练数据作为测试集， 或不进行测试（不设置该参数）
   String flName = "lenet";
   String trainModelPath = "SyncFLClient/lenet_train.mindir0.ms";                      //绝对路径
   String inferModelPath = "SyncFLClient/lenet_train.mindir0.ms";                      //绝对路径
   boolean useSSL = false;
   String domainName = "http://10.113.216.106:6668";

   FLParameter flParameter = FLParameter.getInstance();
   flParameter.setTrainDataset(trainDataset);
   flParameter.setTestDataset(testDataset);
   flParameter.setFlName(flName);
   flParameter.setTrainModelPath(trainModelPath);
   flParameter.setInferModelPath(inferModelPath);
   flParameter.setUseSSL(useSSL);
   flParameter.setDomainName(domainName);

   // start FLJob
   SyncFLJob syncFLJob = new SyncFLJob();
   syncFLJob.flJobRun();
   ```

## 多条数据输入推理接口modelInference()

调用modelInference()接口前，需先实例化参数类FLParameter，进行相关参数设置， 相关参数如下：

| 参数名称  | 参数类型 | 是否必须 | 描述信息                                  | 适应API版本                                                  |
| --------- | -------- | -------- | ----------------------------------------- | ------------------------------------------------------------ |
| flName    | String   | Y        | 联邦学习使用的模型名称                    | 情感分类任务需设置为”albert“; 图片分类任务需设置为”lenet“    |
| dataPath  | String   | Y        | 数据集路径                                | 情感分类任务为txt文档格式，且不能带标签; 图片分类任务为bin文件格式 |
| vocabFile | String   | Y        | 数据预处理的词典文件路径                  | 情感分类任务必须设置；图片分类任务不需要设置                 |
| idsFile   | String   | Y        | 词典的映射id文件路径                      | 情感分类任务必须设置；图片分类任务不需要设置                 |
| modelPath | String   | Y        | 联邦学习推理模型路径，为.ms文件的绝对路径 | 有监督情感分类任务与图片分类任务均需设置为联邦学习训练任务中使用的trainModelPath |

创建SyncFLJob对象，并通过SyncFLJob类的modelInference()方法启动端侧推理任务，返回推理的标签数组。

示例代码如下：

1. 情感分类任务示例代码

   ```java
   // set parameters
   String flName = "albert";
   String dataPath = "SyncFLClient/data/albert/eval/eval.txt";                            //绝对路径
   String vocal_file = "SyncFLClient/data/albert/vocab.txt";                           //绝对路径
   String idsFile = "SyncFLClient/data/albert/vocab_map_ids.txt";                   //绝对路径
   String modelPath = "SyncFLClient/ms/albert/albert_train.mindir.ms";                                //绝对路径
   FLParameter flParameter = FLParameter.getInstance();
   flParameter.setFlName(flName);
   flParameter.setTestDataset(dataPath);
   flParameter.setVocabFile(vocabFile);
   flParameter.setIdsFile(idsFile);
   flParameter.setInferModelPath(modelPath);

   // inference
   SyncFLJob syncFLJob = new SyncFLJob();
   int[] labels = syncFLJob.modelInference();
   ```

2. Lenet图片分类示例代码

   ```java
   // set parameters
   String flName = "lenet";
   String dataPath = "/SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_data.bin,/SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_label.bin";     //绝对路径
   String modelPath = "SyncFLClient/lenet_train.mindir0.ms";           //绝对路径
   FLParameter flParameter = FLParameter.getInstance();
   flParameter.setFlName(flName);
   flParameter.setTestDataset(dataPath);
   flParameter.setInferModelPath(modelPath);

   // inference
   SyncFLJob syncFLJob = new SyncFLJob();
   int[] labels = syncFLJob.modelInference();
   ```

## 获取云侧最新模型接口getModel ()

调用getModel()接口前，需先实例化参数类FLParameter，进行相关参数设置， 相关参数如下：

| 参数名称       | 参数类型 | 是否必须 | 描述信息                                                     | 备注                                                         |
| -------------- | -------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| flName         | String   | Y        | 联邦学习使用的模型名称                                       | 情感分类任务需设置为”albert“; lenet场景需设置为”lenet“       |
| trainModelPath | String   | Y        | 联邦学习使用的训练模型路径，为.ms文件的绝对路径              |                                                              |
| inferModelPath | String   | Y        | 联邦学习使用的推理模型路径，为.ms文件的绝对路径              | 有监督情感分类任务与图片分类任务均需设置为与trainModelPath相同 |
| useSSL         | boolean  | N        | 端云通信是否进行ssl证书认证，ssl证书认证只在https通信中使用  | 设置为false, 不进行ssl证书认证；设置为true，进行ssl证书认证；默认为false |
| certPath       | String   | N        | 端侧https通信所使用的SSL根证书路径                           | 当端云进行https通信时，且useSSL设置为true时必须设置该参数    |
| domainName     | String   | Y        | 端云通信url                                                  | 目前，https和http通信均支持，对应格式分别为：https://......、http://......，当`useElb`设置为true时，格式必须为：https://127.0.0.0:6666 或者http://127.0.0.0:6666 ，其中`127.0.0.0`对应提供云侧服务的机器ip（即云侧参数`--scheduler_ip`），`6666`对应云侧参数`--fl_server_port` |
| ifUseElb       | boolean  | N        | 用于设置是否模拟弹性负载均衡，true代表客户端会将请求随机发给一定范围内的server地址， false客户端的请求会发给固定的server地址 | 默认值为false                                                |
| serverNum      | int      | N        | 用于设置模拟弹性负载均衡时可发送请求的server数量             | 当ifUseElb设置为true时，可设置为与云侧启动server端时的`server_num`参数保持一致，默认值为1 |

注意useSSL设置为true时只支持https通信，还需对以下参数进行设置：

```java
FLParameter flParameter = FLParameter.getInstance();
String certPath  =  "client.crt";             //  端云通信SSL根证书绝对路径
flParameter.setCertPath(certPath);
```

创建SyncFLJob对象，并通过SyncFLJob类的getModel()方法启动异步推理任务，返回getModel请求状态码。

示例代码如下：

1. 情感分类任务版本

   ```java
   // set parameters
   String flName = "albert";     // 情感分类任务场景需设置为"albert", lenet图片分类任务场景需设置为"lenet"
   String trainModelPath = "SyncFLClient/ms/albert/albert_train.mindir.ms";                      //绝对路径
   String inferModelPath = "SyncFLClient/ms/albert/albert_train.mindir.ms";                      //绝对路径
   boolean useSSL = false;
   String domainName = "http://10.113.216.106:6668";
   boolean ifUseElb = false;
   int serverNum = 1;

   FLParameter flParameter = FLParameter.getInstance();
   flParameter.setFlName(flName);
   flParameter.setTrainModelPath(trainModelPath);
   flParameter.setInferModelPath(inferModelPath);
   flParameter.setUseSSL(useSSL);
   flParameter.setDomainName(domainName);
   flParameter.setUseElb(useElb);
   flParameter.setServerNum(serverNum);

   // getModel
   SyncFLJob syncFLJob = new SyncFLJob();
   syncFLJob.getModel();
   ```

2. Lenet图片分类任务版本

   ```java
   // set parameters
   String flName = "lenet";     // 情感分类任务场景需设置为"albert", lenet场景需设置为"lenet"
   String trainModelPath = "SyncFLClient/lenet_train.mindir0.ms";                      //绝对路径
   String inferModelPath = "SyncFLClient/lenet_train.mindir0.ms";                      //绝对路径
   boolean useSSL = false;
   String domainName = "http://10.113.216.106:6668";
   boolean useElb = false;
   int serverNum = 1;

   FLParameter flParameter = FLParameter.getInstance();
   flParameter.setFlName(flName);
   flParameter.setTrainModelPath(trainModelPath);
   flParameter.setInferModelPath(inferModelPath);
   flParameter.setUseSSL(useSSL);
   flParameter.setDomainName(domainName);
   flParameter.setUseElb(useElb);
   flParameter.setServerNum(serverNum);

   // getModel
   SyncFLJob syncFLJob = new SyncFLJob();
   syncFLJob.getModel();
   ```