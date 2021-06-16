# 联邦学习接口介绍

## FL-Client

### (1)联邦学习启动接口flJobRun()

调用flJobRun()接口前，需先实例化参数类FLParameter，进行相关参数设置， 相关参数如下：

| 参数名称       | 参数类型 | 是否必须 | 描述信息                                                 | 备注                                                         |
| -------------- | -------- | -------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| trainDataset   | String   | Y        | 训练数据集路径                                           | 情感分类任务是训练数据txt文件格式；图片分类任务是训练data.bin文件与label.bin文件用逗号拼接 |
| vocabFile      | String   | Y        | 数据预处理的词典文件路径                                 | 情感分类任务必须设置；图片分类任务不需要设置该参数，默认为null |
| idsFile        | String   | Y        | 词典的映射id文件路径                                     | 情感分类任务必须设置；图片分类任务不需要设置该参数，默认为null |
| testDataset    | String   | N        | 测试数据集路径                                           | 1. 图片分类任务不需要设置该参数，默认为null；情感分类任务不设置该参数代表训练过程中不进行验证<br />2.情感分类任务是测试数据txt文件格式；图片分类任务是测试data.bin文件与label.bin文件用逗号拼接 |
| flName         | String   | Y        | 联邦学习使用的模型名称                                   | 情感分类任务需设置为”adbert“; lenet场景需设置为”lenet“       |
| trainModelPath | String   | Y        | 联邦学习使用的训练模型路径，为.ms文件的绝对路径          |                                                              |
| inferModelPath | String   | Y        | 联邦学习使用的推理模型路径，为.ms文件的绝对路径          | 情感分类任务必须设置；图片分类任务可设置为与trainModelPath相同 |
| flID           | String   | Y        | 用于唯一标识客户端的ID                                   |                                                              |
| ip             | String   | Y        | Server端所启动服务的ip地址，形如“http://10.113.216.106:” | 后期ip+port会改为域名                                        |
| port           | int      | Y        | Server端所启动服务的端口号                               | 后期ip+port会改为域名                                        |
| useSSL         | boolean  | N        | 端云通信是否进行ssl证书认证，默认不进行                  |                                                              |

创建SyncFLJob对象，并通过SyncFLJob类的flJobRun()方法启动同步联邦学习任务。

示例代码如下：

1. 情感分类任务示例代码

```java
// set parameters
String trainDataset = "SyncFLClient0604/data/adbert/client/0.txt";                        //绝对路径
String vocal_file = "SyncFLClient0604/data/adbert/vocab.txt";                           //绝对路径
String idsFile = "SyncFLClient0604/data/adbert/vocab_map_ids.txt";                      //绝对路径
String testDataset = "SyncFLClient0604/data/adbert/eval/eval.txt";    //绝对路径, 若不包含单独的测试集, 可使用训练数据作为测试集， 或不进行测试（不设置该参数）
String flName = "adbert";  
String trainModelPath = "SyncFLClient0604/ms/adbert/albert_ad_train.mindir.ms";                      //绝对路径
String inferModelPath = "SyncFLClient0604/ms/adbert/albert_ad_infer.mindir.ms";                      //绝对路径
String flID = UUID.randomUUID().toString();
String ip = "http://10.113.216.106:";
int port = 6668;
boolean useSSL = false;

FLParameter flParameter = FLParameter.getInstance();
flParameter.setTrainDataset(trainDataset);
flParameter.setVocabFile(vocabFile);
flParameter.setIdsFile(idsFile);
flParameter.setTestDataset(testDataset);
flParameter.setFlName(flName);
flParameter.setTrainModelPath(trainModelPath);
flParameter.setInferModelPath(inferModelPath);
flParameter.setFlID(flID);
flParameter.setIp(ip);
flParameter.setPort(port);
flParameter.setUseSSL(useSSL);

// start FLJob
SyncFLJob syncFLJob = new SyncFLJob();
syncFLJob.flJobRun();
```

2. Lenet图片分类任务示例代码

```java
// set parameters
String trainDataset = "SyncFLClient0604/data/3500_clients_bin/f0178_39/f0178_39_bn_9_train_data.bin,SyncFLClient0604/data/3500_clients_bin/f0178_39/f0178_39_bn_9_train_label.bin";                        //绝对路径
String testDataset = "SyncFLClient0604/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_data.bin,SyncFLClient0604/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_label.bin";    //绝对路径, 若不包含单独的测试集, 可使用训练数据作为测试集， 或不进行测试（不设置该参数）
String flName = "lenet";
String trainModelPath = "SyncFLClient0604/lenet_train.mindir0.ms";                      //绝对路径
String inferModelPath = "SyncFLClient0604/lenet_train.mindir0.ms";                      //绝对路径
String flID = UUID.randomUUID().toString();
String ip = "http://10.113.216.106:";
int port = 6668;
boolean useSSL = false;

FLParameter flParameter = FLParameter.getInstance();
flParameter.setTrainDataset(trainDataset);
flParameter.setTestDataset(testDataset);
flParameter.setFlName(flName);
flParameter.setTrainModelPath(trainModelPath);
flParameter.setInferModelPath(inferModelPath);
flParameter.setFlID(flID);
flParameter.setIp(ip);
flParameter.setPort(port);
flParameter.setUseSSL(useSSL);

// start FLJob
SyncFLJob syncFLJob = new SyncFLJob();
syncFLJob.flJobRun();
```

### (2)多条数据输入推理接口modelInference()

#### 输入参数列表

| 参数名称  | 参数类型 | 是否必须 | 描述信息                                  | 适应API版本                                               |
| --------- | -------- | -------- | ----------------------------------------- | --------------------------------------------------------- |
| flName    | String   | Y        | 联邦学习使用的模型名称                    | 情感分类任务需设置为”adbert“; 图片分类任务需设置为”lenet“ |
| dataPath  | String   | Y        | 数据集路径                                | 情感分类任务为txt文档格式; 图片分类任务为bin文件格式      |
| vocabFile | String   | Y        | 数据预处理的词典文件路径                  | 情感分类任务必须设置；图片分类任务设置为null              |
| idsFile   | String   | Y        | 词典的映射id文件路径                      | 情感分类任务必须设置；图片分类任务设置为null              |
| modelPath | String   | Y        | 联邦学习推理模型路径，为.ms文件的绝对路径 |                                                           |

创建SyncFLJob对象，并通过SyncFLJob类的modelInference()方法启动端侧推理任务，返回推理的标签数组。

示例代码如下：

1. 情感分类任务示例代码

```java
// set parameters
String flName = "adbert";
String dataPath = "SyncFLClient0604/data/adbert/eval/eval.txt";                            //绝对路径
String vocal_file = "SyncFLClient0604/data/adbert/vocab.txt";                           //绝对路径
String idsFile = "SyncFLClient0604/data/adbert/vocab_map_ids.txt";                   //绝对路径
String modelPath = "SyncFLClient0604/ms/adbert/albert_ad_infer.mindir.ms";                                //绝对路径

// inference
SyncFLJob syncFLJob = new SyncFLJob();
int[] labels = syncFLJob.modelInference(flName, dataPath, vocal_file, idsFile, modelPath);
```

2. Lenet图片分类示例代码

```java
// set parameters
String flName = "lenet";
String dataPath = "/SyncFLClient0604/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_data.bin,/SyncFLClient0604/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_label.bin";     //绝对路径
String vocal_file = "null";                           //绝对路径
String idsFile = "null";                              //绝对路径
String modelPath = "SyncFLClient0604/lenet_train.mindir0.ms";           //绝对路径

// inference
SyncFLJob syncFLJob = new SyncFLJob();
int[] labels = syncFLJob.modelInference(flName, dataPath, vocal_file, idsFile, modelPath);
```

### (3)获取云侧最新模型接口getModel ()

#### 输入参数列表

| 参数名称       | 参数类型 | 是否必须 | 描述信息                                                     | 适应API版本                                                  |
| -------------- | -------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| useElb         | boolean  | Y        | 用于设置是否模拟弹性负载均衡，true代表客户端会将请求随机发给一定范围内的server地址， false客户端的请求会发给固定的server地址。 |                                                              |
| serverNum      | int      | Y        | 用于设置模拟弹性负载均衡时可发送请求的server数量，需与云侧启动server数量一致。 |                                                              |
| ip             | String   | Y        | Server端所启动服务的ip地址，形如“http://10.113.216.106:”     | 后期ip+port会改为域名                                        |
| port           | int      | Y        | Server端所启动服务的端口号                                   | 后期ip+port会改为域名                                        |
| flName         | String   | Y        | 联邦学习使用的模型名称                                       | 情感分类任务需设置为”adbert“;图片分类任务需设置为”lenet“     |
| trainModelPath | String   | Y        | 联邦学习使用的训练模型路径，为.ms文件的绝对路径              |                                                              |
| inferModelPath | String   | Y        | 联邦学习使用的推理模型路径，为.ms文件的绝对路径              | 情感分类任务必须设置；图片分类任务可设置为与trainModelPath相同 |
| useSSL         | boolean  | Y        | 端云通信是否进行ssl证书认证，默认不进行                      |                                                              |

创建SyncFLJob对象，并通过SyncFLJob类的getModel()方法启动异步推理任务，返回getModel请求状态码。

示例代码如下：

1. 情感分类任务版本

   ```java
   // set parameters
   boolean useElb = false;
   int serverNum = 1;
   String ip = "http://10.113.216.106:";
   int port = 6668;
   String flName = "adbert";     //情感分类任务场景需设置为”adbert“, lenet图片分类任务场景需设置为”lenet“
   String trainModelPath = "SyncFLClient0604/ms/adbert/albert_ad_train.mindir.ms";                      //绝对路径
   String inferModelPath = "SyncFLClient0604/ms/adbert/albert_ad_infer.mindir.ms";                      //绝对路径
   boolean useSSL = false;

   // getModel
   SyncFLJob syncFLJob = new SyncFLJob();
   syncFLJob.getModel(useElb, serverNum, ip, port, flName, trainModelPath, inferModelPath, useSSL);
   ```

2. Lenet图片分类任务版本

   ```java
   // set parameters
   boolean useElb = false;
   int serverNum = 1;
   String ip = "http://10.113.216.106:";
   int port = 6668;
   String flName = "lenet";     //端大脑场景需设置为”adbert“, lenet场景需设置为”lenet“
   String trainModelPath = "SyncFLClient0604/lenet_train.mindir0.ms";                      //绝对路径
   String inferModelPath = "SyncFLClient0604/lenet_train.mindir0.ms";                      //绝对路径
   boolean useSSL = false;

   // getModel
   SyncFLJob syncFLJob = new SyncFLJob();
   syncFLJob.getModel(useElb, serverNum, ip, port, flName, trainModelPath, inferModelPath, useSSL);
   ```
