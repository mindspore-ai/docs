# 使用示例

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/federated/docs/source_zh_cn/interface_description_federated_client.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

注意，在使用以下接口前，可先参照文档[端侧部署](https://www.mindspore.cn/federated/docs/zh-CN/r1.6/deploy_federated_client.html)进行相关环境的部署。

## 联邦学习启动接口flJobRun()

调用flJobRun()接口前，需先实例化参数类FLParameter，进行相关参数设置， 相关参数如下：

| 参数名称             | 参数类型                   | 是否必须 | 描述信息                                                     | 备注                                                         |
| -------------------- | -------------------------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| dataMap              | Map<RunType, List<String/>/> | Y        | 联邦学习数据集路径                                           | Map<RunType, List<String/>/>类型的数据集，map中key为RunType枚举类型，value为对应的数据集列表，key为RunType.TRAINMODE时代表对应的value为训练相关的数据集列表，key为RunType.EVALMODE时代表对应的value为验证相关的数据集列表， key为RunType.INFERMODE时代表对应的value为推理相关的数据集列表。 |
| flName               | String                     | Y        | 联邦学习使用的模型脚本包路径                                 | 我们提供了两个类型的模型脚本供大家参考（[有监督情感分类任务](https://gitee.com/mindspore/mindspore/tree/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert)、[LeNet图片分类任务](https://gitee.com/mindspore/mindspore/tree/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet)），对于有监督情感分类任务，该参数可设置为所提供的脚本文件[AlBertClient.java](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert/AlbertClient.java) 的包路径`com.mindspore.flclient.demo.albert.AlbertClient`；对于LeNet图片分类任务，该参数可设置为所提供的脚本文件[LenetClient.java](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet/LenetClient.java) 的包路径`com.mindspore.flclient.demo.lenet.LenetClient`。同时，用户可参考这两个类型的模型脚本，自定义模型脚本，然后将该参数设置为自定义的模型文件ModelClient.java（需继承于类[Client.java](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/java/java/fl_client/src/main/java/com/mindspore/flclient/model/Client.java)）的包路径即可。 |
| trainModelPath       | String                     | Y        | 联邦学习使用的训练模型路径，为.ms文件的绝对路径              | 建议将路径设置到训练App自身目录下，保护模型本身的数据访问安全性。 |
| inferModelPath       | String                     | Y        | 联邦学习使用的推理模型路径，为.ms文件的绝对路径              | 对于普通联邦学习模式（训练和推理使用同一个模型），该参数需设置为与trainModelPath相同；对于混合学习模式（训练和推理使用不同的模型，且云侧也包含训练过程），该参数设置为实际的推理模型路径。建议将路径设置到训练App自身目录下，保护模型本身的数据访问安全性。 |
| sslProtocol          | String                     | N        | 端云HTTPS通信所使用的TLS协议版本                             | 设置了白名单，目前只支持"TLSv1.3"或者"TLSv1.2"。非必须设置，默认值为"TLSv1.2"。只在HTTPS通信场景中使用。 |
| deployEnv            | String                     | Y        | 联邦学习的部署环境                                           | 设置了白名单，目前只支持"x86", "android"。                   |
| certPath             | String                     | N        | 端云https通信所使用的自签名根证书路径                        | 当部署环境为"x86"，且端云采用自签名证书进行https通信认证时，需要设置该参数，该证书需与生成云侧自签名证书所使用的CA根证书一致才能验证通过，此参数用于非Android场景。 |
| domainName           | String                     | Y        | 端云通信url                                                  | 目前，https和http通信均支持，对应格式分别为：https://......、http://......，当`useElb`设置为true时，格式必须为：https://127.0.0.0:6666 或者http://127.0.0.0:6666 ，其中`127.0.0.0`对应提供云侧服务的机器ip（即云侧参数`--scheduler_ip`），`6666`对应云侧参数`--fl_server_port`。 |
| ifUseElb             | boolean                    | N        | 用于多server场景设置是否将客户端的请求随机发送给一定范围内的不同server | 设置为true代表客户端会将请求随机发给一定范围内的server地址，false代表客户端的请求会发给固定的server地址，此参数用于非Android场景，默认值为false。 |
| serverNum            | int                        | N        | 客户端可选择连接的server数量                                 | 当ifUseElb设置为true时，可设置为与云侧启动server端时的`server_num`参数保持一致，用于随机选择不同的server发送信息，此参数用于非Android场景，默认值为1。 |
| ifPkiVerify          | boolean                    | N        | 端云身份认证开关                                             | 设置为true代表开启端云安全认证，设置为false代表不开启，默认值为false。身份认证需要HUKS提供证书，该参数只在Android环境中使用（目前只支持华为手机）。 |
| threadNum            | int                        | N        | 联邦学习训练和推理时使用的线程数                             | 默认值为1                                                    |
| cpuBindMode          | BindMode                   | N        | 联邦学习训练和推理时线程所需绑定的cpu内核                    | BindMode枚举类型，其中BindMode.NOT_BINDING_CORE代表不绑定内核，由系统自动分配，BindMode.BIND_LARGE_CORE代表绑定大核，BindMode.BIND_MIDDLE_CORE代表绑定中核。默认值为BindMode.NOT_BINDING_CORE。 |
| batchSize            | int                        | Y        | 联邦学习训练和推理时使用的单步训练样本数，即batch size       | 需与模型的输入数据的batch size保持一致。                     |
| iflJobResultCallback | IFLJobResultCallback       | N        | 联邦学习回调函数对象                                         | 用户可根据实际场景所需，实现工程中接口类[IFLJobResultCallback.java](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/java/java/fl_client/src/main/java/com/mindspore/flclient/IFLJobResultCallback.java)的具体方法后，作为回调函数对象设置到联邦学习任务中。我们提供了一个简单的实现用例[FLJobResultCallback.java](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/java/java/fl_client/src/main/java/com/mindspore/flclient/FLJobResultCallback.java)作为该参数默认值。 |

注意1，当使用http通信时，可能会存在通信安全风险，请知悉。

注意2，在Android环境中，进行https通信时还需对以下参数进行设置，设置示例如下：

```java
FLParameter flParameter = FLParameter.getInstance();
SecureSSLSocketFactory sslSocketFactory = SecureSSLSocketFactory.getInstance(applicationContext)
SecureX509TrustManager x509TrustManager = new SecureX509TrustManager(applicationContext);
flParameter.setSslSocketFactory(sslSocketFactory);
flParameter.setX509TrustManager(x509TrustManager);
```

其中 `SecureSSLSocketFactory` 、`SecureX509TrustManager` 两个对象需在Android工程中实现，需要用户根据手机中证书种类自行进行设计。

注意3，在x86环境中，进行https通信时，目前只支持自签名证书认证，还需对以下参数进行设置，设置示例如下：

```java
FLParameter flParameter = FLParameter.getInstance();
String certPath  =  "CARoot.pem";             //  端云https通信所使用的自签名根证书路径
flParameter.setCertPath(certPath);
```

注意4，在Android环境中, 当pkiVerify设置为true且云侧设置encrypt_type=PW_ENCRYPT时，还需要对以下参数进行设置，设置示例如下：

```java
FLParameter flParameter = FLParameter.getInstance();
String equipCrlPath = certPath;
long validIterInterval = 3600000;
flParameter.setEquipCrlPath(equipCrlPath);
flParameter.setValidInterval(validIterInterval);
```

其中`equipCrlPath`是设备之间证书校验需要的CRL证书，即证书吊销列表，一般可以预置 "Huawei CBG Certificate Revocation Lists" 中的设备证书CRL；`validIterInterval`一般可以设置为每轮端云聚合需要的时间（单位：毫秒，默认值为3600000），在PW_ENCRYPT模式下用来辅助防范重放攻击。

注意5，每次联邦学习任务启动前，会实例化类FLParameter进行相关参数设置。而实例化FLParameter时会自动随机生成一个clientID，用于与云侧交互过程中唯一标识该客户端，若用户需要自行设置clientID，可在实例化类FLParameter之后，调用其setClientID方法进行设置，则接着启动联邦学习任务后会使用用户设置的clientID。

创建SyncFLJob对象，并通过SyncFLJob类的flJobRun()方法启动同步联邦学习任务。

示例代码（基本http通信）如下：

1. 有监督情感分类任务示例代码

   ```java
   // 构造dataMap
   String trainTxtPath = "data/albert/supervise/client/1.txt";
   String evalTxtPath = "data/albert/supervise/eval/eval.txt";      // 非必须，getModel之后不进行验证可不设置
   String vocabFile = "data/albert/supervise/vocab.txt";                // 数据预处理的词典文件路径
   String idsFile = "data/albert/supervise/vocab_map_ids.txt"   // 词典的映射id文件路径
   Map<RunType, List<String>> dataMap = new HashMap<>();
   List<String> trainPath = new ArrayList<>();
   trainPath.add(trainTxtPath);
   trainPath.add(vocabFile);
   trainPath.add(idsFile);
   List<String> evalPath = new ArrayList<>();    // 非必须，getModel之后不进行验证可不设置
   evalPath.add(evalTxtPath);                  // 非必须，getModel之后不进行验证可不设置
   evalPath.add(vocabFile);                  // 非必须，getModel之后不进行验证可不设置
   evalPath.add(idsFile);                  // 非必须，getModel之后不进行验证可不设置
   dataMap.put(RunType.TRAINMODE, trainPath);
   dataMap.put(RunType.EVALMODE, evalPath);      // 非必须，getModel之后不进行验证可不设置

   String flName = "com.mindspore.flclient.demo.albert.AlbertClient";                             // AlBertClient.java 包路径
   String trainModelPath = "ms/albert/train/albert_ad_train.mindir0.ms";                      // 绝对路径
   String inferModelPath = "ms/albert/train/albert_ad_train.mindir0.ms";                      // 绝对路径, 和trainModelPath保持一致
   String sslProtocol = "TLSv1.2";
   String deployEnv = "android";
   String domainName = "http://10.113.216.106:6668";
   boolean ifUseElb = true;
   int serverNum = 4;
   int threadNum = 4;
   BindMode cpuBindMode = BindMode.NOT_BINDING_CORE;
   int batchSize = 32;

   FLParameter flParameter = FLParameter.getInstance();
   flParameter.setFlName(flName);
   flParameter.setDataMap(dataMap);
   flParameter.setTrainModelPath(trainModelPath);
   flParameter.setInferModelPath(inferModelPath);
   flParameter.setSslProtocol(sslProtocol);
   flParameter.setDeployEnv(deployEnv);
   flParameter.setDomainName(domainName);
   flParameter.setUseElb(useElb);
   flParameter.setServerNum(serverNum);
   flParameter.setThreadNum(threadNum);
   flParameter.setCpuBindMode(BindMode.valueOf(cpuBindMode));

   // start FLJob
   SyncFLJob syncFLJob = new SyncFLJob();
   syncFLJob.flJobRun();
   ```

2. LeNet图片分类任务示例代码

   ```java
   // 构造dataMap
   String trainImagePath = "SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_9_train_data.bin";
   String trainLabelPath = "SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_9_train_label.bin";
   String evalImagePath = "SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_data.bin";   // 非必须，getModel之后不进行验证可不设置
   String evalLabelPath = "SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_label.bin";   // 非必须，getModel之后不进行验证可不设置
   Map<RunType, List<String>> dataMap = new HashMap<>();
   List<String> trainPath = new ArrayList<>();
   trainPath.add(trainImagePath);
   trainPath.add(trainLabelPath);
   List<String> evalPath = new ArrayList<>();    // 非必须，getModel之后不进行验证可不设置
   evalPath.add(evalImagePath);                  // 非必须，getModel之后不进行验证可不设置
   evalPath.add(evalLabelPath);                  // 非必须，getModel之后不进行验证可不设置
   dataMap.put(RunType.TRAINMODE, trainPath);
   dataMap.put(RunType.EVALMODE, evalPath);      // 非必须，getModel之后不进行验证可不设置

   String flName = "com.mindspore.flclient.demo.lenet.LenetClient";                         // LenetClient.java 包路径
   String trainModelPath = "SyncFLClient/lenet_train.mindir0.ms";                      //绝对路径
   String inferModelPath = "SyncFLClient/lenet_train.mindir0.ms";                      //绝对路径
   String sslProtocol = "TLSv1.2";
   String deployEnv = "android";
   String domainName = "http://10.113.216.106:6668";
   boolean ifUseElb = true;
   int serverNum = 4;
   int threadNum = 4;
   BindMode cpuBindMode = BindMode.NOT_BINDING_CORE;
   int batchSize = 32;

   FLParameter flParameter = FLParameter.getInstance();
   flParameter.setFlName(flName);
   flParameter.setDataMap(dataMap);
   flParameter.setTrainModelPath(trainModelPath);
   flParameter.setInferModelPath(inferModelPath);
   flParameter.setSslProtocol(sslProtocol);
   flParameter.setDeployEnv(deployEnv);
   flParameter.setDomainName(domainName);
   flParameter.setUseElb(useElb);
   flParameter.setServerNum(serverNum);
   flParameter.setThreadNum(threadNum);
   flParameter.setCpuBindMode(BindMode.valueOf(cpuBindMode));
   flParameter.setBatchSize(batchSize);

   // start FLJob
   SyncFLJob syncFLJob = new SyncFLJob();
   syncFLJob.flJobRun();
   ```

## 多条数据输入推理接口modelInference()

调用modelInference()接口前，需先实例化参数类FLParameter，进行相关参数设置， 相关参数如下：

| 参数名称       | 参数类型                   | 是否必须 | 描述信息                                               | 适应API版本                                                  |
| -------------- | -------------------------- | -------- | ------------------------------------------------------ | ------------------------------------------------------------ |
| flName         | String                     | Y        | 联邦学习使用的模型脚本包路径                           | 我们提供了两个类型的模型脚本供大家参考（[有监督情感分类任务](https://gitee.com/mindspore/mindspore/tree/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert)、[LeNet图片分类任务](https://gitee.com/mindspore/mindspore/tree/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet)），对于有监督情感分类任务，该参数可设置为所提供的脚本文件[AlBertClient.java](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert/AlbertClient.java) 的包路径`com.mindspore.flclient.demo.albert.AlbertClient`；对于LeNet图片分类任务，该参数可设置为所提供的脚本文件[LenetClient.java](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet/LenetClient.java) 的包路径`com.mindspore.flclient.demo.lenet.LenetClient`。同时，用户可参考这两个类型的模型脚本，自定义模型脚本，然后将该参数设置为自定义的模型文件ModelClient.java（需继承于类[Client.java](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/java/java/fl_client/src/main/java/com/mindspore/flclient/model/Client.java)）的包路径即可。 |
| dataMap        | Map<RunType, List<String/>/> | Y        | 联邦学习数据集路径                                     | Map<RunType, List<String/>/>类型的数据集，map中key为RunType枚举类型，value为对应的数据集列表，key为RunType.TRAINMODE时代表对应的value为训练相关的数据集列表，key为RunType.EVALMODE时代表对应的value为验证相关的数据集列表， key为RunType.INFERMODE时代表对应的value为推理相关的数据集列表。 |
| inferModelPath | String                     | Y        | 联邦学习推理模型路径，为.ms文件的绝对路径              | 对于普通联邦学习模式（训练和推理使用同一个模型），该参数需设置为与trainModelPath相同；对于混合学习模式（训练和推理使用不同的模型，且云侧也包含训练过程），该参数设置为实际的推理模型路径。建议将路径设置到训练App自身目录下，保护模型本身的数据访问安全性。 |
| threadNum      | int                        | N        | 联邦学习训练和推理时使用的线程数                       | 默认值为1                                                    |
| cpuBindMode    | BindMode                   | N        | 联邦学习训练和推理时线程所需绑定的cpu内核              | BindMode枚举类型，其中BindMode.NOT_BINDING_CORE代表不绑定内核，由系统自动分配，BindMode.BIND_LARGE_CORE代表绑定大核，BindMode.BIND_MIDDLE_CORE代表绑定中核。默认值为BindMode.NOT_BINDING_CORE。 |
| batchSize      | int                        | Y        | 联邦学习训练和推理时使用的单步训练样本数，即batch size | 需与模型的输入数据的batch size保持一致。                     |

创建SyncFLJob对象，并通过SyncFLJob类的modelInference()方法启动端侧推理任务，返回推理的标签数组。

示例代码如下：

1. 有监督情感分类任务示例代码

   ```java
   // 构造dataMap
   String inferTxtPath = "data/albert/supervise/eval/eval.txt";
   String vocabFile = "data/albert/supervise/vocab.txt";
   String idsFile = "data/albert/supervise/vocab_map_ids.txt"
   Map<RunType, List<String>> dataMap = new HashMap<>();
   List<String> inferPath = new ArrayList<>();
   inferPath.add(inferTxtPath);
   inferPath.add(vocabFile);
   inferPath.add(idsFile);
   dataMap.put(RunType.INFERMODE, inferPath);

   String flName = "com.mindspore.flclient.demo.albert.AlbertClient";                             // AlBertClient.java 包路径
   String inferModelPath = "ms/albert/train/albert_ad_train.mindir0.ms";                      // 绝对路径, 和trainModelPath保持一致;
   int threadNum = 4;
   BindMode cpuBindMode = BindMode.NOT_BINDING_CORE;
   int batchSize = 32;

   FLParameter flParameter = FLParameter.getInstance();
   flParameter.setFlName(flName);
   flParameter.setDataMap(dataMap);
   flParameter.setInferModelPath(inferModelPath);
   flParameter.setThreadNum(threadNum);
   flParameter.setCpuBindMode(BindMode.valueOf(cpuBindMode));
   flParameter.setBatchSize(batchSize);

   // inference
   SyncFLJob syncFLJob = new SyncFLJob();
   int[] labels = syncFLJob.modelInference();
   ```

2. LeNet图片分类示例代码

   ```java
   // 构造dataMap
   String inferImagePath = "SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_data.bin";
   String inferLabelPath = "SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_label.bin";
   Map<RunType, List<String>> dataMap = new HashMap<>();
   List<String> inferPath = new ArrayList<>();
   inferPath.add(inferImagePath);
   inferPath.add(inferLabelPath);
   dataMap.put(RunType.INFERMODE, inferPath);

   String flName = "com.mindspore.flclient.demo.lenet.LenetClient";                         // LenetClient.java 包路径
   String inferModelPath = "SyncFLClient/lenet_train.mindir0.ms";
   int threadNum = 4;
   BindMode cpuBindMode = BindMode.NOT_BINDING_CORE;
   int batchSize = 32;

   FLParameter flParameter = FLParameter.getInstance();
   flParameter.setFlName(flName);
   flParameter.setDataMap(dataMap);
   flParameter.setInferModelPath(inferModelPath);
   flParameter.setThreadNum(threadNum);
   flParameter.setCpuBindMode(BindMode.valueOf(cpuBindMode));
   flParameter.setBatchSize(batchSize);

   // inference
   SyncFLJob syncFLJob = new SyncFLJob();
   int[] labels = syncFLJob.modelInference();
   ```

## 获取云侧最新模型接口getModel ()

调用getModel()接口前，需先实例化参数类FLParameter，进行相关参数设置， 相关参数如下：

| 参数名称       | 参数类型  | 是否必须 | 描述信息                                                     | 备注                                                         |
| -------------- | --------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| flName         | String    | Y        | 联邦学习使用的模型脚本包路径                                 | 我们提供了两个类型的模型脚本供大家参考（[有监督情感分类任务](https://gitee.com/mindspore/mindspore/tree/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert)、[LeNet图片分类任务](https://gitee.com/mindspore/mindspore/tree/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet)），对于有监督情感分类任务，该参数可设置为所提供的脚本文件[AlBertClient.java](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert/AlbertClient.java) 的包路径`com.mindspore.flclient.demo.albert.AlbertClient`；对于LeNet图片分类任务，该参数可设置为所提供的脚本文件[LenetClient.java](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet/LenetClient.java) 的包路径`com.mindspore.flclient.demo.lenet.LenetClient`。同时，用户可参考这两个类型的模型脚本，自定义模型脚本，然后将该参数设置为自定义的模型文件ModelClient.java（需继承于类[Client.java](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/java/java/fl_client/src/main/java/com/mindspore/flclient/model/Client.java)）的包路径即可。 |
| trainModelPath | String    | Y        | 联邦学习使用的训练模型路径，为.ms文件的绝对路径              | 建议将路径设置到训练App自身目录下，保护模型本身的数据访问安全性。 |
| inferModelPath | String    | Y        | 联邦学习推理模型路径，为.ms文件的绝对路径                    | 对于普通联邦学习模式（训练和推理使用同一个模型），该参数需设置为与trainModelPath相同；对于混合学习模式（训练和推理使用不同的模型，且云侧也包含训练过程），该参数设置为实际的推理模型路径。建议将路径设置到训练App自身目录下，保护模型本身的数据访问安全性。 |
| sslProtocol    | String    | N        | 端云HTTPS通信所使用的TLS协议版本                             | 设置了白名单，目前只支持"TLSv1.3"或者"TLSv1.2"。非必须设置，默认值为"TLSv1.2"。只在HTTPS通信场景中使用。 |
| deployEnv      | String    | Y        | 联邦学习的部署环境                                           | 设置了白名单，目前只支持"x86", "android"。                   |
| certPath       | String    | N        | 端云https通信所使用的自签名根证书路径                        | 当部署环境为"x86"，且端云采用自签名证书进行https通信认证时，需要设置该参数，该证书需与生成云侧自签名证书所使用的CA根证书一致才能验证通过，此参数用于非Android场景。 |
| domainName     | String    | Y        | 端云通信url                                                  | 目前，https和http通信均支持，对应格式分别为：https://......、http://......，当`useElb`设置为true时，格式必须为：https://127.0.0.0:6666 或者http://127.0.0.0:6666 ，其中`127.0.0.0`对应提供云侧服务的机器ip（即云侧参数`--scheduler_ip`），`6666`对应云侧参数`--fl_server_port`。 |
| ifUseElb       | boolean   | N        | 用于多server场景设置是否将客户端的请求随机发送给一定范围内的不同server | 设置为true代表客户端会将请求随机发给一定范围内的server地址，false代表客户端的请求会发给固定的server地址，此参数用于非Android场景，默认值为false。 |
| serverNum      | int       | N        | 客户端可选择连接的server数量                                 | 当ifUseElb设置为true时，可设置为与云侧启动server端时的`server_num`参数保持一致，用于随机选择不同的server发送信息，此参数用于非Android场景，默认值为1。 |
| serverMod      | ServerMod | Y        | 联邦学习训练模式。                                           | ServerMod枚举类型的联邦学习训练模式，其中ServerMod.FEDERATED_LEARNING代表普通联邦学习模式（训练和推理使用同一个模型）ServerMod.HYBRID_TRAINING代表混合学习模式（训练和推理使用不同的模型，且云侧也包含训练过程）。 |

注意1，当使用http通信时，可能会存在通信安全风险，请知悉。

注意2，在Android环境中，进行https通信时还需对以下参数进行设置，设置示例如下：

```java
FLParameter flParameter = FLParameter.getInstance();
SecureSSLSocketFactory sslSocketFactory = SecureSSLSocketFactory.getInstance(applicationContext)
SecureX509TrustManager x509TrustManager = new SecureX509TrustManager(applicationContext);
flParameter.setSslSocketFactory(sslSocketFactory);
flParameter.setX509TrustManager(x509TrustManager);
```

其中 `SecureSSLSocketFactory` 、`SecureX509TrustManager` 两个对象需在Android工程中实现，需要用户根据手机中证书种类自行进行设计。

注意3，在X86环境中，进行https通信时，目前只支持自签名证书认证，还需对以下参数进行设置，设置示例如下：

```java
FLParameter flParameter = FLParameter.getInstance();
String certPath  =  "CARoot.pem";             //  端云https通信所使用的自签名根证书路径
flParameter.setCertPath(certPath);
```

注意4，在调用getModel方法前，会实例化类FLParameter进行相关参数设置。而实例化FLParameter时会自动随机生成一个clientID，用于与云侧交互过程中唯一标识该客户端，若用户需要自行设置clientID，可在实例化类FLParameter之后，调用其setCertPath方法进行设置，则接着启动getModel任务后会使用用户设置的clientID。

创建SyncFLJob对象，并通过SyncFLJob类的getModel()方法启动异步推理任务，返回getModel请求状态码。

示例代码如下：

1. 有监督情感分类任务版本

   ```java
   String flName = "com.mindspore.flclient.demo.albert.AlbertClient";                         // AlBertClient.java 包路径
   String trainModelPath = "ms/albert/train/albert_ad_train.mindir0.ms";                      //绝对路径
   String inferModelPath = "ms/albert/train/albert_ad_train.mindir0.ms";                      //绝对路径, 和trainModelPath保持一致
   String sslProtocol = "TLSv1.2";
   String deployEnv = "android";
   String domainName = "http://10.113.216.106:6668";
   boolean ifUseElb = true;
   int serverNum = 4;
   ServerMod serverMod = ServerMod.FEDERATED_LEARNING;

   FLParameter flParameter = FLParameter.getInstance();
   flParameter.setFlName(flName);
   flParameter.setTrainModelPath(trainModelPath);
   flParameter.setInferModelPath(inferModelPath);
   flParameter.setSslProtocol(sslProtocol);
   flParameter.setDeployEnv(deployEnv);
   flParameter.setDomainName(domainName);
   flParameter.setUseElb(useElb);
   flParameter.setServerNum(serverNum);
   flParameter.setServerMod(ServerMod.valueOf(serverMod));

   // getModel
   SyncFLJob syncFLJob = new SyncFLJob();
   syncFLJob.getModel();
   ```

2. LeNet图片分类任务版本

   ```java
   String flName = "com.mindspore.flclient.demo.lenet.LenetClient";                         // LenetClient.java 包路径
   String trainModelPath = "SyncFLClient/lenet_train.mindir0.ms";                      //绝对路径
   String inferModelPath = "SyncFLClient/lenet_train.mindir0.ms";                      //绝对路径, 和trainModelPath保持一致
   String sslProtocol = "TLSv1.2";
   String deployEnv = "android";
   String domainName = "http://10.113.216.106:6668";
   boolean ifUseElb = true;
   int serverNum = 4;
   ServerMod serverMod = ServerMod.FEDERATED_LEARNING;

   FLParameter flParameter = FLParameter.getInstance();
   flParameter.setFlName(flName);
   flParameter.setTrainModelPath(trainModelPath);
   flParameter.setInferModelPath(inferModelPath);
   flParameter.setSslProtocol(sslProtocol);
   flParameter.setDeployEnv(deployEnv);
   flParameter.setDomainName(domainName);
   flParameter.setUseElb(useElb);
   flParameter.setServerNum(serverNum);
   flParameter.setServerMod(ServerMod.valueOf(serverMod));

   // getModel
   SyncFLJob syncFLJob = new SyncFLJob();
   syncFLJob.getModel();
   ```
