# Examples

<!-- TOC -->

- [Examples](#examples)
    - [flJobRun() for Starting Federated Learning](#fljobrun-for-starting-federated-learning)
    - [ModelInference() for Inferring Multiple Input Data Records](#modelinference-for-inferring-multiple-input-data-records)
    - [getModel() for Obtaining the Latest Model on the Cloud](#getmodel-for-obtaining-the-latest-model-on-the-cloud)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/docs/source_en/interface_description_federated_client.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

Note that before using the following interfaces, you can first refer to the document [on-device deployment](https://www.mindspore.cn/federated/docs/en/master/deploy_federated_client.html) to deploy related environments.

## flJobRun() for Starting Federated Learning

Before calling the flJobRun() API, instantiate the parameter class FLParameter and set related parameters as follows:

| Parameter            | Type                         | Mandatory | Description                                                  | Remarks                                                      |
| -------------------- | ---------------------------- | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| dataMap              | Map<RunType, List<String/>/> | Y         | The path of Federated learning dataset.                      | The dataset of Map<RunType, List<String/>/> type, the key in the map is the RunType enumeration type, the value is the corresponding dataset list, when the key is RunType.TRAINMODE, the corresponding value is the training-related dataset list, when the key  is RunType.EVALMODE, it means that the corresponding value is a list of verification-related datasets, and when the key is RunType.INFERMODE, it means that the corresponding value is a list of inference-related datasets. |
| flName               | String                       | Y         | The package path of model script used by federated learning. | We provide two types of model scripts for your reference ([Supervised sentiment classification task](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert), [LeNet image classification task](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet)). For supervised sentiment classification tasks, this parameter can be set to the package path of the provided script file [AlBertClient.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert/AlbertClient.java), like as `com.mindspore.flclient.demo.albert.AlbertClient`; for LeNet image classification tasks, this parameter can be set to the package path of the provided script file [LenetClient.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet/LenetClient.java), like as `com.mindspore.flclient.demo.lenet.LenetClient`. At the same time, users can refer to these two types of model scripts, define the model script by themselves, and then set the parameter to the package path of the customized model file ModelClient.java (which needs to inherit from the class [Client.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/java/fl_client/src/main/java/com/mindspore/flclient/model/Client.java)). |
| trainModelPath       | String                       | Y         | Path of a training model used for federated learning, which is an absolute path of the .ms file. | It is recommended to set the path to the training App's own directory to protect the data access security of the model itself. |
| inferModelPath       | String                       | Y         | Path of an inference model used for federated learning, which is an absolute path of the .ms file. | For the normal federated learning mode (training and inference use the same model), the value of this parameter needs to be the same as that of `trainModelPath`; for the hybrid learning mode (training and inference use different models, and the server side also includes training process), this parameter is set to the path of actual inference model. It is recommended to set the path to the training App's own directory to protect the data access security of the model itself. |
| sslProtocol          | String                       | N         | The TLS protocol version used by the device-cloud HTTPS communication. | A whitelist is set, and currently only "TLSv1.3" or "TLSv1.2" is supported. Only need to set it up in the HTTPS communication scenario. |
| deployEnv            | String                       | Y         | The deployment environment for federated learning.           | A whitelist is set, currently only "x86", "android" are supported. |
| certPath             | String                       | N         | The self-signed root certificate path used for device-cloud HTTPS communication. | When the deployment environment is "x86" and the device-cloud uses a self-signed certificate for HTTPS communication authentication, this parameter needs to be set. The certificate must be consistent with the CA root certificate used to generate the cloud-side self-signed certificate to pass the verification. This parameter is used for non-Android scenarios. |
| domainName           | String                       | Y         | The url for device-cloud communication.                      | Currently, https and http communication are supported, the corresponding formats are like: https://......, http://......, and when `useElb` is set to true, the format must be: https://127.0.0.0 : 6666 or http://127.0.0.0 : 6666 , where `127.0.0.0` corresponds to the ip of the machine providing cloud-side services (corresponding to the cloud-side parameter `--scheduler_ip`), and `6666` corresponds to the cloud-side parameter `--fl_server_port`. |
| ifUseElb             | boolean                      | N         | Used for multi-server scenarios to set whether to randomly send client requests to different servers within a certain range. | Setting to true means that the client will randomly send requests to a certain range of server addresses, and false means that the client's requests will be sent to a fixed server address. This parameter is used in non-Android scenarios, and the default value is false. |
| serverNum            | int                          | N         | The number of servers that the client can choose to connect to. | When `ifUseElb` is set to true, it can be set to be consistent with the `server_num` parameter when the server is started on the cloud side. It is used to randomly select different servers to send information. This parameter is used in non-Android scenarios. The default value is 1. |
| ifPkiVerify          | boolean                      | N         | The switch of device-cloud identity authentication.          | Set to true to enable device-cloud security authentication, set to false to disable, and the default value is false. Identity authentication requires HUKS to provide a certificate. This parameter is only used in the Android environment (currently only supports HUAWEI phones). |
| threadNum            | int                          | N         | The number of threads used in federated learning training and inference. | The default value is 1.                                      |
| cpuBindMode          | BindMode                     | N         | The cpu core that threads need to bind during federated learning training and inference. | It is the enumeration type `BindMode`, where BindMode.NOT_BINDING_CORE represents the unbound core, which is automatically assigned by the system, BindMode.BIND_LARGE_CORE represents the bound large core, and BindMode.BIND_MIDDLE_CORE represents the bound middle core. The default value is BindMode.NOT_BINDING_CORE. |
| batchSize            | int                          | Y         | The number of single-step training samples used in federated learning training and inference, that is, batch size. | It needs to be consistent with the batch size of the input data of the model. |
| iflJobResultCallback | IFLJobResultCallback         | N         | The federated learning callback function object `iflJobResultCallback`. | The user can implement the specific method of the interface class [IFLJobResultCallback.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/java/fl_client/src/main/java/com/mindspore/flclient/IFLJobResultCallback.java) in the project according to the needs of the actual scene, and set it as a callback function object in the federated learning task. We provide a simple implementation use case [FLJobResultCallback.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/java/fl_client/src/main/java/com/mindspore/flclient/FLJobResultCallback.java) as the default value of this parameter. |

Note 1: When using HTTP communication, there may exist communication security risks, please be aware.

Note 2: In the Android environment, the following parameters need to be set when using HTTPS communication. The setting examples are as follows:

```java
FLParameter flParameter = FLParameter.getInstance();
SecureSSLSocketFactory sslSocketFactory = SecureSSLSocketFactory.getInstance(applicationContext)
SecureX509TrustManager x509TrustManager = new SecureX509TrustManager(applicationContext);
flParameter.setSslSocketFactory(sslSocketFactory);
flParameter.setX509TrustManager(x509TrustManager);
```

Among them, the two objects `SecureSSLSocketFactory` and `SecureX509TrustManager` need to be implemented in the Android project, and users need to design by themselves according to the type of certificate in the mobile phone.

Note 3: In the x86 environment, currently only self-signed certificate authentication is supported when using HTTPS communication, and the following parameters need to be set. The setting examples are as follows:

```java
FLParameter flParameter = FLParameter.getInstance();
String certPath  =  "CARoot.pem";             //  the self-signed root certificate path used for device-cloud HTTPS communication.
flParameter.setCertPath(certPath);
```

Note 4: In the Android environment, when `pkiVerify` is set to true and encrypt_type is set to PW_ENCRYPT on the cloud side, the following parameters need to be set. The setting examples are as follows:

```java
FLParameter flParameter = FLParameter.getInstance();
String equipCrlPath = certPath;
long validIterInterval = 3600000;
flParameter.setEquipCrlPath(equipCrlPath);
flParameter.setValidInterval(validIterInterval);
```

Among them, `equipCrlPath` is the CRL certificate required for certificate verification among devices, that is, the certificate revocation list. Generally, the device certificate CRL in "Huawei CBG Certificate Revocation Lists" can be preset; `validIterInterval` which is used to help prevent replay attacks in PW_ENCRYPT mode can generally be set to the time required for each round of device-cloud aggregation (unit: milliseconds, the default value is 3600000).

Note 5: Before each federated learning task is started, the FLParameter class will be instantiated for related parameter settings. When FLParameter is instantiated, a clientID is automatically generated randomly, which is used to uniquely identify the client during the interaction with the cloud side. If the user needs to set the clientID by himself, after instantiating the FLParameter class, call its setClientID method to set it, and then after starting the federated learning task, the clientID set by the user will be used.

Create a SyncFLJob object and use the flJobRun() method of the SyncFLJob class to start a federated learning task.

The sample code (basic http communication) is as follows:

1. Sample code of a supervised sentiment classification task

   ```java
   // create dataMap
   String trainTxtPath = "data/albert/supervise/client/1.txt";
   String evalTxtPath = "data/albert/supervise/eval/eval.txt";      // Not necessary, if you don't need verify model accuracy after getModel, you don't need to set this parameter
   String vocabFile = "data/albert/supervise/vocab.txt";                // Path of the dictionary file for data preprocessing.
   String idsFile = "data/albert/supervise/vocab_map_ids.txt"   // Path of the mapping ID file of a dictionary.
   Map<RunType, List<String>> dataMap = new HashMap<>();
   List<String> trainPath = new ArrayList<>();
   trainPath.add(trainTxtPath);
   trainPath.add(vocabFile);
   trainPath.add(idsFile);
   List<String> evalPath = new ArrayList<>();    // Not necessary, if you don't need verify model accuracy after getModel, you don't need to set this parameter
   evalPath.add(evalTxtPath);                  // Not necessary, if you don't need verify model accuracy after getModel, you don't need to set this parameter
   evalPath.add(vocabFile);                  // Not necessary, if you don't need verify model accuracy after getModel, you don't need to set this parameter
   evalPath.add(idsFile);                  // Not necessary, if you don't need verify model accuracy after getModel, you don't need to set this parameter
   dataMap.put(RunType.TRAINMODE, trainPath);
   dataMap.put(RunType.EVALMODE, evalPath);      // Not necessary, if you don't need verify model accuracy after getModel, you don't need to set this parameter

   String flName = "com.mindspore.flclient.demo.albert.AlbertClient";                             // The package path of AlBertClient.java
   String trainModelPath = "ms/albert/train/albert_ad_train.mindir0.ms";                      // Absolute path
   String inferModelPath = "ms/albert/train/albert_ad_train.mindir0.ms";                      // Absolute path, consistent with trainModelPath
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

2. Sample code of a LeNet image classification task

   ```java
   // create dataMap
   String trainImagePath = "SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_9_train_data.bin";
   String trainLabelPath = "SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_9_train_label.bin";
   String evalImagePath = "SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_data.bin";   // Not necessary, if you don't need verify model accuracy after getModel, you don't need to set this parameter
   String evalLabelPath = "SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_label.bin";   // Not necessary, if you don't need verify model accuracy after getModel, you don't need to set this parameter
   Map<RunType, List<String>> dataMap = new HashMap<>();
   List<String> trainPath = new ArrayList<>();
   trainPath.add(trainImagePath);
   trainPath.add(trainLabelPath);
   List<String> evalPath = new ArrayList<>();    // Not necessary, if you don't need verify model accuracy after getModel, you don't need to set this parameter
   evalPath.add(evalImagePath);                  // Not necessary, if you don't need verify model accuracy after getModel, you don't need to set this parameter
   evalPath.add(evalLabelPath);                  // Not necessary, if you don't need verify model accuracy after getModel, you don't need to set this parameter
   dataMap.put(RunType.TRAINMODE, trainPath);
   dataMap.put(RunType.EVALMODE, evalPath);      // Not necessary, if you don't need verify model accuracy after getModel, you don't need to set this parameter

   String flName = "com.mindspore.flclient.demo.lenet.LenetClient";                         // The package path of LenetClient.java
   String trainModelPath = "SyncFLClient/lenet_train.mindir0.ms";                      // Absolute path
   String inferModelPath = "SyncFLClient/lenet_train.mindir0.ms";                      // Absolute path, consistent with trainModelPath
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

## modelInference() for Inferring Multiple Input Data Records

Before calling the modelInference() API, instantiate the parameter class FLParameter and set related parameters as follows:

| Parameter      | Type                         | Mandatory | Description                                                  | Remarks                                                      |
| -------------- | ---------------------------- | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| flName         | String                       | Y         | The package path of model script used by federated learning. | We provide two types of model scripts for your reference ([Supervised sentiment classification task](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert), [LeNet image classification task](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet)). For supervised sentiment classification tasks, this parameter can be set to the package path of the provided script file [AlBertClient.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert/AlbertClient.java), like as `com.mindspore.flclient.demo.albert.AlbertClient`; for LeNet image classification tasks, this parameter can be set to the package path of the provided script file [LenetClient.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet/LenetClient.java), like as `com.mindspore.flclient.demo.lenet.LenetClient`. At the same time, users can refer to these two types of model scripts, define the model script by themselves, and then set the parameter to the package path of the customized model file ModelClient.java (which needs to inherit from the class [Client.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/java/fl_client/src/main/java/com/mindspore/flclient/model/Client.java)). |
| dataMap        | Map<RunType, List<String/>/> | Y         | The path of Federated learning dataset.                      | The dataset of Map<RunType, List<String/>/> type, the key in the map is the RunType enumeration type, the value is the corresponding dataset list, when the key is RunType.TRAINMODE, the corresponding value is the training-related dataset list, when the key  is RunType.EVALMODE, it means that the corresponding value is a list of verification-related datasets, and when the key is RunType.INFERMODE, it means that the corresponding value is a list of inference-related datasets. |
| inferModelPath | String                       | Y         | Path of an inference model used for federated learning, which is an absolute path of the .ms file. | For the normal federated learning mode (training and inference use the same model), the value of this parameter needs to be the same as that of `trainModelPath`; for the hybrid learning mode (training and inference use different models, and the server side also includes training process), this parameter is set to the path of actual inference model. It is recommended to set the path to the training App's own directory to protect the data access security of the model itself. |
| threadNum      | int                          | N         | The number of threads used in federated learning training and inference. | The default value is 1.                                      |
| cpuBindMode    | BindMode                     | N         | The cpu core that threads need to bind during federated learning training and inference. | It is the enumeration type `BindMode`, where BindMode.NOT_BINDING_CORE represents the unbound core, which is automatically assigned by the system, BindMode.BIND_LARGE_CORE represents the bound large core, and BindMode.BIND_MIDDLE_CORE represents the bound middle core. The default value is BindMode.NOT_BINDING_CORE. |
| batchSize      | int                          | Y         | The number of single-step training samples used in federated learning training and inference, that is, batch size. | It needs to be consistent with the batch size of the input data of the model. |

Create a SyncFLJob object and use the modelInference() method of the SyncFLJob class to start an inference task on the device. The inferred label array is returned.

The sample code is as follows:

1. Sample code of a supervised sentiment classification task

   ```java
   // create dataMap
   String inferTxtPath = "data/albert/supervise/eval/eval.txt";
   String vocabFile = "data/albert/supervise/vocab.txt";
   String idsFile = "data/albert/supervise/vocab_map_ids.txt"
   Map<RunType, List<String>> dataMap = new HashMap<>();
   List<String> inferPath = new ArrayList<>();
   inferPath.add(inferTxtPath);
   inferPath.add(vocabFile);
   inferPath.add(idsFile);
   dataMap.put(RunType.INFERMODE, inferPath);

   String flName = "com.mindspore.flclient.demo.albert.AlbertClient";                             // The package path of AlBertClient.java
   String inferModelPath = "ms/albert/train/albert_ad_train.mindir0.ms";                      // Absolute path, consistent with trainModelPath
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

2. Sample code of a LeNet image classification

   ```java
   // create dataMap
   String inferImagePath = "SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_data.bin";
   String inferLabelPath = "SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_label.bin";
   Map<RunType, List<String>> dataMap = new HashMap<>();
   List<String> inferPath = new ArrayList<>();
   inferPath.add(inferImagePath);
   inferPath.add(inferLabelPath);
   dataMap.put(RunType.INFERMODE, inferPath);

   String flName = "com.mindspore.flclient.demo.lenet.LenetClient";                         // The package path of LenetClient.java package
   String inferModelPath = "SyncFLClient/lenet_train.mindir0.ms";                              // Absolute path, consistent with trainModelPath
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

## getModel() for Obtaining the Latest Model on the Cloud

Before calling the getModel() API, instantiate the parameter class FLParameter and set related parameters as follows:

| Parameter      | Type      | Mandatory | Description                                                  | Remarks                                                      |
| -------------- | --------- | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| flName         | String    | Y         | The package path of model script used by federated learning. | We provide two types of model scripts for your reference ([Supervised sentiment classification task](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert), [LeNet image classification task](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet)). For supervised sentiment classification tasks, this parameter can be set to the package path of the provided script file [AlBertClient.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert/AlbertClient.java), like as `com.mindspore.flclient.demo.albert.AlbertClient`; for LeNet image classification tasks, this parameter can be set to the package path of the provided script file [LenetClient.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet/LenetClient.java), like as `com.mindspore.flclient.demo.lenet.LenetClient`. At the same time, users can refer to these two types of model scripts, define the model script by themselves, and then set the parameter to the package path of the customized model file ModelClient.java (which needs to inherit from the class [Client.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/java/fl_client/src/main/java/com/mindspore/flclient/model/Client.java)). |
| trainModelPath | String    | Y         | Path of a training model used for federated learning, which is an absolute path of the .ms file. | It is recommended to set the path to the training App's own directory to protect the data access security of the model itself. |
| inferModelPath | String    | Y         | Path of an inference model used for federated learning, which is an absolute path of the .ms file. | For the normal federated learning mode (training and inference use the same model), the value of this parameter needs to be the same as that of `trainModelPath`; for the hybrid learning mode (training and inference use different models, and the server side also includes training process), this parameter is set to the path of actual inference model. It is recommended to set the path to the training App's own directory to protect the data access security of the model itself. |
| sslProtocol    | String    | N         | The TLS protocol version used by the device-cloud HTTPS communication. | A whitelist is set, and currently only "TLSv1.3" or "TLSv1.2" is supported. Only need to set it up in the HTTPS communication scenario. |
| deployEnv      | String    | Y         | The deployment environment for federated learning.           | A whitelist is set, currently only "x86", "android" are supported. |
| certPath       | String    | N         | The self-signed root certificate path used for device-cloud HTTPS communication. | When the deployment environment is "x86" and the device-cloud uses a self-signed certificate for HTTPS communication authentication, this parameter needs to be set. The certificate must be consistent with the CA root certificate used to generate the cloud-side self-signed certificate to pass the verification. This parameter is used for non-Android scenarios. |
| domainName     | String    | Y         | The url for device-cloud communication.                      | Currently, https and http communication are supported, the corresponding formats are like: https://......, http://......, and when `useElb` is set to true, the format must be: https://127.0.0.0 : 6666 or http://127.0.0.0 : 6666 , where `127.0.0.0` corresponds to the ip of the machine providing cloud-side services (corresponding to the cloud-side parameter `--scheduler_ip`), and `6666` corresponds to the cloud-side parameter `--fl_server_port`. |
| ifUseElb       | boolean   | N         | Used for multi-server scenarios to set whether to randomly send client requests to different servers within a certain range. | Setting to true means that the client will randomly send requests to a certain range of server addresses, and false means that the client's requests will be sent to a fixed server address. This parameter is used in non-Android scenarios, and the default value is false. |
| serverNum      | int       | N         | The number of servers that the client can choose to connect to. | When `ifUseElb` is set to true, it can be set to be consistent with the `server_num` parameter when the server is started on the cloud side. It is used to randomly select different servers to send information. This parameter is used in non-Android scenarios. The default value is 1. |
| serverMod      | ServerMod | Y         | The federated learning training mode.                        | The federated learning training mode of ServerMod enumeration type, where ServerMod.FEDERATED_LEARNING represents the normal federated learning mode (training and inference use the same model) ServerMod.HYBRID_TRAINING represents the hybrid learning mode (training and inference use different models, and the server side also includes training process). |

Note 1: When using HTTP communication, there may exist communication security risks, please be aware.

Note 2: In the Android environment, the following parameters need to be set when using HTTPS communication. The setting examples are as follows:

```java
FLParameter flParameter = FLParameter.getInstance();
SecureSSLSocketFactory sslSocketFactory = SecureSSLSocketFactory.getInstance(applicationContext)
SecureX509TrustManager x509TrustManager = new SecureX509TrustManager(applicationContext);
flParameter.setSslSocketFactory(sslSocketFactory);
flParameter.setX509TrustManager(x509TrustManager);
```

Among them, the two objects `SecureSSLSocketFactory` and `SecureX509TrustManager` need to be implemented in the Android project, and users need to design themselves according to the type of certificate in the mobile phone.

Note 3: In the x86 environment, currently only self-signed certificate authentication is  supported when using HTTPS communication, and the following parameters need to be set. The setting examples are as follows:

```java
FLParameter flParameter = FLParameter.getInstance();
String certPath  =  "CARoot.pem";             //  the self-signed root certificate path used for device-cloud HTTPS communication.
flParameter.setCertPath(certPath);
```

Note 4: Before calling the getModel method, the FLParameter class will be instantiated for related parameter settings. When FLParameter is instantiated, a clientID is automatically generated randomly, which is used to uniquely identify the client during the interaction with the cloud side. If the user needs to set the clientID by himself, after instantiating the FLParameter class, call its setCertPath method to set it, and then after starting the getModel task, the clientID set by the user will be used.

Create a SyncFLJob object and use the getModel() method of the SyncFLJob class to start an asynchronous inference task. The status code of the getModel request is returned.

The sample code is as follows:

1. Supervised sentiment classification task

   ```java
   String flName = "com.mindspore.flclient.demo.albert.AlbertClient";                             // The package path of AlBertClient.java package
   String trainModelPath = "ms/albert/train/albert_ad_train.mindir0.ms";                      // Absolute path
   String inferModelPath = "ms/albert/train/albert_ad_train.mindir0.ms";                      // Absolute path, consistent with trainModelPath
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

2. LeNet image classification task

   ```java
   String flName = "com.mindspore.flclient.demo.lenet.LenetClient";                         // The package path of LenetClient.java package
   String trainModelPath = "SyncFLClient/lenet_train.mindir0.ms";                             // Absolute path
   String inferModelPath = "SyncFLClient/lenet_train.mindir0.ms";                             // Absolute path, consistent with trainModelPath
   String sslProtocol = "TLSv1.2";
   String deployEnv = "android";
   String domainName = "http://10.113.216.106:6668";
   boolean ifUseElb = true;
   int serverNum = 4
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