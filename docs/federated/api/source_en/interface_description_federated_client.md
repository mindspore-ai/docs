# Examples

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/federated/api/source_en/interface_description_federated_client.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

## flJobRun() for Starting Federated Learning

Before calling the flJobRun() API, instantiate the parameter class FLParameter and set related parameters as follows:

| Parameter       | Type | Mandatory | Description                                                    | Remarks                                                         |
| -------------- | -------- | -------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| trainDataset   | String   | Y        | Path of the training dataset                                              | The training data of a sentiment classification task is in .txt format. The training files data.bin and label.bin in the image classification task are combined by commas (,).|
| vocabFile      | String   | Y        | Path of the dictionary file for data preprocessing                                    | This parameter is mandatory for sentiment classification tasks. This parameter does not need to be set for image classification tasks, and the default value is null.|
| idsFile        | String   | Y        | Path of the mapping ID file of a dictionary                                        | This parameter is mandatory for sentiment classification tasks. This parameter does not need to be set for image classification tasks, and the default value is null.|
| testDataset    | String   | N        | Path of the test dataset                                              | 1. For image classification tasks, this parameter does not need to be set and the default value is null. For sentiment classification tasks, if this parameter is not set, verification is not performed during training.<br />2. The test data of a sentiment classification task is in .txt format. The test files data.bin and label.bin in the image classification task are combined by commas (,).|
| flName         | String   | Y        | Name of the model used for federated learning                                      | Set this parameter to `albert` for sentiment classification tasks or `lenet` for the LeNet scenario.|
| trainModelPath | String   | Y        | Path of a training model used for federated learning, which is an absolute path of the .ms file|                                                              |
| inferModelPath | String   | Y        | Path of an inference model used for federated learning, which is an absolute path of the .ms file| The value of this parameter need to be the same as that of trainModelPath for supervised sentiment classification tasks and image classification tasks. |
| clientID       | String   | Y        | Unique ID of a client                                      |                                                              |
| ip             | String   | Y        | IP address of a service started on the server, for example, `10.*.*.*`| The IP address and port number will be changed to the hostname later.|
| port           | Integer| Y        | Port number of a service started on the server                                  | The IP address and port number will be changed to the hostname later.|
| useHttps       | Boolean| N        | Whether to perform SSL certificate authentication for device-cloud communication                                   | The value false indicates HTTP communication, and the value true indicates HTTPS communication. The default value is false. |
| useSSL         | Boolean| N        | Whether to perform SSL certificate authentication (which applies only to the HTTPS communication scenario) for device-cloud communication | The value false indicates that SSL certificate authentication is not performed. The value true indicates that SSL certificate authentication is performed. The default value is false. |

When `useSSL` is set to `true`, only HTTPS communication is supported. In the preceding parameters, `useHttps` must be set to `true`. In addition, the following parameters need to be set:

```java
FLParameter flParameter = FLParameter.getInstance();
String hostName  =  "10.*.*.*";
String certPath  =  "client.crt";             // Provide the absolute path of the certificate.
flParameter.setHostName(hostName);
flParameter.setCertPath(certPath);
```

Create a SyncFLJob object and use the flJobRun() method of the SyncFLJob class to start a federated learning task.

The sample code is as follows:

1. Sample code of a sentiment classification task

   ```java
   // set parameters
   String trainDataset = "SyncFLClient/data/albert/client/0.txt";                        // Absolute path
   String vocal_file = "SyncFLClient/data/albert/vocab.txt";                           // Absolute path
   String idsFile = "SyncFLClient/data/albert/vocab_map_ids.txt";                      // Absolute path
   String testDataset = "SyncFLClient/data/albert/eval/eval.txt";    // Absolute path. If no independent test set is included, use the training data as the test set or do not perform the test (do not set this parameter).
   String flName = "albert";  
   String trainModelPath = "SyncFLClient/ms/albert/albert_train.mindir.ms";                      // Absolute path
   String inferModelPath = "SyncFLClient/ms/albert/albert_train.mindir.ms";                      // Absolute path
   String clientID = UUID.randomUUID().toString();
   String ip = "10.*.*.*";
   int port = 6668;
   boolean useHttps = false;
   boolean useSSL = false;

   FLParameter flParameter = FLParameter.getInstance();
   flParameter.setTrainDataset(trainDataset);
   flParameter.setVocabFile(vocabFile);
   flParameter.setIdsFile(idsFile);
   flParameter.setTestDataset(testDataset);
   flParameter.setFlName(flName);
   flParameter.setTrainModelPath(trainModelPath);
   flParameter.setInferModelPath(inferModelPath);
   flParameter.setClientID(clientID);
   flParameter.setIp(ip);
   flParameter.setPort(port);
   flParameter.setUseHttps(useHttps);
   flParameter.setUseSSL(useSSL);

   // start FLJob
   SyncFLJob syncFLJob = new SyncFLJob();
   syncFLJob.flJobRun();
   ```

2. Sample code of a LeNet image classification task

   ```java
   // set parameters
   String trainDataset = "SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_9_train_data.bin,SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_9_train_label.bin";                        // Absolute path
   String testDataset = "SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_data.bin,SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_label.bin";    // Absolute path. If no independent test set is included, use the training data as the test set or do not perform the test (do not set this parameter).
   String flName = "lenet";
   String trainModelPath = "SyncFLClient/lenet_train.mindir0.ms";                      // Absolute path
   String inferModelPath = "SyncFLClient/lenet_train.mindir0.ms";                      // Absolute path
   String clientID = UUID.randomUUID().toString();
   String ip = "10.*.*.*";
   int port = 6668;
   boolean useHttps = false;
   boolean useSSL = false;

   FLParameter flParameter = FLParameter.getInstance();
   flParameter.setTrainDataset(trainDataset);
   flParameter.setTestDataset(testDataset);
   flParameter.setFlName(flName);
   flParameter.setTrainModelPath(trainModelPath);
   flParameter.setInferModelPath(inferModelPath);
   flParameter.setClientID(clientID);
   flParameter.setIp(ip);
   flParameter.setPort(port);
   flParameter.setUseHttps(useHttps);
   flParameter.setUseSSL(useSSL);

   // start FLJob
   SyncFLJob syncFLJob = new SyncFLJob();
   syncFLJob.flJobRun();
   ```

## modelInference() for Inferring Multiple Input Data Records

Before calling the modelInference() API, instantiate the parameter class FLParameter and set related parameters as follows:

| Parameter  | Type | Mandatory | Description                                  | Applicable API Version                                                 |
| --------- | -------- | -------- | ----------------------------------------- | ----------------------------------------------------------- |
| flName    | String   | Y        | Name of the model used for federated learning                    | Set this parameter to `albert` for sentiment classification tasks or `lenet` for image classification tasks.|
| dataPath  | String   | Y        | Dataset path                                | The sentiment classification task is in .txt format without labels. The image classification task is in .bin format. |
| vocabFile | String   | Y        | Path of the dictionary file for data preprocessing                  | This parameter is mandatory for sentiment classification tasks.  This parameter is not needed for image classification task. |
| idsFile   | String   | Y        | Path of the mapping ID file of a dictionary                      | This parameter is mandatory for sentiment classification tasks. This parameter is not needed for image classification task. |
| modelPath | String   | Y        | Path of an inference model used for federated learning, which is an absolute path of the .ms file| Both the supervised sentiment classification task and the image classification task need to be set as the same as trainModelPath that used in the federated learning training task |

Create a SyncFLJob object and use the modelInference() method of the SyncFLJob class to start an inference task on the device. The inferred label array is returned.

The sample code is as follows:

1. Sample code of a sentiment classification task

   ```java
   // set parameters
   String flName = "albert";
   String dataPath = "SyncFLClient/data/albert/eval/eval.txt";                             // Absolute path
   String vocal_file = "SyncFLClient/data/albert/vocab.txt";                                  // Absolute path
   String idsFile = "SyncFLClient/data/albert/vocab_map_ids.txt";                    // Absolute path
   String modelPath = "SyncFLClient/ms/albert/albert_train.mindir.ms";                                // Absolute path
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

2. Sample code of a LeNet image classification

   ```java
   // set parameters
   String flName = "lenet";
   String dataPath = "/SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_data.bin,/SyncFLClient/data/3500_clients_bin/f0178_39/f0178_39_bn_1_test_label.bin";     //绝对路径
   String modelPath = "SyncFLClient/lenet_train.mindir0.ms";            // Absolute path
   FLParameter flParameter = FLParameter.getInstance();
   flParameter.setFlName(flName);
flParameter.setTestDataset(dataPath);
   flParameter.setInferModelPath(modelPath);

   // inference
   SyncFLJob syncFLJob = new SyncFLJob();
   int[] labels = syncFLJob.modelInference();
   ```

## getModel() for Obtaining the Latest Model on the Cloud

Before calling the getModel() API, instantiate the parameter class FLParameter and set related parameters as follows:

| Parameter       | Type | Mandatory | Description                                                     | Remarks                                                         |
| -------------- | -------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| flName         | String   | Y        | Name of the model used for federated learning                                       | Set this parameter to `albert` for sentiment classification tasks or `lenet` for the LeNet scenario.|
| trainModelPath | String   | Y        | Path of a training model used for federated learning, which is an absolute path of the .ms file|                                                              |
| inferModelPath | String   | Y        | Path of an inference model used for federated learning, which is an absolute path of the .ms file| The value of this parameter need to be the same as that of trainModelPath for supervised sentiment classification tasks and image classification tasks. |
| ip             | String   | Y        | IP address of a service started on the server, for example, `10.*.*.*`| The IP address and port number will be changed to the hostname later.|
| port           | Integer| Y        | Port number of a service started on the server                                  | The IP address and port number will be changed to the hostname later.|
| useHttps       | Boolean| N        | Whether to perform SSL certificate authentication for device-cloud communication                                   | The value false indicates HTTP communication, and the value true indicates HTTPS communication. The default value is false. |
| useSSL         | Boolean| N        | Whether to perform SSL certificate authentication (which applies only to the HTTPS communication scenario) for device-cloud communication | The value false indicates that SSL certificate authentication is not performed. The value true indicates that SSL certificate authentication is performed. The default value is false. |
| useElb         | Boolean| Y        | Whether to simulate the elastic load balancing. The value true indicates that the client sends requests to a random server address within a specified range. The value false indicates that the client sends a request to a fixed server address.|                                                              |
| serverNum      | Integer| Y        | Number of servers that can send requests during elastic load balancing simulation, which must be the same as the number of servers started on the cloud|                                                              |

When `useSSL` is set to `true`, only HTTPS communication is supported. In the preceding parameters, `useHttps` must be set to `true`. In addition, the following parameters need to be set:

```java
FLParameter flParameter = FLParameter.getInstance();
String hostName  =  "10.*.*.*";
String certPath  =  "client.crt";             // Provide the absolute path of the certificate.

flParameter.setHostName(hostName);
flParameter.setCertPath(certPath);
```

Create a SyncFLJob object and use the getModel() method of the SyncFLJob class to start an asynchronous inference task. The status code of the getModel request is returned.

The sample code is as follows:

1. Sentiment classification task

   ```java
   // set parameters
   String flName = "albert";     // Set this parameter to `albert` for sentiment classification tasks or `lenet` for image classification tasks.
   String trainModelPath = "SyncFLClient/ms/albert/albert_train.mindir.ms";                      // Absolute path
   String inferModelPath = "SyncFLClient/ms/albert/albert_train.mindir.ms";                      // Absolute path
   String ip = "10.*.*.*";
   int port = 6668;
   boolean useHttps = false;
   boolean useSSL = false;
   boolean useElb = false;
   int serverNum = 1;

   FLParameter flParameter = FLParameter.getInstance();
   flParameter.setFlName(flName);
   flParameter.setTrainModelPath(trainModelPath);
   flParameter.setInferModelPath(inferModelPath);
   flParameter.setIp(ip);
   flParameter.setPort(port);
   flParameter.setUseHttps(useHttps);
   flParameter.setUseSSL(useSSL);
   flParameter.setUseElb(useElb);
   flParameter.setServerNum(serverNum);

   // getModel
   SyncFLJob syncFLJob = new SyncFLJob();
   syncFLJob.getModel();
   ```

2. LeNet image classification task

   ```java
   // set parameters
   String flName = "lenet";     // Set this parameter to `albert` for sentiment classification tasks or `lenet` for the LeNet scenario.
   String trainModelPath = "SyncFLClient/lenet_train.mindir0.ms";                      // Absolute path
   String inferModelPath = "SyncFLClient/lenet_train.mindir0.ms";                      // Absolute path
   String ip = "10.*.*.*";
   int port = 6668
   boolean useHttps = false;
   boolean useSSL = false;
   boolean useElb = false;
   int serverNum = 1;

   FLParameter flParameter = FLParameter.getInstance();
   flParameter.setFlName(flName);
   flParameter.setTrainModelPath(trainModelPath);
   flParameter.setInferModelPath(inferModelPath);
   flParameter.setIp(ip);
   flParameter.setPort(port);
   flParameter.setUseHttps(useHttps);
   flParameter.setUseSSL(useSSL);
   flParameter.setUseElb(useElb);
   flParameter.setServerNum(serverNum);

   // getModel
   SyncFLJob syncFLJob = new SyncFLJob();
   syncFLJob.getModel();
   ```
