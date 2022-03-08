# FLParameter

<a href="https://gitee.com/mindspore/docs/blob/master/docs/federated/api/source_en/java_api_flparameter.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

```java
import com.mindspore.flclient.FLParameter
```

FLParameter is used to define parameters related to federated learning.

## Public Member Functions

| **function**                                                 |
| ------------------------------------------------------------ |
| public static synchronized FLParameter getInstance()         |
| public String getDeployEnv()                                 |
| public void setDeployEnv(String env)                         |
| public String getDomainName()                                |
| public void setDomainName(String domainName)                 |
| public String getClientID()                                  |
| public void setClientID(String clientID)                     |
| public String getCertPath()                                  |
| public void setCertPath(String certPath)                     |
| public SSLSocketFactory getSslSocketFactory()                |
| public void setSslSocketFactory(SSLSocketFactory sslSocketFactory) |
| public X509TrustManager getX509TrustManager(                 |
| public void setX509TrustManager(X509TrustManager x509TrustManager) |
| public IFLJobResultCallback getIflJobResultCallback()        |
| public void setIflJobResultCallback(IFLJobResultCallback iflJobResultCallback) |
| public String getFlName()                                    |
| public void setFlName(String flName)                         |
| public String getTrainModelPath()                            |
| public void setTrainModelPath(String trainModelPath)         |
| public String getInferModelPath()                            |
| public void setInferModelPath(String inferModelPath)         |
| public String getSslProtocol()                               |
| public void setSslProtocol(String sslProtocol)               |
| public int getTimeOut()                                      |
| public void setTimeOut(int timeOut)                          |
| public int getSleepTime()                                    |
| public void setSleepTime(int sleepTime)                      |
| public boolean isUseElb()                                    |
| public void setUseElb(boolean useElb)                        |
| public int getServerNum()                                    |
| public void setServerNum(int serverNum)                      |
| public boolean isPkiVerify()                                 |
| public void setPkiVerify(boolean ifPkiVerify)                |
| public String getEquipCrlPath()                              |
| public void setEquipCrlPath(String certPath)                 |
| public long getValidInterval()                               |
| public void setValidInterval(long validInterval)             |
| public int getThreadNum()                                    |
| public void setThreadNum(int threadNum)                      |
| public int getCpuBindMode(                                   |
| public void setCpuBindMode(BindMode cpuBindMode)             |
| public List<String/> getHybridWeightName(RunType runType)    |
| public void setHybridWeightName(List<String/> hybridWeightName, RunType runType) |
| public Map<RunType, List<String>> getDataMap()               |
| public void setDataMap(Map<RunType, List<String/>/> dataMap) |
| public ServerMod getServerMod()                              |
| public void setServerMod(ServerMod serverMod)                |
| public int getBatchSize()                                    |
| public void setBatchSize(int batchSize)                      |

## getInstance

```java
public static synchronized FLParameter getInstance()
```

Obtains a single FLParameter instance.

- Return value

    Single object of the FLParameter type.

## getDeployEnv

```java
public String getDeployEnv()
```

Obtains the deployment environment for federated learning set by users.

- Return value

    The deployment environment for federated learning of the string type.

## setDeployEnv

```java
public void setDeployEnv(String env)
```

Used to set the deployment environment for federated learning, a whitelist is set, currently only "x86", "android" are supported.

- Parameter

    - `env`: the deployment environment for federated learning.

## getDomainName

```java
public String getDomainName()
```

Obtains the domain name set by a user.

- Return value

    Domain name of the string type.

## setDomainName

```java
public void setDomainName(String domainName)
```

Used to set the url for device-cloud communication. Currently, https and http communication are supported, the corresponding formats are like: https://......, http://......, and when `useElb` is set to true, the format must be: https://127.0.0.0 : 6666 or http://127.0.0.0 : 6666 , where `127.0.0.0` corresponds to the ip of the machine providing cloud-side services (corresponding to the cloud-side parameter `--scheduler_ip`), and `6666` corresponds to the cloud-side parameter `--fl_server_port`.

- Parameter

    - `domainName`: domain name.

## getClientID

```java
public String getClientID()
```

The method `getClientID` is used to obtain the unique ID of the client, the ID also can be used to generate related certificates in the device-cloud security authentication scenario.

- Return value

    Unique ID of the client, which is of the string type.

## setClientID

```java
public void setClientID(String clientID)
```

Each time the federated learning task is started, a unique client ID will be automatically generated in the program, if the user needs to set the clientID by himself,  he can set the ID by calling the method `setClientID` before starting the federated learning training task.

- Parameter

    - `clientID`: unique ID of the client.

## getCertPath

```java
public String getCertPath()
```

Obtains the self-signed root certificate path used for device-cloud HTTPS communication.

- Return value

    The self-signed root certificate pat of the string type.

## setCertPath

```java
public void setCertPath(String certPath)
```

Sets the self-signed root certificate path used for device-cloud HTTPS communication. When the deployment environment is "x86" and the device-cloud uses a self-signed certificate for HTTPS communication authentication, this parameter needs to be set. The certificate must be consistent with the CA root certificate used to generate the cloud-side self-signed certificate to pass the verification. This parameter is used for non-Android scenarios.

- Parameter
    - `certPath`: the self-signed root certificate path used for device-cloud HTTPS communication.

## getSslSocketFactory

```java
public SSLSocketFactory getSslSocketFactory()
```

Obtains the ssl certificate authentication library `sslSocketFactory` set by the user.

- Return value

    The ssl certificate authentication library `sslSocketFactory` , which is of the SSLSocketFactory type.

## setSslSocketFactory

```java
public void setSslSocketFactory(SSLSocketFactory sslSocketFactory)
```

Used to set the ssl certificate authentication library `sslSocketFactory`.

- Parameter
    - `sslSocketFactory`: the ssl certificate authentication library.

## getX509TrustManager

```java
public X509TrustManager getX509TrustManager()
```

Obtains the ssl certificate authentication manager `x509TrustManager` set by the user.

- Return value

    the ssl certificate authentication manager `x509TrustManager`, which is of the X509TrustManager type.

## setX509TrustManager

```java
public void setX509TrustManager(X509TrustManager x509TrustManager)
```

Used to set the ssl certificate authentication manager `x509TrustManager`.

- Parameter
    - `x509TrustManager`: the ssl certificate authentication manager.

## getIflJobResultCallback

```java
public IFLJobResultCallback getIflJobResultCallback()
```

Obtains the federated learning callback function object `iflJobResultCallback` set by the user.

- Return value

    The federated learning callback function object `iflJobResultCallback`, which is of the IFLJobResultCallback type.

## setIflJobResultCallback

```java
public void setIflJobResultCallback(IFLJobResultCallback iflJobResultCallback)
```

Used to set the federated learning callback function object `iflJobResultCallback`, the user can implement the specific method of the interface class [IFLJobResultCallback.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/java/fl_client/src/main/java/com/mindspore/flclient/IFLJobResultCallback.java) in the project according to the needs of the actual scene, and set it as a callback function object in the federated learning task.

- Parameter
    - `iflJobResultCallback`: the federated learning callback function object.

## getFlName

```java
public String getFlName()
```

Obtains the package path of model script set by a user.

- Return value

    Name of the package path of model script of the string type.

## setFlName

```java
public void setFlName(String flName)
```

Sets the package path of model script . We provide two types of model scripts for your reference ([Supervised sentiment classification task](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert), [Lenet image classification task](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet)). For supervised sentiment classification tasks, this parameter can be set to the package path of the provided script file [AlBertClient.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert/AlbertClient.java), like as `com.mindspore.flclient.demo.albert.AlbertClient`; for Lenet image classification tasks, this parameter can be set to the package path of the provided script file [LenetClient.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet/LenetClient.java), like as `com.mindspore.flclient.demo.lenet.LenetClient`. At the same time, users can refer to these two types of model scripts, define the model script by themselves, and then set the parameter to the package path of the customized model file ModelClient.java (which needs to inherit from the class [Client.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/java/fl_client/src/main/java/com/mindspore/flclient/model/Client.java)).

- Parameter
    - `flName`: package path of model script.

## getTrainModelPath

```java
public String getTrainModelPath()
```

Obtains the path of the training model set by a user.

- Return value

    Path of the training model of the string type.

## setTrainModelPath

```java
public void setTrainModelPath(String trainModelPath)
```

Sets the path of the training model.

- Parameter
    - `trainModelPath`: training model path.

## getInferModelPath

```java
public String getInferModelPath()
```

Obtains the path of the inference model set by a user.

- Return value

    Path of the inference model of the string type.

## setInferModelPath

```java
public void setInferModelPath(String inferModelPath)
```

Sets the path of the inference model.

- Parameter
    - `inferModelPath`: path of the inference model.

## getSslProtocol

```java
public String getSslProtocol()
```

Obtains the TLS protocol version used by the device-cloud HTTPS communication.

- Return value

    The TLS protocol version used by the device-cloud HTTPS communication of the string type.

## setSslProtocol

```java
public void setSslProtocol(String sslProtocol)
```

Used to set the TLS protocol version used by the device-cloud HTTPS communication, a whitelist is set, and currently only "TLSv1.3" or "TLSv1.2" is supported. Only need to set it up in the HTTPS communication scenario.

- Parameter
    - `sslProtocol`: the TLS protocol version used by the device-cloud HTTPS communication.

## getTimeOut

```java
public int getTimeOut()
```

Obtains the timeout interval set by a user for device-side communication.

- Return value

    Timeout interval for communication on the device, which is an integer.

## setTimeOut

```java
public void setTimeOut(int timeOut)
```

Sets the timeout interval for communication on the device.

- Parameter
    - `timeOut`: timeout interval for communication on the device.

## getSleepTime

```java
public int getSleepTime()
```

Obtains the waiting time of repeated requests set by a user.

- Return value

    Waiting time of repeated requests, which is an integer.

## setSleepTime

```java
public void setSleepTime(int sleepTime)
```

Sets the waiting time of repeated requests.

- Parameter
    - `sleepTime`: waiting time for repeated requests.

## isUseElb

```java
public boolean isUseElb()
```

Determines whether the elastic load balancing is simulated, that is, whether a client randomly sends requests to a server address within a specified range.

- Return value

    The value is of the Boolean type. The value true indicates that the client sends requests to a random server address within a specified range. The value false indicates that the client sends a request to a fixed server address.

## setUseElb

```java
public void setUseElb(boolean useElb)
```

Determines whether to simulate the elastic load balancing, that is, whether a client randomly sends a request to a server address within a specified range.

- Parameter
    - `useElb`: determines whether to simulate the elastic load balancing. The default value is false.

## getServerNum

```java
public int getServerNum()
```

Obtains the number of servers that can send requests when simulating the elastic load balancing.

- Return value

    Number of servers that can send requests during elastic load balancing simulation, which is an integer.

## setServerNum

```java
public void setServerNum(int serverNum)
```

Sets the number of servers that can send requests during elastic load balancing simulation.

- Parameter
    - `serverNum`: number of servers that can send requests during elastic load balancing simulation. The default value is 1.

## isPkiVerify

```java
public boolean isPkiVerify()
```

Whether to perform device-cloud security authentication.

- Return value

    The value is of the Boolean type. The value true indicates that device-cloud security authentication is performed, and the value false indicates that device-cloud security authentication is not performed.

## setPkiVerify

```java
public void setPkiVerify(boolean pkiVerify)
```

Determines whether to perform device-cloud security authentication.

- Parameter

    - `pkiVerify`: whether to perform device-cloud security authentication.

## getEquipCrlPath

```java
public String getEquipCrlPath()
```

Obtains the CRL certification path `equipCrlPath` of the device certificate set by the user. This parameter is used in the Android environment.

- Return value

    The certification path of the string type.

## setEquipCrlPath

```java
public void setEquipCrlPath(String certPath)
```

Used to set the CRL certification path of the device certificate. It is used to verify whether the digital certificate is revoked. This parameter is used in the Android environment.

- Parameter
    - `certPath`: the certification path.

## getValidInterval

```java
public long getValidInterval()
```

Obtains the valid iteration interval validIterInterval set by the user. This parameter is used in the Android environment.

- Return value

    The valid iteration interval validIterInterval of the long type.

## setValidInterval

```java
public void setValidInterval(long validInterval)
```

Used to set the valid iteration interval validIterInterval. The recommended duration is the duration of one training epoch between the device-cloud(unit: milliseconds). It is used to prevent replay attacks. This parameter is used in the Android environment.

- Parameter
    - `validInterval`: the valid iteration interval validIterInterval.

## getThreadNum

```java
public int getThreadNum()
```

Obtains the number of threads used in federated learning training and inference. The default value is 1.

- Return value

    The number of threads used in federated learning training and inference, which is of the int type.

## setThreadNum

```java
public void setThreadNum(int threadNum)
```

Used to set the number of threads used in federated learning training and inference.

- Parameter
    - `threadNum`: the number of threads used in federated learning training and inference.

## getCpuBindMode

```java
public int getCpuBindMode()
```

Obtains the cpu core that threads need to bind during federated learning training and inference.

- Return value

    Convert the enumerated type of cpu core to int type and return.

## setCpuBindMode

```java
public void setCpuBindMode(BindMode cpuBindMode)
```

Used to set the cpu core that threads need to bind during federated learning training and inference.

- Parameter
    - `cpuBindMode`: it is the enumeration type `BindMode`, where BindMode.NOT_BINDING_CORE represents the unbound core, which is automatically assigned by the system, BindMode.BIND_LARGE_CORE represents the bound large core, and BindMode.BIND_MIDDLE_CORE represents the bound middle core.

## getHybridWeightName

```java
public List<String> getHybridWeightName(RunType runType)
```

Used in hybrid training mode. Get the training weight name and inference weight name set by the user.

- Parameter

- `runType`: RunType enumeration type, only supports to be set to RunType.TRAINMODE (representing the training weight name) , RunType.INFERMODE (representing the inference weight name).

- Return value

    A list of corresponding weight names according to the parameter runType, which is of the List<String> type.

## setHybridWeightName

```java
public void setHybridWeightName(List<String> hybridWeightName, RunType runType)
```

Due to the hybrid training mode, part of the weights delivered by the server is imported into the training model, and part is imported into the inference model, but the framework itself cannot judge it, so the user needs to set the relevant training weight name and inference weight name by himself. This method is provided for the user to set.

- Parameter
    - `hybridWeightName`: a list of weight names of the List<String> type.
    - `runType`: RunType enumeration type, only supports setting to RunType.TRAINMODE (representing setting training weight name), RunType.INFERMODE (representing setting reasoning weight name).

## getDataMap

```java
public Map<RunType, List<String>> getDataMap()
```

Obtains the federated learning dataset set by the user.

- Return value

    the federated learning dataset set of the Map<RunType, List<String>> type.

## setDataMap

```java
public void setDataMap(Map<RunType, List<String>> dataMap)
```

Used to set the federated learning dataset set by the user.

- Parameter
    - `dataMap`:  the dataset of Map<RunType, List<String>> type, the key in the map is the RunType enumeration type, the value is the corresponding dataset list, when the key is RunType.TRAINMODE, the corresponding value is the training-related dataset list, when the key  is RunType.EVALMODE, it means that the corresponding value is a list of verification-related datasets, and when the key is RunType.INFERMODE, it means that the corresponding value is a list of inference-related datasets.

## getServerMod

```java
public ServerMod getServerMod()
```

 Obtains the federated learning training mode.

- Return value

    The federated learning training mode of ServerMod enumeration type.

## setServerMod

```java
public void setServerMod(ServerMod serverMod)
```

Used to set the federated learning training mode.

- Parameter
    - `serverMod`:  the federated learning training mode of ServerMod enumeration type, where ServerMod.FEDERATED_LEARNING represents the normal federated learning mode (training and inference use the same model) ServerMod.HYBRID_TRAINING represents the hybrid learning mode (training and inference use different models, and the server side also includes training process).

## getBatchSize

```java
public int getBatchSize()
```

Obtains the number of single-step training samples used in federated learning training and inference, that is, batch size.

- Return value

    BatchSize, the number of single-step training samples of int type.

## setBatchSize

```java
public void setBatchSize(int batchSize)
```

Used to set the number of single-step training samples used in federated learning training and inference, that is, batch size. It needs to be consistent with the batch size of the input data of the model.

- Parameter
    - `batchSize`:  the number of single-step training samples of int type.

## getInputShape

```java
public int[][] getInputShape()
```

Used to obtain the real input shape of the model set by the user When the user uses a dynamic input model.

- Return value

    The input shape of the model of int[\][\] type.

## setInputShape

```java
public void setInputShape(int[][] inputShape)
```

This parameter is used to set the real input shape of the model, which is only needed when the user uses a dynamic input model. the `inputShape` is a two-dimensional array of type int, which supports multiple inputs. Refer to the following usage examples:

When the dynamic input model contains only one input, assuming that the input dimension in the model is displayed as [-1, 24], the -1 dimension is a variable dimension, and the user needs to indicate its true dimension when using the model:

```java
int[][] inputShape = {{32, 24}}      // Specify that the true dimension corresponding to the -1 dimension is 32
FLParameter flParameter = FLParameter.getInstance();
flParameter.setInputShape(inputShape)
```

When the dynamic input model only contains multiple inputs, assume that the dimension of input 1 in the model is displayed as [-1, 24], and the dimension of input 2 is displayed as [-1, -1], then the -1 dimension is a variable dimension , when users use the model, they need to indicate their true dimensions:

```java
int[][] inputShape = {{32, 24}, {32, 96}}      // Specify that the real dimension of input 1 is {32, 24}, and the real dimension of input 2 is {32, 96}
FLParameter flParameter = FLParameter.getInstance();
flParameter.setInputShape(inputShape)
```

Note that there is usually a dimension in the input shape of the model that represents the batch size, and this value should be consistent with the value set by the `FLParameter.setBatchSize()` interface.

- Parameter
    - `inputShape`: The input shape of the model of int[\][\] type.

