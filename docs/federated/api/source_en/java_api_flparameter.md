# FLParameter

<!-- TOC -->

- [FLParameter](#flparameter)
    - [Public Member Functions](#public-member-functions)
    - [getInstance](#getinstance)
    - [getDomainName](#getdomainname)
    - [setDomainName](#setdomainname)
    - [getCertPath](#getcertpath)
    - [setCertPath](#setcertpath)
    - [getTrainDataset](#gettraindataset)
    - [setTrainDataset](#settraindataset)
    - [getVocabFile](#getvocabfile)
    - [setVocabFile](#setvocabfile)
    - [getIdsFile](#getidsfile)
    - [setIdsFile](#setidsfile)
    - [getTestDataset](#gettestdataset)
    - [setTestDataset](#settestdataset)
    - [getFlName](#getflname)
    - [setFlName](#setflname)
    - [getTrainModelPath](#gettrainmodelpath)
    - [setTrainModelPath](#settrainmodelpath)
    - [getInferModelPath](#getinfermodelpath)
    - [setInferModelPath](#setinfermodelpath)
    - [isUseSSL](#isusessl)
    - [setUseSSL](#setusessl)
    - [getTimeOut](#gettimeout)
    - [setTimeOut](#settimeout)
    - [getSleepTime](#getsleeptime)
    - [setSleepTime](#setsleeptime)
    - [isUseElb](#isuseelb)
    - [setUseElb](#setuseelb)
    - [getServerNum](#getservernum)
    - [setServerNum](#setservernum)
    - [getClientID](#getclientid)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/federated/api/source_en/java_api_flparameter.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

```java
import com.mindspore.flclient.FLParameter
```

FLParameter is used to define parameters related to federated learning.

## Public Member Functions

| **Function**                                         |
| ---------------------------------------------------- |
| public static synchronized FLParameter getInstance() |
| public String getDomainName()                        |
| public void setDomainName(String domainName)         |
| public String getCertPath()                          |
| public void setCertPath(String certPath)             |
| public String getTrainDataset()                      |
| public void setTrainDataset(String trainDataset)     |
| public String getVocabFile()                         |
| public void setVocabFile(String vocabFile)           |
| public String getIdsFile()                           |
| public void setIdsFile(String idsFile)               |
| public String getTestDataset()                       |
| public void setTestDataset(String testDataset)       |
| public String getFlName()                            |
| public void setFlName(String flName)                 |
| public String getTrainModelPath()                    |
| public void setTrainModelPath(String trainModelPath) |
| public String getInferModelPath()                    |
| public void setInferModelPath(String inferModelPath) |
| public boolean isUseSSL()                            |
| public void setUseSSL(boolean useSSL)                |
| public int getTimeOut()                              |
| public void setTimeOut(int timeOut)                  |
| public int getSleepTime()                            |
| public void setSleepTime(int sleepTime)              |
| public boolean isUseElb()                            |
| public void setUseElb(boolean useElb)                |
| public int getServerNum()                            |
| public void setServerNum(int serverNum)              |
| public String getClientID()                          |

## getInstance

```java
public static synchronized FLParameter getInstance()
```

Obtains a single FLParameter instance.

- Return value

    Single object of the FLParameter type.

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

## getCertPath

```java
public String getCertPath()
```

Obtains the certificate path set by a user.

- Return value

    Certificate path of the string type.

## setCertPath

```java
public void setCertPath(String certPath)
```

Sets the certificate path.

- Parameter
    - `certPath`: certificate path.

## getTrainDataset

```java
public String getTrainDataset()
```

Obtains the training dataset path set by a user.

- Return value

    Training dataset path of the string type.

## setTrainDataset

```java
public void setTrainDataset(String trainDataset)
```

Sets the training dataset path.

- Parameter
    - `trainDataset`: training dataset path.

## getVocabFile

```java
public String getVocabFile()
```

Obtains the path of the dictionary file for data preprocessing, which is set by a user.

- Return value

    Path of the dictionary file for preprocessing data, which is of the string type.

## setVocabFile

```java
public void setVocabFile(String vocabFile)
```

Sets the path of the dictionary file for data preprocessing.

- Parameter
    - `vocabFile`: path of the dictionary file for data preprocessing.

## getIdsFile

```java
public String getIdsFile()
```

Obtains the path of the mapping ID file of a dictionary set by a user.

- Return value

    Path of the mapping ID file of a dictionary of the string type.

## setIdsFile

```java
public void setIdsFile(String idsFile)
```

Sets the path of the mapping ID file of a dictionary.

- Parameter

    - `idsFile`: path of the mapping ID file of a dictionary.

## getTestDataset

```java
public String getTestDataset()
```

Obtains the path of the test dataset set by a user.

- Return value

    Path of the test dataset of the string type.

## setTestDataset

```java
public void setTestDataset(String testDataset)
```

Sets the path of the test dataset.

- Parameter
    - `testDataset`: path of the test dataset.

## getFlName

```java
public String getFlName()
```

Obtains the model name set by a user.

- Return value

    Name of a model of the string type.

## setFlName

```java
public void setFlName(String flName)
```

Sets the model name.

- Parameter
    - `flName`: model name.

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

## isUseSSL

```java
public boolean isUseSSL()
```

Determines whether SSL certificate authentication is performed for device-cloud communication.

- Return value

    The value is of the Boolean type. The value true indicates that SSL certificate authentication is performed, and the value false indicates that SSL certificate authentication is not performed.

## setUseSSL

```java
public void setUseSSL(boolean useSSL)
```

Determines whether to perform SSL certificate authentication (which applies only to the HTTPS communication scenario) for device-cloud communication.

- Parameter
    - `useSSL`: determines whether to perform SSL certificate authentication for device-cloud communication.

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

## getClientID

```java
public String getClientID()
```

Obtains the unique client ID which is automatically generated in the program before starting the federated learning task.

- Return value

    Unique ID of the client, which is of the string type.
