# FLParameter

<!-- TOC -->

- [FLParameter](#flparameter)
    - [公有成员函数](#公有成员函数)
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

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/federated/api/source_zh_cn/java_api_flparameter.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

```java
import com.mindspore.flclient.FLParameter
```

FLParameter定义联邦学习相关参数，供用户进行设置。

## 公有成员函数

| **function**                                         |
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

获取FLParameter单例。

- 返回值

    FLParameter类型的单例对象。

## getDomainName

```java
public String getDomainName()
```

获取用户设置的域名domainName。

- 返回值

    String类型的域名。

## setDomainName

```java
public void setDomainName(String domainName)
```

用于设置端云通信url，目前，可支持https和http通信，对应格式分别为：https://......、http://......，当`useElb`设置为true时，格式必须为：https://127.0.0.0:6666 或者http://127.0.0.0:6666 ，其中`127.0.0.0`对应提供云侧服务的机器ip（即云侧参数`--scheduler_ip`），`6666`对应云侧参数`--fl_server_port`。

- 参数

    - `domainName`: 域名。

## getCertPath

```java
public String getCertPath()
```

获取用户设置的证书路径certPath。

- 返回值

    String类型的证书路径certPath。

## setCertPath

```java
public void setCertPath(String certPath)
```

用于设置证书路径certPath。

- 参数
    - `certPath`: 证书路径。

## getTrainDataset

```java
public String getTrainDataset()
```

获取用户设置的训练数据集路径trainDataset。

- 返回值

    String类型的训练数据集路径trainDataset。

## setTrainDataset

```java
public void setTrainDataset(String trainDataset)
```

用于设置训练数据集路径trainDataset。

- 参数
    - `trainDataset`: 训练数据集路径。

## getVocabFile

```java
public String getVocabFile()
```

用于获取用户设置的数据预处理的词典文件路径vocabFile。

- 返回值

    String类型的数据预处理的词典文件路径vocabFile。

## setVocabFile

```java
public void setVocabFile(String vocabFile)
```

设置数据预处理的词典文件路径VocabFile。

- 参数
    - `vocabFile`: 数据预处理的词典文件路径。

## getIdsFile

```java
public String getIdsFile()
```

用于获取用户设置的词典的映射id文件路径idsFile。

- 返回值

    String类型的词典的映射id文件路径idsFile。

## setIdsFile

```java
public void setIdsFile(String idsFile)
```

设置词典的映射id文件路径idsFile。

- 参数

    - `idsFile`: 词典的映射id文件路径。

## getTestDataset

```java
public String getTestDataset()
```

用于获取用户设置的测试数据集路径testDataset。

- 返回值

    String类型的测试数据集路径testDataset。

## setTestDataset

```java
public void setTestDataset(String testDataset)
```

设置测试数据集路径testDataset。

- 参数
    - `testDataset`: 测试数据集路径。

## getFlName

```java
public String getFlName()
```

用于获取用户设置的模型名称flName。

- 返回值

    String类型的模型名称flName。

## setFlName

```java
public void setFlName(String flName)
```

设置模型名称flName。

- 参数
    - `flName`: 模型名称。

## getTrainModelPath

```java
public String getTrainModelPath()
```

用于获取用户设置的训练模型路径trainModelPath。

- 返回值

    String类型的训练模型路径trainModelPath。

## setTrainModelPath

```java
public void setTrainModelPath(String trainModelPath)
```

设置训练模型路径trainModelPath。

- 参数
    - `trainModelPath`: 训练模型路径。

## getInferModelPath

```java
public String getInferModelPath()
```

用于获取用户设置的推理模型路径inferModelPath。

- 返回值

    String类型的推理模型路径inferModelPath。

## setInferModelPath

```java
public void setInferModelPath(String inferModelPath)
```

设置推理模型路径inferModelPath。

- 参数
    - `inferModelPath`: 推理模型路径。

## isUseSSL

```java
public boolean isUseSSL()
```

端云通信是否进行ssl证书认证。

- 返回值

    boolean类型，true代表进行ssl证书认证， false代表不进行ssl证书认证。

## setUseSSL

```java
public void setUseSSL(boolean useSSL)
```

用于设置端云通信是否进行ssl证书认证，ssl证书认证只用于https通信场景。

- 参数
    - `useSSL`: 端云通信是否进行ssl证书认证。

## getTimeOut

```java
public int getTimeOut()
```

用于获取用户设置的端侧通信的超时时间timeOut。

- 返回值

    int类型的端侧通信的超时时间timeOut。

## setTimeOut

```java
public void setTimeOut(int timeOut)
```

用于设置端侧通信的超时时间timeOut。

- 参数
    - `timeOut`: 端侧通信的超时时间。

## getSleepTime

```java
public int getSleepTime()
```

用于获取用户设置的重复请求的等待时间sleepTime。

- 返回值

    int类型的重复请求的等待时间sleepTime。

## setSleepTime

```java
public void setSleepTime(int sleepTime)
```

用于设置重复请求的等待时间sleepTime。

- 参数
    - `sleepTime`: 重复请求的等待时间。

## isUseElb

```java
public boolean isUseElb()
```

是否模拟弹性负载均衡，即客户端将请求随机发给一定范围内的server地址。

- 返回值

    boolean类型，true代表客户端会将请求随机发给一定范围内的server地址， false客户端的请求会发给固定的server地址。

## setUseElb

```java
public void setUseElb(boolean useElb)
```

用于设置是否模拟弹性负载均衡，即客户端将请求随机发给一定范围内的server地址。

- 参数
    - `useElb`: 是否模拟弹性负载均衡，默认为false。

## getServerNum

```java
public int getServerNum()
```

用于获取用户设置的模拟弹性负载均衡时可发送请求的server数量。

- 返回值

    int类型的模拟弹性负载均衡时可发送请求的server数量。

## setServerNum

```java
public void setServerNum(int serverNum)
```

用于设置模拟弹性负载均衡时可发送请求的server数量。

- 参数
    - `serverNum`: 模拟弹性负载均衡时可发送请求的server数量，默认为1。

## getClientID

```java
public String getClientID()
```

启动联邦学习任务前，在程序中会自动生成一个唯一标识客户端的ID，该方法用于获取该ID。

- 返回值

    String类型的唯一标识客户端的ID。
