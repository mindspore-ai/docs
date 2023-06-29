# FLParameter

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/federated/api/source_zh_cn/java_api_flparameter.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

```java
import com.mindspore.flclient.FLParameter
```

FLParameter定义联邦学习相关参数，供用户进行设置。

## 公有成员函数

| **function**                                         |
| ---------------------------------------------------- |
| public static synchronized FLParameter getInstance() |
| public String getHostName()                          |
| public void setHostName(String hostName)             |
| public String getCertPath()                          |
| public void setCertPath(String certPath)             |
| public boolean isUseHttps()                          |
| public void setUseHttps(boolean useHttps)            |
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
| public String getIp()                                |
| public void setIp(String ip)                         |
| public boolean isUseSSL()                            |
| public void setUseSSL(boolean useSSL)                |
| public int getPort()                                 |
| public void setPort(int port)                        |
| public int getTimeOut()                              |
| public void setTimeOut(int timeOut)                  |
| public int getSleepTime()                            |
| public void setSleepTime(int sleepTime)              |
| public boolean isUseElb()                            |
| public void setUseElb(boolean useElb)                |
| public int getServerNum()                            |
| public void setServerNum(int serverNum)              |
| public String getClientID()                          |
| public void setClientID(String clientID)             |

## getInstance

```java
public static synchronized FLParameter getInstance()
```

获取FLParameter单例。

- 返回值

    FLParameter类型的单例对象。

## getHostName

```java
public String getHostName()
```

获取用户设置的域名hostName。

- 返回值

    String类型的域名。

## setHostName

```java
public void setHostName(String hostName)
```

用于设置域名hostName。

- 参数

    - `hostName`: 域名。

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

## isUseHttps

```java
public boolean isUseHttps()
```

端云通信是否采用https通信方式。

- 返回值

    boolean类型，true代表进行https通信， false代表进行http通信，默认值为false，目前云侧暂不支持https通信。

## setUseHttps

```java
 public void setUseHttps(boolean useHttps)
```

用于设置端云通信是否采用https通信方式。

- 参数
    - `useHttps`: 是否采用https通信方式。

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

## getIp

```java
public String getIp()
```

用于获取用户设置的端云通信的ip地址。

- 返回值

    String类型的ip地址。

## setIp

```java
public void setIp(String ip)
```

设置端云通信的ip地址。

- 参数
    - `ip`: 端云通信的ip地址。

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

## getPort

```java
public int getPort()
```

用于获取用户设置的端云通信的端口号port。

- 返回值

    int类型的端云通信的端口号port。

## setPort

```java
public void setPort(int port)
```

用于设置端云通信的端口号port。

- 参数
    - `port`: 端云通信的端口号。

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

用于获取用户设置的唯一标识客户端的ID。

- 返回值

    String类型的唯一标识客户端的ID。

## setClientID

```java
public void setClientID(String clientID)
```

用于设置唯一标识客户端的ID。

- 参数
    - `clientID`: 唯一标识客户端的ID。