# FLParameter

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/federated/docs/source_zh_cn/java_api_flparameter.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

```java
import com.mindspore.flclient.FLParameter
```

FLParameter定义联邦学习相关参数，供用户进行设置。

## 公有成员函数

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
| public int getCpuBindMode()                                  |
| public void setCpuBindMode(BindMode cpuBindMode)             |
| public List<String/> getHybridWeightName(RunType runType)    |
| public void setHybridWeightName(List<String/> hybridWeightName, RunType runType) |
| public Map<RunType, List<String/>/> getDataMap()             |
| public void setDataMap(Map<RunType, List<String/>/> dataMap) |
| public ServerMod getServerMod()                              |
| public void setServerMod(ServerMod serverMod)                |
| public int getBatchSize()                                    |
| public void setBatchSize(int batchSize)                      |

## getInstance

```java
public static synchronized FLParameter getInstance()
```

获取FLParameter单例。

- 返回值

    FLParameter类型的单例对象。

## getDeployEnv

```java
public String getDeployEnv()
```

获取用户设置联邦学习的部署环境。

- 返回值

    String类型的联邦学习的部署环境。

## setDeployEnv

```java
public void setDeployEnv(String env)
```

用于设置联邦学习的部署环境， 设置了白名单，目前只支持"x86", "android"。

- 参数

    - `env`: 联邦学习的部署环境。

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

## getClientID

```java
public String getClientID()
```

每次联邦学习任务启动前会自动生成一个唯一标识客户端的clientID（若用户需要自行设置clientID，可在启动联邦学习训练任务前使用setClientID进行设置），该方法用于获取该ID，可用于端云安全认证场景中生成相关证书。

- 返回值

    String类型的唯一标识客户端的clientID。

## setClientID

```java
public void setClientID(String clientID)
```

用于用户设置唯一标识客户端的clientID。

- 参数

    - `clientID`: 唯一标识客户端的clientID。

## getCertPath

```java
public String getCertPath()
```

获取用户设置的端云https通信所使用的自签名根证书路径certPath。

- 返回值

    String类型的自签名根证书路径certPath。

## setCertPath

```java
public void setCertPath(String certPath)
```

用于设置端云HTTPS通信所使用的自签名根证书路径certPath。当部署环境为"x86"，且端云采用自签名证书进行https通信认证时，需要设置该参数，该证书需与生成云侧自签名证书所使用的CA根证书一致才能验证通过，此参数用于非Android场景。

- 参数
    - `certPath`: 端云https通信所使用的自签名根证书路径。

## getSslSocketFactory

```java
public SSLSocketFactory getSslSocketFactory()
```

获取用户设置的ssl证书认证库sslSocketFactory。

- 返回值

    SSLSocketFactory类型的ssl证书认证库sslSocketFactory。

## setSslSocketFactory

```java
public void setSslSocketFactory(SSLSocketFactory sslSocketFactory)
```

用于设置ssl证书认证库sslSocketFactory。

- 参数
    - `sslSocketFactory`: ssl证书认证库。

## getX509TrustManager

```java
public X509TrustManager getX509TrustManager()
```

  获取用户设置的ssl证书认证管理器x509TrustManager。

- 返回值

    X509TrustManager类型的ssl证书认证管理器x509TrustManager。

## setX509TrustManager

```java
public void setX509TrustManager(X509TrustManager x509TrustManager)
```

用于设置ssl证书认证管理器x509TrustManager。

- 参数
    - `x509TrustManager`:ssl证书认证管理器。

## getIflJobResultCallback

```java
public IFLJobResultCallback getIflJobResultCallback()
```

  获取用户设置的联邦学习回调函数对象iflJobResultCallback。

- 返回值

    IFLJobResultCallback类型的联邦学习回调函数对象iflJobResultCallback。

## setIflJobResultCallback

```java
public void setIflJobResultCallback(IFLJobResultCallback iflJobResultCallback)
```

用于设置联邦学习回调函数对象iflJobResultCallback，用户可根据实际场景所需，实现工程中接口类[IFLJobResultCallback.java](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/java/java/fl_client/src/main/java/com/mindspore/flclient/IFLJobResultCallback.java)的具体方法后，作为回调函数对象设置到联邦学习任务中。

- 参数
    - `iflJobResultCallback`:联邦学习回调函数。

## getFlName

```java
public String getFlName()
```

用于获取用户设置的模型脚本包路径。

- 返回值

    String类型的模型脚本包路径。

## setFlName

```java
public void setFlName(String flName)
```

设置模型脚本包路径。我们提供了两个类型的模型脚本供大家参考（[有监督情感分类任务](https://gitee.com/mindspore/mindspore/tree/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert)、[Lenet图片分类任务](https://gitee.com/mindspore/mindspore/tree/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet)），对于有监督情感分类任务，该参数可设置为所提供的脚本文件[AlBertClient.java](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/albert/AlbertClient.java) 的包路径`com.mindspore.flclient.demo.albert.AlbertClient`；对于Lenet图片分类任务，该参数可设置为所提供的脚本文件[LenetClient.java](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/quick_start_flclient/src/main/java/com/mindspore/flclient/demo/lenet/LenetClient.java) 的包路径`com.mindspore.flclient.demo.lenet.LenetClient`。同时，用户可参考这两个类型的模型脚本，自定义模型脚本，然后将该参数设置为自定义的模型文件ModelClient.java（需继承于类[Client.java](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/java/java/fl_client/src/main/java/com/mindspore/flclient/model/Client.java)）的包路径即可。

- 参数
    - `flName`: 模型脚本包路径。

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

## getSslProtocol

```java
public String getSslProtocol()
```

用于获取用户设置的端云HTTPS通信所使用的TLS协议版本。

- 返回值

    String类型的端云HTTPS通信所使用的TLS协议版本。

## setSslProtocol

```java
public void setSslProtocol(String sslProtocol)
```

用于设置端云HTTPS通信所使用的TLS协议版本， 设置了白名单，目前只支持"TLSv1.3"或者"TLSv1.2"。只在HTTPS通信场景中使用。

- 参数
    - `sslProtocol`: 端云HTTPS通信所使用的TLS协议版本。

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

## isPkiVerify

```java
public boolean isPkiVerify()
```

是否进行端云认证。

- 返回值

    boolean类型，true代表进行端云认证，false代表不进行端云认证。

## setPkiVerify

```java
public void setPkiVerify(boolean pkiVerify)
```

用于设置是否进行端云认证。

- 参数

    - `pkiVerify`: 是否进行端云认证。

## getEquipCrlPath

```java
public String getEquipCrlPath()
```

获取用户设置的设备证书的CRL证书路径equipCrlPath，此参数用于Android环境。

- 返回值

    String类型的证书路径equipCrlPath。

## setEquipCrlPath

```java
public void setEquipCrlPath(String certPath)
```

用于设置设备证书的CRL证书路径，用于验证数字证书是否被吊销，此参数用于Android环境。

- 参数
    - `certPath`: 证书路径。

## getValidInterval

```java
public long getValidInterval()
```

获取用户设置的有效迭代时间间隔validIterInterval，此参数用于Android环境。

- 返回值

    long类型的有效迭代时间间隔validIterInterval。

## setValidInterval

```java
public void setValidInterval(long validInterval)
```

用于设置有效迭代时间间隔validIterInterval，建议时长为端云间一个训练epoch的时长（单位：毫秒），用于防范重放攻击，此参数用于Android环境。

- 参数
    - `validInterval`: 有效迭代时间间隔。

## getThreadNum

```java
public int getThreadNum()
```

获取联邦学习训练和推理时使用的线程数，默认值为1。

- 返回值

    int类型的线程数threadNum。

## setThreadNum

```java
public void setThreadNum(int threadNum)
```

设置联邦学习训练和推理时使用的线程数。

- 参数
    - `threadNum`: 线程数。

## getCpuBindMode

```java
public int getCpuBindMode()
```

获取联邦学习训练和推理时线程所需绑定的cpu内核。

- 返回值

    将枚举类型的cpu内核cpuBindMode转换为int型返回。

## setCpuBindMode

```java
public void setCpuBindMode(BindMode cpuBindMode)
```

设置联邦学习训练和推理时线程所需绑定的cpu内核。

- 参数
    - `cpuBindMode`: BindMode枚举类型，其中BindMode.NOT_BINDING_CORE代表不绑定内核，由系统自动分配，BindMode.BIND_LARGE_CORE代表绑定大核，BindMode.BIND_MIDDLE_CORE代表绑定中核。

## getHybridWeightName

```java
public List<String> getHybridWeightName(RunType runType)
```

混合学习模式时使用。获取用户设置的训练权重名和推理权重名。

- 参数

    - `runType`: RunType枚举类型，只支持设置为RunType.TRAINMODE（代表获取训练权重名）、RunType.INFERMODE（代表获取推理权重名）。

- 返回值

    List<String> 类型，根据参数runType返回相应的权重名列表。

## setHybridWeightName

```java
public void setHybridWeightName(List<String> hybridWeightName, RunType runType)
```

由于混合学习模式时，云侧下发的权重，一部分导入到训练模型，一部分导入到推理模型，但框架本身无法判断，需要用户自行设置相关训练权重名和推理权重名。该方法提供给用户进行设置。

- 参数
    - `hybridWeightName`: List<String> 类型的权重名列表。
    - `runType`: RunType枚举类型，只支持设置为RunType.TRAINMODE（代表设置训练权重名）、RunType.INFERMODE（代表设置推理权重名）。

## getDataMap

```java
public Map<RunType, List<String>> getDataMap()
```

获取用户设置的联邦学习数据集。

- 返回值

    Map<RunType, List<String>>类型的数据集。

## setDataMap

```java
public void setDataMap(Map<RunType, List<String>> dataMap)
```

设置联邦学习数据集。

- 参数
    - `dataMap`: Map<RunType, List<String>>类型的数据集，map中key为RunType枚举类型，value为对应的数据集列表，key为RunType.TRAINMODE时代表对应的value为训练相关的数据集列表，key为RunType.EVALMODE时代表对应的value为验证相关的数据集列表， key为RunType.INFERMODE时代表对应的value为推理相关的数据集列表。

## getServerMod

```java
public ServerMod getServerMod()
```

获取联邦学习训练模式。

- 返回值

    ServerMod枚举类型的联邦学习训练模式。

## setServerMod

```java
public void setServerMod(ServerMod serverMod)
```

设置联邦学习训练模式。

- 参数
    - `serverMod`: ServerMod枚举类型的联邦学习训练模式，其中ServerMod.FEDERATED_LEARNING代表普通联邦学习模式（训练和推理使用同一个模型）ServerMod.HYBRID_TRAINING代表混合学习模式（训练和推理使用不同的模型，且云侧也包含训练过程）。

## getBatchSize

```java
public int getBatchSize()
```

获取联邦学习训练和推理时使用的单步训练样本数，即batch size。

- 返回值

    int类型的单步训练样本数batchSize。

## setBatchSize

```java
public void setBatchSize(int batchSize)
```

设置联邦学习训练和推理时使用的单步训练样本数，即batch size。需与模型的输入数据的batch size保持一致。

- 参数
    - `batchSize`: 单步训练样本数，即batch size。