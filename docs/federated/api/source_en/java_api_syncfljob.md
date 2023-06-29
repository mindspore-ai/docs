# SyncFLJob

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/federated/api/source_en/java_api_syncfljob.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

```java
import com.mindspore.flclient.SyncFLJob
```

SyncFLJob defines the API flJobRun() for starting federated learning on the device, the API modelInference() for inference on the device, and the API getModel() for obtaining the latest model on the cloud.

## Public Member Functions

| **Function**                     |
| -------------------------------- |
| public FLClientStatus flJobRun() |
| public int[] modelInference()    |
| public FLClientStatus getModel() |

## flJobRun

```java
public FLClientStatus flJobRun()
```

Starts a federated learning task on the device, for specific usage, please refer to the [interface introduction document](https://www.mindspore.cn/federated/api/en/r1.3/interface_description_federated_client.html).

- Return value

    The status code of the flJobRun request.

## modelInference

```java
public int[] modelInference()
```

Starts an inference task on the device, for specific usage, please refer to the [interface introduction document](https://www.mindspore.cn/federated/api/en/r1.3/interface_description_federated_client.html).

- Return value

  int[] composed of the labels inferred from the input.

## getModel

```java
public FLClientStatus getModel()
```

Obtains the latest model on the cloud, for specific usage, please refer to the [interface introduction document](https://www.mindspore.cn/federated/api/en/r1.3/interface_description_federated_client.html).

- Return value

  The status code of the getModel request.
