# ClientManager

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/federated/docs/source_en/java_api_clientmanager.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

```java
import com.mindspore.flclient.model.ClientManager
```

ClientManager defines end-side federated learning custom algorithm model management objects.

## Public Member Functions

| function                    |
| -------------------------------- |
| [static void registerClient(Client client)](#registerclient)  |
| [static Client getClient(String name)](#getclient)  |

## registerClient

```java
public static void registerClient(Client client)
```

Register client object.

- Parameters

    - `client`: Need register client object.

## getClient

```java
public static Client getClient(String name)
```

Get client object.

- Parameters

    - `name`: Client object name.
- Returns

  Client object.
