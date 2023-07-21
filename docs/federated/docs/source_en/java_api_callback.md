# Callback

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/federated/docs/source_en/java_api_callback.md)

```java
import com.mindspore.flclient.model.Callback
```

Callback defines the hook function used to record training, evaluate and predict the results of different stages in end-to-side federated learning.

## Public Member Functions

| function                    |
| -------------------------------- |
| [abstract Status stepBegin()](#stepbegin) |
| [abstract Status stepEnd()](#stepend)   |
| [abstract Status epochBegin()](#epochbegin) |
| [abstract Status epochEnd()](#epochend) |

## stepBegin

```java
   public abstract Status stepBegin()
```

Execute step begin function.

- Returns

  Whether the execution is successful.

## stepEnd

```java
public abstract Status stepEnd()
```

Execute step end function.

- Returns

  Whether the execution is successful.

## epochBegin

```java
public abstract Status epochBegin()
```

Execute epoch begin function.

- Returns

  Whether the execution is successful.

## epochEnd

```java
public abstract Status epochEnd()
```

Execute epoch end function.

- Returns

  Whether the execution is successful.
