# Version

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/feature/atomgit/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/lite/api/source_en/api_java/version.md)

```java
import com.mindspore.config.Version;
```

Get MindSpore Lite version info.

## Public Member Functions

| function                                   | Supported At Cloud-side Inference | Supported At Device-side Inference |
| ------------------------------------------ |--------|--------|
| [static void init()](#init) | √    | √      |
| [static native String version()](#version) | √    | √      |

## init

```java
public static void init()
```

Init function.

## version

```java
public static native String version()
```

Get MindSpore Lite version info.

- Returns

  MindSpore Lite version info.
