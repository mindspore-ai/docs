# TrainCfg

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/feature/atomgit/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/lite/api/source_en/api_java/train_cfg.md)

```java
import com.mindspore.config.TrainCfg;
```

Configuration parameters used for model training on the device.

## Public Member Functions

| function                                   | Supported At Cloud-side Inference | Supported At Device-side Inference |
| ------------------------------------------ |--------|--------|
| [boolean init()](#init) | ✕     | √      |
| [boolean init(String loss_name)](#init) | ✕     | √      |
| [void free()](#free) | ✕     | √      |
| [boolean addMixPrecisionCfg(boolean dynamicLossScale, float lossScale, int thresholdIterNum)](#addmixprecisioncfg) | ✕     | √      |
| [long getTrainCfgPtr()](#gettraincfgptr) | ✕ | √ |

## init

```java
public boolean init()
```

Init train config.

- Returns

  init status.

```java
public boolean init(String loss_name)
```

Init train config specified loss name.

- Parameters

- `loss_name`: loss_name loss name used for split inference and train part.

- Returns

  Initialization state.  

## free

```java
public void free()
```

Free train config.

## addMixPrecisionCfg

```java
public boolean addMixPrecisionCfg(boolean dynamicLossScale, float lossScale, int thresholdIterNum)
```

Add mix precision config to train config.

- Parameters
- `dynamicLossScale`: dynamicLossScale if dynamic or fix loss scale factor.

- `lossScale`: loss scale factor.

- `thresholdIterNum`: thresholdIterNum a threshold for modifying loss scale when dynamic loss scale is enabled.

- Returns

  add status.

## getTrainCfgPtr

```java
public long getTrainCfgPtr()
```

Get train config pointer.

- Returns

  train config pointer.
