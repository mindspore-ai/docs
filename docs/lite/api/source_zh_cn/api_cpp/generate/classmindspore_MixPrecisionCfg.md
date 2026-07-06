# Class MixPrecisionCfg

\#include &lt;[cfg.h](https://atomgit.com/mindspore/mindspore-lite/blob/r2.10/include/api/cfg.h)&gt;

`MixPrecisionCfg`MindSpore Lite训练混合精度配置类。

## 构造函数

```cpp
  MixPrecisionCfg() {
    dynamic_loss_scale_ = false;
    loss_scale_ = 128.0f;
    num_of_not_nan_iter_th_ = 1000;
  }
```

```cpp
  MixPrecisionCfg(const MixPrecisionCfg &rhs) {
    this->dynamic_loss_scale_ = rhs.dynamic_loss_scale_;
    this->loss_scale_ = rhs.loss_scale_;
    this->keep_batchnorm_fp32_ = rhs.keep_batchnorm_fp32_;
    this->num_of_not_nan_iter_th_ = rhs.num_of_not_nan_iter_th_;
  }
```

- 参数

    - `rhs`: 混合精度配置。tr

## 析构函数

```cpp
~MixPrecisionCfg() = default
```

## 公有成员变量

```cpp
bool dynamic_loss_scale_ = false
```

混合精度训练中是否启用动态损失比例。

```cpp
float loss_scale_
```

初始损失比例。

```cpp
uint32_t num_of_not_nan_iter_th_
```

动态损失阈值。

```cpp
bool is_raw_mix_precision_
```

原始模型是否是原生混合精度模型。

```cpp
bool keep_batchnorm_fp32_ = true
```

原型模型是否保持BatchNorm算子为Fp32格式。
