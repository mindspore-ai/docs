# Class GPUDeviceInfo

\#include &lt;[context.h](https://atomgit.com/mindspore/mindspore-lite/blob/r2.9.0/include/api/context.h)&gt;

派生自[DeviceInfoContext](./classmindspore_DeviceInfoContext.md)，模型运行在GPU上的配置。

## 公有成员函数

| 函数                                                       | 云侧推理是否支持 | 端侧推理是否支持 |
|------------------------------------------------------------|----------|----------|
| [enum DeviceType GetDeviceType() const override](#getdevicetype)                    | √        | √        |
| [void SetDeviceID(uint32_t device_id)](#setdeviceid)                     | √        | √        |
| [uint32_t GetDeviceID() const](#getdeviceid)                             | √        | √        |
| [int GetRankID() const](#getrankid)                                    | √        | √        |
| [int GetGroupSize() const](#getgroupsize)                                 | √        | √        |
| [inline void SetPrecisionMode(const std::string &precision_mode)](#setprecisionmode) | √        | √        |
| [inline std::string GetPrecisionMode() const](#getprecisionmode)                     | √        | √        |
| [void SetEnableFP16(bool is_fp16)](#setenablefp16)                         | √        | √        |
| [bool GetEnableFP16() const](#getenablefp16)                               | √        | √        |
| [void SetEnableGLTexture(bool is_enable_gl_texture)](#setenablegltexture)             | ✕        | √        |
| [bool GetEnableGLTexture() const](#getenablegltexture)                          | ✕        | √        |
| [void SetGLContext(void *gl_context)](#setglcontext)                      | ✕        | √        |
| [void *GetGLContext() const](#getglcontext)                               | ✕        | √        |
| [void SetGLDisplay(void *gl_display)()](#setgldisplay)                      | ✕        | √        |
| [void *GetGLDisplay() const](#getgldisplay)                               | ✕        | √        |

### GetDeviceType

```cpp
enum DeviceType GetDeviceType() const override
```

- 返回值

  DeviceType::kGPU

### SetDeviceID

```cpp
void SetDeviceID(uint32_t device_id)
```

用于指定设备ID。

- 参数

    - `device_id`: 设备ID。

### GetDeviceID

```cpp
uint32_t GetDeviceID() const
```

- 返回值

  已配置的设备ID。

### GetRankID

```cpp
int GetRankID() const
```

- 返回值

  当前运行的RANK ID。

### GetGroupSize

```cpp
int GetGroupSize() const
```

- 返回值

  当前运行的GROUP SIZE。

### SetPrecisionMode

```cpp
inline void SetPrecisionMode(const std::string &precision_mode)
```

用于指定推理时算子精度。

- 参数

    - `precision_mode`: 可选值`origin`（以模型中指定精度进行推理）， `fp16`（以FP16精度进行推理），默认值: `origin`。

### GetPrecisionMode

```cpp
inline std::string GetPrecisionMode() const
```

- 返回值

  已配置的精度模式。

### SetEnableFP16

```cpp
void SetEnableFP16(bool is_fp16)
```

用于指定是否以FP16精度进行推理。

- 参数

    - `is_fp16`: 是否以FP16精度进行推理。

### GetEnableFP16

```cpp
bool GetEnableFP16() const
```

- 返回值

  已配置的精度模式。

### SetEnableGLTexture

```cpp
void SetEnableGLTexture(bool is_enable_gl_texture)
```

用于指定是否绑定OpenGL纹理数据。

- 参数

    - `is_enable_gl_texture`: 是否在推理时绑定OpenGL纹理数据。

### GetEnableGLTexture

```cpp
bool GetEnableGLTexture() const
```

- 返回值

  已配置的绑定OpenGL纹理数据模式。

### SetGLContext

```cpp
void SetGLContext(void *gl_context)
```

用于指定OpenGL EGLContext。

- 参数

    - `gl_context`: OpenGL的当前运行时的EGLContext值。

### GetGLContext

```cpp
void *GetGLContext() const
```

- 返回值

  已配置的指向OpenGL EGLContext的指针。

### SetGLDisplay

```cpp
void SetGLDisplay(void *gl_display)
```

用于指定OpenGL EGLDisplay。

- 参数

    - `gl_display`: OpenGL的当前运行时的EGLDisplay值。

### GetGLDisplay

```cpp
void *GetGLDisplay() const
```

- 返回值

  已配置的指向OpenGL EGLDisplay的指针。
