# Class Status

\#include &lt;[status.h](https://atomgit.com/mindspore/mindspore-lite/blob/master/include/api/status.h)&gt;

## 构造函数和析构函数

```cpp
Status()
inline Status(enum StatusCode status_code, const std::string &status_msg = "")
inline Status(const StatusCode code, int line_of_code, const char *file_name, const std::string &extra = "")
~Status() = default;
```

## 公有成员函数

| 函数                                | 云侧推理是否支持 | 端侧推理是否支持 |
|-----------------------------|--------|--------|
| [enum StatusCode StatusCode() const](#statuscode)     |    √    |    √    |
| [inline std::string ToString() const](#tostring)     |    √    |    √    |
| [int GetLineOfCode() const](#getlineofcode)     |    √    |    √    |
| [inline std::string GetFileName() const](#getfilename)     |    √    |    √    |
| [inline std::string GetErrDescription() const](#geterrdescription)     |    √    |    √    |
| [inline std::string SetErrDescription(const std::string &err_description)](#seterrdescription)     |    √    |    √    |
| [inline void SetStatusMsg(const std::string &status_msg)](#setstatusmsg)     |    √    |    √    |
| [friend std::ostream &operator\<\<(std::ostream &os, const Status &s)](https://www.mindspore.cn/lite/api/zh-CN/master/generate/classmindspore_Status.html#operator<<std-ostream-os,-const-status-s)     |    √    |    √    |
| [bool operator==(const Status &other) const](#operatorconst-status-other)     |    √    |    √    |
| [bool operator==(enum StatusCode other_code) const](https://www.mindspore.cn/lite/api/zh-CN/master/generate/classmindspore_Status.html#operatorenum-statuscode-other-code)     |    √    |    √    |
| [bool operator!=(const Status &other) const](#operatorconst-status-other-1)     |    √    |    √    |
| [bool operator!=(enum StatusCode other_code) const](https://www.mindspore.cn/lite/api/zh-CN/master/generate/classmindspore_Status.html#operatorenum-statuscode-other-code-1)     |    √    |    √    |
| [explicit operator bool() const](#operator-bool)     |    √    |    √    |
| [explicit operator int() const](#explicit-operator-int-const)     |    √    |    √    |
| [static Status OK()](#ok)     |    √    |    √    |
| [bool IsOk() const](#isok)     |    √    |    √    |
| [bool IsError() const](#iserror)     |    √    |    √    |
| [static inline std::string CodeAsString(enum StatusCode c)](#codeasstring)     |    √    |    √    |

### StatusCode

```cpp
enum StatusCode StatusCode() const
```

获取状态码。

- 返回值

  状态码。

### ToString

```cpp
inline std::string ToString() const
```

状态码转成字符串。

- 返回值

  状态码的字符串。

### GetLineOfCode

```cpp
int GetLineOfCode() const
```

获取代码行数。

- 返回值

  代码行数。

### GetFileName

```cpp
inline std::string GetFileName() const
```

获取文件名。

- 返回值

  文件名。

### GetErrDescription

```cpp
inline std::string GetErrDescription() const
```

获取错误描述字符串。

- 返回值

  错误描述字符串。

### SetErrDescription

```cpp
inline std::string SetErrDescription(const std::string &err_description)
```

配置错误描述字符串。

- 参数

    - `err_description`: 错误描述字符串。

- 返回值

  状态信息字符串。

### SetStatusMsg

```cpp
inline void SetStatusMsg(const std::string &status_msg)
```

配置状态描述字符串。

- 参数

    - `status_msg`: 状态描述字符串。

### operator<<(std::ostream &os, const Status &s)

```cpp
friend std::ostream &operator<<(std::ostream &os, const Status &s)
```

状态信息写到输出流。

- 参数

    - `os`: 输出流。
    - `s`: 状态类。

- 返回值

  输出流。

### operator==(const Status &other)

```cpp
bool operator==(const Status &other) const
```

判断是否与另一个Status相等。

- 参数

    - `other`: 另一个Status。

- 返回值

  是否与另一个Status相等。

### operator==(enum StatusCode other_code)

```cpp
bool operator==(enum StatusCode other_code) const
```

判断是否与一个StatusCode相等。

- 参数

    - `other_code`: 一个StatusCode。

- 返回值

  是否与一个StatusCode相等。

### operator!=(const Status &other)

```cpp
bool operator!=(const Status &other) const
```

判断是否与另一个Status不相等。

- 参数

    - `other`: 另一个Status。

- 返回值

  是否与另一个Status不相等。

### operator!=(enum StatusCode other_code)

```cpp
bool operator!=(enum StatusCode other_code) const
```

判断是否与一个StatusCode不等。

- 参数

    - `other_code`: 一个StatusCode。

- 返回值

  是否与一个StatusCode不等。

### operator bool()

```cpp
explicit operator bool() const
```

重载bool操作，判断是否当前状态为kSuccess。

- 返回值

  是否当前状态为kSuccess。

### explicit operator int() const

```cpp
explicit operator int() const
```

重载int操作。当`Status`对象被作为整型表达式使用时，返回整型表示的当前状态值。

- 返回值

  当前状态值。

### OK

```cpp
static Status OK()
```

获取kSuccess的状态码。

- 返回值

  StatusCode::kSuccess。

### IsOk

```cpp
bool IsOk() const
```

判断是否是kSuccess的状态码。

- 返回值

  是否是kSuccess。

### IsError

```cpp
bool IsError() const
```

判断是否不是kSuccess的状态码。

- 返回值

  是否不是kSuccess。

### CodeAsString

```cpp
static inline std::string CodeAsString(enum StatusCode c)
```

获取StatusCode对应的字符串。

- 参数

    - `c`: 状态码枚举值。

- 返回值

  状态码对应的字符串。
