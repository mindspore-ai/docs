# format_c

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_zh_cn/api_c/format_c.md)

```C
#include<format_c.h>
```

以下表格描述了MindSpore MSTensor保存的数据支持的排列格式。

## MSFormat

```C
typedef enum MSFormat {
  kMSFormatNCHW = 0,
  kMSFormatNHWC = 1,
  kMSFormatNHWC4 = 2,
  kMSFormatHWKC = 3,
  kMSFormatHWCK = 4,
  kMSFormatKCHW = 5,
  kMSFormatCKHW = 6,
  kMSFormatKHWC = 7,
  kMSFormatCHWK = 8,
  kMSFormatHW = 9,
  kMSFormatHW4 = 10,
  kMSFormatNC = 11,
  kMSFormatNC4 = 12,
  kMSFormatNC4HW4 = 13,
  kMSFormatNCDHW = 15,
  kMSFormatNWC = 16,
  kMSFormatNCW = 17
} MSFormat;
```

**enum**类型变量。

| 类型定义 | 值  | 描述   |
| -------- | --- | ------ |
| kMSFormatNCHW     | 0   | NCHW   |
| kMSFormatNHWC     | 1   | NHWC   |
| kMSFormatNHWC4    | 2   | NHWC4  |
| kMSFormatHWKC     | 3   | HWKC   |
| kMSFormatHWCK     | 4   | HWCK   |
| kMSFormatKCHW     | 5   | KCHW   |
| kMSFormatCKHW     | 6   | CKHW   |
| kMSFormatKHWC     | 7   | KHWC   |
| kMSFormatCHWK     | 8   | CHWK   |
| kMSFormatHW       | 9   | HW     |
| kMSFormatHW4      | 10  | HW4    |
| kMSFormatNC       | 11  | NC     |
| kMSFormatNC4      | 12  | NC4    |
| kMSFormatNC4HW4   | 13  | NC4HW4 |
| kMSFormatNCDHW    | 15  | NCDHW  |
| kMSFormatNWC      | 16  | NWC    |
| kMSFormatNCW      | 17  | NCW    |
