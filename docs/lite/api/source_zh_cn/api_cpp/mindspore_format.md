# mindspore::Format

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.0/docs/lite/api/source_zh_cn/api_cpp/mindspore_format.md)

以下表格描述了MindSpore MSTensor保存的数据支持的排列格式。

## Format

 **enum**类型变量。

| 定义 | 值 |
| --- | --- |
| DEFAULT_FORMAT | -1 |
| NCHW | 0 |
| NHWC | 1 |
| NHWC4 | 2 |
| HWKC | 3 |
| HWCK | 4 |
| KCHW | 5 |
| CKHW | 6 |
| KHWC | 7 |
| CHWK | 8 |
| HW | 9 |
| HW4 | 10 |
| NC | 11 |
| NC4 | 12 |
| NC4HW4 | 13 |
| NCDHW | 14 |
| NWC | 15 |
| NCW | 16 |
| NDHWC| 17 |
| NC8HW8 | 18 |
|  FRACTAL_NZ | 19 |
| ND | 20 |
| NC1HWC0| 21 |
|  FRACTAL_Z | 22 |
| NC1C0HWPAD | 23 |
| NHWC1C0 | 24 |
| FSR_NCHW | 25 |
| FRACTAL_DECONV | 26 |
| C1HWNC0 | 27 |
| FRACTAL_DECONV_TRANSPOSE | 28 |
| FRACTAL_DECONV_SP_STRIDE_TRANS | 29 |
| NC1HWC0_C04 | 30 |
| FRACTAL_Z_C04 | 31 |
| CHWN | 32 |
| FRACTAL_DECONV_SP_STRIDE8_TRANS | 33 |
| HWCN | 34 |
| NC1KHKWHWC0 | 35 |
| BN_WEIGHT | 36 |
|  FILTER_HWCK | 37 |
| LOOKUP_LOOKUPS | 38 |
| LOOKUP_KEYS | 39 |
| LOOKUP_VALUE | 40 |
| LOOKUP_OUTPUT | 41 |
| LOOKUP_HITS | 42 |
| C1HWNCoC0 | 43 |
| MD | 44 |
| FRACTAL_ZZ | 45 |
| DHWCN | 46 |
| NDC1HWC0 | 47 |
| FRACTAL_Z_3D | 48 |
| CN | 49 |
| DHWNC | 50 |
| FRACTAL_Z_3D_TRANSPOSE | 51 |
| FRACTAL_ZN_LSTM | 52 |
| FRACTAL_Z_G | 53 |
| ND_RNN_BIAS | 54 |
| FRACTAL_ZN_RNN | 55 |
| NYUV | 56 |
| NYUV_A | 57 |
| NCL | 58 |
| NUM_OF_FORMAT | 59 |
