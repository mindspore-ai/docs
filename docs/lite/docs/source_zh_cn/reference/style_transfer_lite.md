# 风格迁移模型

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/reference/style_transfer_lite.md)

## 风格迁移介绍

风格迁移模型可以根据demo内置的标准图片改变用户目标图片的艺术风格，并在App图像预览界面中显示出来。用户可保存风格迁移结果，或者恢复图片的原始形态。

使用demo打开目标图片：

![image_before_transfer](images/before_transfer.png)

选择左起第一张标准图片进行风格迁移，效果如图：

![image_after_transfer](images/after_transfer.png)

使用MindSpore Lite实现风格迁移的[示例代码](https://gitee.com/mindspore/models/tree/master/official/lite/style_transfer)。
