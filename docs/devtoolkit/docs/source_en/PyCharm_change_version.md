# API Mapping - API Version Switching

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/devtoolkit/docs/source_en/PyCharm_change_version.md)

## Overview

API mapping refers to the mapping relationship between PyTorch API and MindSpore API.
In MindSpore Dev Toolkit, it provides two functions: API mapping search and API mapping scan, and users can freely switch the version of API mapping data.

## API Mapping Data Version Switching

1. When the plug-in starts, it defaults to the same API mapping data version as the current version of the plug-in. The API mapping data version is shown in the lower right. This version number only affects the API mapping functionality of this section and does not change the version of MindSpore in the environment.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image137.jpg)

2. Click the API mapping data version to bring up the selection list. You can choose to switch to other version by clicking on the preset version, or you can choose "other version" to try to switch by inputting other version number.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image138.jpg)

3. Click on any version number to start switching versions. An animation below indicates the switching status.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image139.jpg)

4. If you want to customize the version number, select "other version" in the selection list, enter the version number in the popup box, and click ok to start switching versions. Note: Please input the version number in 2.1 or 2.1.0 format, otherwise there will be no response when you click ok.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image140.jpg)

5. If the switch is successful, the lower right status bar displays the API mapping data version information after the switch.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image141.jpg)

6. If the switch fails, the lower right status bar shows the API mapping data version information before the switch. If the switch fails due to non-existent version number or network error, please check and try again. If you want to see the latest documentation, you can switch to the master version.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image142.jpg)

7. When a customized version number is successfully switched, this version number is added to the list of versions to display.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image143.jpg)