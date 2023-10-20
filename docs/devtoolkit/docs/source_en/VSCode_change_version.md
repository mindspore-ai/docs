# API Mapping - Version Switching

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/devtoolkit/docs/source_en/VSCode_change_version.md)

## Overview

API mapping refers to the mapping relationship between PyTorch API and MindSpore API. In MindSpore Dev Toolkit, it provides two functions: API mapping search and API mapping scan, and users can freely switch the version of API mapping data.

## API Mapping Data Version Switching

1. Different versions of API mapping data will result in different API mapping scans and API mapping search results, but will not affect the version of MindSpore in the environment. The default version is the same as the plugin version, and the version information is displayed in the bottom left status bar.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/docs/devtoolkit/docs/source_zh_cn/images/clip_image129.jpg)

2. Clicking on this status bar will bring up a drop-down box at the top of the page containing options for the default version numbers that can be switched. Users can click on any version number to switch versions, or click on the "Customize Input" option and enter another version number in the pop-up input box to switch versions.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/docs/devtoolkit/docs/source_zh_cn/images/clip_image130.jpg)

3. Click on any version number to start switching versions, and the status bar in the lower left indicates the status of version switching.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/docs/devtoolkit/docs/source_zh_cn/images/clip_image131.jpg)

4. If you want to customize the version number, click the "Customize Input" option in the drop-down box, and the drop-down box will be changed to an input box, enter the version number according to the format of 2.1 or 2.1.0, and then press the Enter key to start switching the version, and the status bar in the lower-left corner will indicate the status of the switching.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/docs/devtoolkit/docs/source_zh_cn/images/clip_image132.jpg)

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/docs/devtoolkit/docs/source_zh_cn/images/clip_image133.jpg)

5. If the switch is successful, the message in the lower right indicates that the switch is successful, and the status bar in the lower left displays information about the version of the API mapping data after the switch.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/docs/devtoolkit/docs/source_zh_cn/images/clip_image134.jpg)

6. If the switch fails, the message in the lower right indicates that the switch fails, and the status bar in the lower left shows the API mapping data version information before the switch. If the switch fails due to non-existent version number or network error, please check and try again. If you want to see the latest documentation, you can switch to the master version.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/docs/devtoolkit/docs/source_zh_cn/images/clip_image135.jpg)

7. When the customized version number is switched successfully, this version number is added to the drop-down box for display.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/docs/devtoolkit/docs/source_zh_cn/images/clip_image136.jpg)