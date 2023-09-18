# API Scanning

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/devtoolkit/docs/source_en/api_scanning.md)

## Functions Introduction

* Quickly scan the APIs in the code and display the API details directly in the sidebar.
* For the convenience of other machine learning framework users, by scanning the mainstream framework APIs that appear in the code, associative matching the corresponding MindSpore API.

## Usage Steps

### Document-level API Scanning

1. Right click anywhere in the current file to open the menu, and click "API scan" at the top of the menu.

    ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image100.jpg)

2. The right sidebar will automatically pop up to show the scanned operator and display a detailed list containing the name, URL and other information. If no operator is scanned in this document, no pop-up window will appear.

    where:

    * "PyTorch/TensorFlow APIs that can be converted to MindSpore APIs" means PyTorch or TensorFlow APIs used in the Documentation that can be converted to MindSpore APIs.
    * "APIs that cannot be converted at this time" means APIs that are PyTorch or TensorFlow APIs but do not have a direct equivalent to MindSpore APIs.
    * "Possible PyTorch/TensorFlow API" refers to a convertible case where there is a possible PyTorch or TensorFlow API because of chained calls.
    * TensorFlow API scanning is an experimental feature.

    ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image101.jpg)

3. Click the blue words, and another column will automatically open at the top to show the page.

    ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image102.jpg)

4. Click the "export" button in the upper right corner to export the content to a csv table.

    ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image103.jpg)

### Project-level API Scanning

1. Right-click anywhere on the current file to open the menu, click the second option "API scan project-level" at the top of the menu, or select "Tools" in the toolbar above, and then select "API scan project-level".

    ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image104.jpg)

    ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image105.jpg)

2. The right sidebar pops up a list of scanned operators from the entire project, and displays a detailed list containing information such as name, URL, etc.

    ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image106.jpg)

3. In the upper box you can select a single file, and in the lower box the operators in this file will be shown separately, and the file selection can be switched at will.

    ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image107.jpg)

    ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image108.jpg)

4. Click the blue words, and another column will automatically open at the top to show the page.

    ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image109.jpg)

5. Click the "export" button in the upper right corner to export the content to a csv table.

    ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image110.jpg)
