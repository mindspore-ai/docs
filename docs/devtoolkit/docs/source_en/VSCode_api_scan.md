# API Mapping - API Sacnning

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/devtoolkit/docs/source_en/VSCode_api_scan.md)

## Functions Introduction

* Quickly scan for APIs that appear in your code and display API details directly in the sidebar.
* For the convenience of users of other machine learning frameworks, the corresponding MindSpore APIs are matched by association by scanning the mainstream framework APIs that appear in the code.
* The data version of API mapping supports switching. Please refer to the section [API Mapping - Version Switching](https://www.mindspore.cn/devtoolkit/docs/en/master/VSCode_change_version.html) for details.

## File-level API Mapping Scanning

1. Right-click anywhere in the current file to open the menu and select "Scan Local Files".

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image116.jpg)

2. The right-hand column will populate with the scanned operators in the current file, including three scanning result list "PyTorch APIs that can be transformed", "Probably the result of torch.Tensor API" and "PyTorch API that does not provide a direct mapping relationship at this time".

    where

    * "PyTorch APIs that can be transformed" means PyTorch APIs used in the Documentation can be converted to MindSpore APIs.
    * "Probably the result of torch.Tensor API" means APIs with the same name as torch.Tensor, which may be torch.Tensor APIs and can be converted to MindSpore APIs.
    * "PyTorch API that does not provide a direct mapping relationship at this time" means APIs that are PyTorch APIs or possibly torch.Tensor APIs, but don't directly correspond to MindSpore APIs.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image117.jpg)

## Project-level API Mapping Scanning

1. Click the MindSpore API Mapping Scan icon on the left sidebar of Visual Studio Code.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image118.jpg)

2. Generate a project tree view of the current IDE project containing only Python files on the left sidebar.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image119.jpg)

3. If you select a single Python file in the view, you can get a list of operator scan results for that file.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image120.jpg)

4. If you select a file directory in the view, you can get a list of operator scan results for all Python files in that directory.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image121.jpg)

5. The blue font parts are all clickable and will automatically open the page in the user-default browser.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image122.jpg)

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/devtoolkit/docs/source_zh_cn/images/clip_image123.jpg)
