# API Sacnning

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.1/docs/devtoolkit/docs/source_en/VSCode_api_scan.md)

## Single-file Analysis

1. Right-click on the open Python file editing screen and select "Scan Local Files".

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/docs/devtoolkit/docs/source_zh_cn/images/clip_image116.jpg)

2. The following figure is an example of a scan that generates the analysis results of the use of the torch.Tensor interface in this file, including "PyTorch APIs that can be transformed", "Probably the result of torch.Tensor API" and "PyTorch API that does not provide a direct mapping relationship at this time".

   where

   - "PyTorch APIs that can be transformed" means PyTorch APIs used in the Documentation can be converted to MindSpore APIs.
   - "Probably the result of torch.Tensor API" means APIs with the same name as torch.Tensor, which may be torch.Tensor APIs and can be converted to MindSpore APIs.
   - "PyTorch API that does not provide a direct mapping relationship at this time" means APIs that are PyTorch APIs or possibly torch.Tensor APIs, but don't directly correspond to MindSpore APIs.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/docs/devtoolkit/docs/source_zh_cn/images/clip_image117.jpg)

## Multi-file Analysis

1. Click the MindSpore Dev Toolkit icon in the left sidebar.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/docs/devtoolkit/docs/source_zh_cn/images/clip_image118.jpg)

2. Generate a project tree view of the current IDE project containing only Python files on the left.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/docs/devtoolkit/docs/source_zh_cn/images/clip_image119.jpg)

3. If the Python file is selected, you can get the interface analysis results of the file.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/docs/devtoolkit/docs/source_zh_cn/images/clip_image120.jpg)

4. If the file directory is selected, you can get the interface analysis results of all Python files in that directory.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/docs/devtoolkit/docs/source_zh_cn/images/clip_image121.jpg)

5. The blue font parts are all clickable and will automatically open the page in the user-default browser.

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/docs/devtoolkit/docs/source_zh_cn/images/clip_image122.jpg)

   ![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/docs/devtoolkit/docs/source_zh_cn/images/clip_image123.jpg)
