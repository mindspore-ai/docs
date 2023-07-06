# API扫描

<a href="https://gitee.com/mindspore/docs/blob/master/docs/devtoolkit/docs/source_zh_cn/VSCode_api_scan.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 单文件分析

1. 在打开的Python文件编辑界面点击右键，选择“扫描本地文件”。

   ![img](./images/clip_image116.jpg)

2. 扫描后，生成该文件中torch.tensor接口使用分析结果，包括“可以转化的PyTorch API”、“可能是torch.Tensor API的结果”、
   “暂未提供直接映射关系的PyTorch API”三种分析结果。

   ![img](./images/clip_image117.jpg)

## 多文件分析

1. 点击左侧边栏MindSpore Dev Toolkit图标。

   ![img](./images/clip_image118.jpg)

2. 在左侧生成当前IDE工程中仅含Python文件的工程树视图。

   ![img](./images/clip_image119.jpg)

3. 若选中Python文件，可获取该文件的接口分析结果。

   ![img](./images/clip_image120.jpg)

4. 若选中文件目录，可获取该目录下所有Python文件的接口分析结果。

   ![img](./images/clip_image121.jpg)
