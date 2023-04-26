# 算子扫描

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/devtoolkit/docs/source_zh_cn/operator_scanning.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>

## 功能介绍

* 快速扫描代码中出现的API，在侧边栏直接展示API详情。
* 为方便其他机器学习框架用户，通过扫描代码中出现的主流框架API，联想匹配对应MindSpore API。

## 使用步骤

### 文件级别算子扫描

1. 在当前文件任意位置处点击鼠标右键，打开菜单，点击菜单最上方的"operator scan"。

   ![img](./images/clip_image100.jpg)

2. 右边栏会自动弹出，展示扫描出的算子，并展示包含名称，网址等信息的详细列表。若本文件中未扫描到算子，则不会弹出窗口。

   ![img](./images/clip_image101.jpg)

3. 蓝色字体的部分均可以点击，会自动在上方再打开一栏，展示网页。

   ![img](./images/clip_image102.jpg)

4. 点击右上角"导出"按钮，可将内容导出到csv表格。

   ![img](./images/clip_image103.jpg)

### 项目级别算子扫描

1. 在当前文件任意位置处点击鼠标右键，打开菜单，点击菜单上方第二个"operator scan project-level"，或在上方工具栏选择"Tools"，再选择"operator scan project-level"。

   ![img](./images/clip_image104.jpg)

   ![img](./images/clip_image105.jpg)

2. 右边栏会弹出整个项目中扫描出的算子，并展示包含名称，网址等信息的详细列表。

   ![img](./images/clip_image106.jpg)

3. 在上方框中可以选择单个文件，下方框中将单独展示此文件中的算子，文件选择可以随意切换。

   ![img](./images/clip_image107.jpg)

   ![img](./images/clip_image108.jpg)

4. 蓝色字体部分均可以点击，会自动在上方再打开一栏，展示网页。

   ![img](./images/clip_image109.jpg)

5. 点击"导出"按钮，可将内容导出到csv表格。

   ![img](./images/clip_image110.jpg)