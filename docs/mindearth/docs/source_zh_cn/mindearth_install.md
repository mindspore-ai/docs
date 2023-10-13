# 安装MindSpore Earth

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindearth/docs/source_zh_cn/mindearth_install.md)&nbsp;&nbsp;

## 确认系统环境信息

- 硬件平台为Ascend、GPU。
- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装。
- 其余依赖请参见[requirements.txt](https://gitee.com/mindspore/mindscience/blob/r0.5/MindEarth/requirements.txt)。

## 安装方式

可以采用pip安装或者源码编译安装两种方式。

### pip安装

即将到来 ...

### 源码安装

1. 从Gitee下载源码。

   ```bash
   git clone https://gitee.com/mindspore/mindscience.git
   cd {PATH}/mindscience/MindEarth
   ```

2. 编译Ascend后端源码。

   ```bash
   bash build.sh -e ascend -j8
   ```

3. 编译GPU后端源码。

   ```bash
   export CUDA_PATH={your_cuda_path}
   bash build.sh -e GPU -j8
   ```

4. 安装编译所得whl包。

   ```bash
   cd {PATH}/mindscience/MindEarth/output
   pip install mindearth_*.whl
   ```

## 验证是否成功安装

执行如下命令，如果没有报错`No module named 'mindearth'`，则说明安装成功。

```bash
python -c 'import mindearth'
```