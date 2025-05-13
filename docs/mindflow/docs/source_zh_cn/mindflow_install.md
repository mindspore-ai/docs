# 安装MindSpore Flow

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.5.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.5.0/docs/mindflow/docs/source_zh_cn/mindflow_install.md)&nbsp;&nbsp;

## 确认系统环境信息

- 硬件平台为Ascend、GPU。
- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装。
- 其余依赖请参见[requirements.txt](https://gitee.com/mindspore/mindscience/blob/r0.7/MindFlow/requirements.txt)。
- MindSpore Flow需MindSpore版本>=2.5.0，Python版本需>=3.9。

## 安装方式

可以采用pip安装或者源码编译安装两种方式。

### pip安装

```bash
# gpu and ascend are supported
export DEVICE_NAME=gpu
pip install mindflow_${DEVICE_NAME}
```

### 源码安装

1. 从Gitee下载源码。

   ```bash
   git clone -b r0.7 https://gitee.com/mindspore/mindscience.git
   cd {PATH}/mindscience/MindFlow
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
   cd {PATH}/mindscience/MindFLow/output
   pip install mindflow_*.whl
   ```

## 验证是否成功安装

执行如下命令，如果没有报错`No module named 'mindflow'`，则说明安装成功。

```bash
python -c 'import mindflow'
```