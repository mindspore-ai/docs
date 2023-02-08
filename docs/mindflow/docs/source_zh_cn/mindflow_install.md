# 安装MindFlow

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindflow/docs/source_zh_cn/mindflow_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>&nbsp;&nbsp;

## 确认系统环境信息

- 硬件平台为Ascend、GPU。
- 参考[MindSpore安装指南](https://www.mindspore.cn/install)，完成MindSpore的安装。
- 其余依赖请参见[requirements.txt](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/requirements.txt)。

## 安装方式

可以采用pip安装或者源码编译安装两种方式。

### pip安装

```bash
export MS_VERSION=2.0.0a0
export MindFlow_VERSION=0.1.0a0
# gpu and ascend are supported
export DEVICE_NAME=gpu
# cuda-10.1 and cuda-11.1 are supported
export CUDA_VERSION=cuda-11.1

# Python3.7
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindScience/${DEVICE_NAME}/x86_64/${CUDA_VERSION}/mindflow_${DEVICE_NAME}-${MindFlow_VERSION}-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# Python3.8
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindScience/${DEVICE_NAME}/x86_64/${CUDA_VERSION}/mindflow_${DEVICE_NAME}-${MindFlow_VERSION}-cp38-cp38-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindScience/${DEVICE_NAME}/x86_64/${CUDA_VERSION}/mindflow_${DEVICE_NAME}-${MindFlow_VERSION}-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 源码安装

1. 从Gitee下载源码。

   ```bash
   git clone https://gitee.com/mindspore/mindscience.git
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