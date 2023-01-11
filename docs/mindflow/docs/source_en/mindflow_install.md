# MindFlow Installation

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindflow/docs/source_en/mindflow_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>&nbsp;&nbsp;

## System Environment Information Confirmation

- The hardware platform should be Ascend, GPU.
- See our [MindSpore Installation Guide](https://www.mindspore.cn/install/en) to install MindSpore.
- All other dependencies are included in [requirements.txt](https://gitee.com/mindspore/mindscience/blob/r0.2.0-alpha/MindFlow/requirements.txt).

## Installation

You can install MindFlow either by pip or by source code.

### Installation by pip

```bash
pip install mindflow_[gpu|ascend]
```

### Installation by Source Code

1. Download source code from Gitee.

   ```bash
   git clone https://gitee.com/mindspore/mindscience.git -b r0.2.0-alpha
   cd {PATH}/mindscience/MindFlow
   ```

2. Compile in Ascend backend.

   ```bash
   bash build.sh -e ascend -j8
   ```

3. Compile in GPU backend.

   ```bash
   export CUDA_PATH={your_cuda_path}
   bash build.sh -e GPU -j8
   ```

4. Install the compiled .whl file.

   ```bash
   cd {PATH}/mindscience/MindFLow/output
   pip install mindflow_*.whl
   ```

## Installation Verification

Successfully installed, if there is no error message such as `No module named 'mindflow'` when execute the following command:

```bash
python -c 'import mindflow'
```
