# MindSpore Earth Installation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindearth/docs/source_en/mindearth_install.md)&nbsp;&nbsp;

## System Environment Information Confirmation

- The hardware platform should be Ascend, GPU.
- See our [MindSpore Installation Guide](https://www.mindspore.cn/install/en) to install MindSpore.
- All other dependencies are included in [requirements.txt](https://gitee.com/mindspore/mindscience/blob/master/MindEarth/requirements.txt).

## Installation

You can install MindSpore Earth either by pip or by source code.

### Installation by pip

Coming soon ...

### Installation by Source Code

1. Download source code from Gitee.

   ```bash
   git clone https://gitee.com/mindspore/mindscience.git
   cd {PATH}/mindscience/MindEarth
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
   cd {PATH}/mindscience/MindEarth/output
   pip install mindearth_*.whl
   ```

## Installation Verification

Successfully installed, if there is no error message such as `No module named 'mindearth'` when execute the following command:

```bash
python -c 'import mindearth'
```