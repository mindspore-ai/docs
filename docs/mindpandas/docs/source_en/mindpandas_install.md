# Installing MindSpore Pandas

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindpandas/docs/source_en/mindpandas_install.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Confirming System Environment Information

The following table lists the environment required for installing, compiling and running MindSpore Pandas:

| software |             version              |
| :------: |:--------------------------------:|
|  Linux-x86_64 | Ubuntu \>=18.04<br/>Euler \>=2.9 |
|  Python  |             3.8-3.9              |
|  glibc  |             \>=2.25              |

- Make sure libxml2-utils is installed in your environment.
- Please refer to [requirements](https://gitee.com/mindspore/mindpandas/blob/master/requirements.txt) for other third party dependencies.

## Version Dependency

### Installing from pip command

If you use the pip, please download the whl package from [MindSpore Pandas](https://www.mindspore.cn/versions/en) page and install it.

> Installing whl package will download MindSpore Pandas dependencies automatically (detail of dependencies is shown in requirements.txt) in the networked state, and other dependencies should be installed manually.

### Installing from source code

Download [source code](https://gitee.com/mindspore/mindpandas), then enter the `mindpandas` directory to run build.sh script.

```shell
git clone https://gitee.com/mindspore/mindpandas.git
cd mindpandas
bash build.sh
```

The package is in output directory after compiled and you can install with pip. Taking Python3.8 for example, the command is as follows:

```shell
pip install output/mindpandas-0.2.0-cp38-cp38-linux_x86_64.whl
```

## Verification

Execute the following command in shell. If no `No module named 'mindpandas'` error is reported, the installation is successful.

```shell
python -c "import mindpandas"
```
