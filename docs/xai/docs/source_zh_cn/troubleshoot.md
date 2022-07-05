# 故障排除

<a href="https://gitee.com/mindspore/docs/blob/r1.8/docs/xai/docs/source_zh_cn/troubleshoot.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 导入错误

<font size=3>**Q: 在导入`mindspore_xai`或其子包时遇到 libgomp `cannot allocate memory in static TLS block` 错误应该怎么办？**</font>

A: 你须要进行以下的步骤：

重新安装 scikit-learn 1.0.2：

```bash
pip install --force-reinstall scikit-learn==1.0.2
```

列出所有 site-packages 文件夹：

```bash
python -m site
```

在上一步中显示的`sys.path`文件夹列表中的目录找出`scikit_learn.libs/`子文件夹的位置，当你找到位置后（例如位于`<SITE_PKGS>/`）把里面的文件列出：

```bash
ls <SITE_PKGS>/scikit_learn.libs
```

文件夹内有一个 libgomp 动态链接库 `libgomp-XXX.so.XXX`，把它的绝对路径加进环境变量 `LD_PRELOAD`：

```bash
export LD_PRELOAD=$LD_PRELOAD:<SITE_PKGS>/scikit_learn.libs/libgomp-XXX.so.XXX
```

重新运行你的 MindSpore XAI 脚本。
