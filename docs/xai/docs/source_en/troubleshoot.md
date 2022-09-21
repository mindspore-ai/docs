# Troubleshooting

<a href="https://gitee.com/mindspore/docs/blob/r1.9/docs/xai/docs/source_en/troubleshoot.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source.png"></a>

## Import Errors

<font size=3>**Q: What can I do if libgomp `cannot allocate memory in static TLS block` error occurs when importing `mindspore_xai` or its subpackages?**</font>

A: You have to do the following steps:

Reinstall scikit-learn 1.0.2:

```bash
pip install --force-reinstall scikit-learn==1.0.2
```

List all site-packages directories:

```bash
python -m site
```

There is a list of directories in `sys.path` shown the previous, find the `scikit_learn.libs/` sub-directory in the directories of the list.
Once you located the `scikit_learn.libs/` directory, say which is underneath `<SITE_PKGS>/`, list files inside it:

```bash
ls <SITE_PKGS>/scikit_learn.libs
```

There is a dynamical library libgomp inside with a filename like `libgomp-XXX.so.XXX`, append the absolute path to the environment variable `LD_PRELOAD`:

```bash
export LD_PRELOAD=$LD_PRELOAD:<SITE_PKGS>/scikit_learn.libs/libgomp-XXX.so.XXX
```

Run your MindSpore XAI scripts again.
