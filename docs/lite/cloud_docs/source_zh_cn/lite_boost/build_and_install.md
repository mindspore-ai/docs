
# 编译安装

## 环境要求

- Python >= 3.8
- pytorch >= 2.6.0
- torch_npu >= 2.6.0
- 昇腾CANN >= 8.5.0

> 列举主要依赖包以及对应的版本号，其他依赖包请参考项目代码[requirements.txt](https://atomgit.com/mindspore/mindspore-lite/blob/r2.10/mindspore-lite/lite_boost/requirements.txt)文件。

## 编译

```bash
# 从MindSpore Lite项目根目录编译
bash build.sh -I arm64 -O lite_boost -j 32
```

### 编译参数介绍

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-d` | Debug 模式 | Release |
| `-r` | Release 模式 | Release |
| `-v` | 显示完整编译命令 | 关闭 |
| `-i` | 增量编译（不清理 build 目录） | 关闭 |
| `-j[n]` | 编译线程数 | 8 |
| `-h` | 打印帮助信息 | - |

编译生成的产物在MindSpore Lite工程的mindspore-lite/output目录下。

## 安装

```bash
pip install output/lite_boost-<version>-<tag>.whl
```
