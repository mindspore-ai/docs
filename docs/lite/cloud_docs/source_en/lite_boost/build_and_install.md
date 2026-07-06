
# Build and Install

## Prerequisites

- Python >= 3.8
- pytorch >= 2.6.0
- torch_npu >= 2.6.0
- Ascend CANN >= 8.5.0

> The above lists the main dependencies and their version numbers. For other dependencies, please refer to the project source code [requirements.txt](https://atomgit.com/mindspore/mindspore-lite/blob/r2.10/mindspore-lite/lite_boost/requirements.txt).

## Build

```bash
# Build from the MindSpore Lite project root directory
bash build.sh -I arm64 -O lite_boost -j 32
```

### Build Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-d` | Debug mode | Release |
| `-r` | Release mode | Release |
| `-v` | Show full build commands | Off |
| `-i` | Incremental build (do not clean build directory) | Off |
| `-j[n]` | Number of build threads | 8 |
| `-h` | Print help information | - |

The build output is located in the mindspore-lite/output directory of the MindSpore Lite project.

## Install

```bash
pip install output/lite_boost-<version>-<tag>.whl
```
