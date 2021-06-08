# README

Usage:

Cd into the directory:

```bash
cd ascend310_resnet50_preprocess_sample
```

Configure the cmake project, if MindSpore is installed by pip:

```bash
cmake . -DMINDSPORE_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`
```

or installed by binary:

```bash
cmake . -DMINDSPORE_PATH=path-to-your-custom-dir
```

Then compile:

```bash
make
```

Run the sample:

```bash
./resnet50_sample
```
