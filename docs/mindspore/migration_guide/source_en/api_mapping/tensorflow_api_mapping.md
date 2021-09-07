# TensorFlow and MindSpore

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/tensorflow_api_mapping.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

Mapping between TensorFlow APIs and MindSpore APIs, which is provided by the community.

| TensorFlow 1.15 APIs                                         | MindSpore APIs                                                 | Description |
|------------------------------------------------------|----------------------------------------------------------------|------|
| [tf.gradients](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/gradients)                                                            |[mindspore.ops.GradOperation](https://www.mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.GradOperation.html)                                            |[差异对比](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/tensorflow_diff/GradOperation.md)|
| [tf.sparse.SparseTensor](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/sparse/SparseTensor)                                        |[mindspore.SparseTensor](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore/mindspore.SparseTensor.html)                                                | 功能一致 |
| [tf.stop_gradient](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/stop_gradient)                                                    |[mindspore.ops.stop_gradient](https://www.mindspore.cn/tutorials/en/master/autograd.html#stop-gradient)                                                               | 功能一致 |
| [tf.Tensor](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/Tensor)                                                                  |[mindspore.Tensor](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore/mindspore.Tensor.html)                                                            | 功能一致 |
