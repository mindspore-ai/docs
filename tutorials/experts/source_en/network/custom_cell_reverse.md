# Customizing **bprop** Function

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/network/custom_cell_reverse.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

Users can customize backpropagation (calculation) function of the nn.Cell object, thus control the process of the nn.Cell object gradient calculation, locating gradient problems.

Custom bprop functions are used by: adding a user-defined bprop function to the defined nn. Cell object. The training process uses user-defined bprop functions to generate reverse graphs.

The sample code is as follows:

```python
ms.set_context(mode=ms.PYNATIVE_MODE)

class Net(nn.Cell):
    def construct(self, x, y):
        z = x * y
        z = z * y
        return z

    def bprop(self, x, y, out, dout):
        x_dout = x + y
        y_dout = x * y
        return x_dout, y_dout

grad_all = ops.GradOperation(get_all=True)
output = grad_all(Net())(ms.Tensor(1, ms.float32), ms.Tensor(2, ms.float32))
print(output)
```
