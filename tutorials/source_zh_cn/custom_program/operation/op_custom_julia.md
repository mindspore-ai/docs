# 自定义算子接入第三方前端

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_zh_cn/custom_program/operation/op_custom_julia.md)

作为MindSpore未来的发展方向之一，AI和科学计算的融合越来越受到业界的重视。MindSpore自定义算子基于自身表达的灵活性，也在科学计算方面做出了探索：把面向HPC的编程前端以自定义算子的方式接入MindSpore。

## julia类型的自定义算子开发概述

Julia是一种速度快且使用简单的高级通用编程语言，最初设计用于科学计算领域，而由于其高效而实用的特性，近些年来越来越受到用户的青睐，逐步迈向主流编程语言。
julia类型的自定义算子使用Julia语法定义算子实现函数，描述算子内部计算逻辑的实现。网络运行时框架会自动调用执行相应的Julia函数。

算子输出shape和数据类型推导可以通过定义Python函数实现，描述算子输出shape和数据类型的推导逻辑。

若自定义算子只支持特定的输入输出数据类型，则需要定义算子信息，算子信息生成方式请参考[算子信息注册](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/custom_program/operation/op_custom_adv.html#算子信息注册)。

## julia类型的自定义算子开发用例

下面以两个输入张量相加为例，介绍julia类型的自定义算子开发流程:

首先，用户需要通过单独文件实现Julia函数，如(add.jl)：

```julia
# add.jl
module Add
# inputs: x, y, output: z, output should use .= to inplace assign
function add(x, y, z)
    z .= x + y
end
end
```

其次，在网络脚本中通过自定义算子方式引用上面所写的Julia函数，以test_custom_julia.py为例：

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_device(device_target="CPU")

if __name__ == "__main__":
    # 定义julia类型的自定义算子
    op = ops.Custom("./add.jl:Add:add", out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="julia")
    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(ms.Tensor(x0), ms.Tensor(x1))
    print(output)
```

本例中，有如下几点需要说明：

- 用Python lambda函数定义输出shape和数据类型推理函数，并分别传给`Custom`原语的`out_shape`和`out_dtype`参数。本例中lambda函数表明输出shape和数据类型与第一个输入张量的信息相同。
- 未注册算子信息，所以自定义算子的算子信息将会从算子输入中推理。

执行用例：

```shell
python test_custom_julia.py
```

执行结果：

```text
[[2. 2.]
 [4. 4.]]
```

注意事项：

1. 用户需确保下载正确版本的Julia，即version>=1.6.0。
2. 由于运行时调用的Julia C API是从`libjulia.so`中获取的，因此需要用户设置`julia/lib`到`LD_LIBRARY_PATH`，以julia-1.6.5为例:

   ```bash
   # download julia-1.6.5
   wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.5-linux-x86_64.tar.gz
   # for arm server
   # wget https://julialang-s3.julialang.org/bin/linux/aarch64/1.6/julia-1.6.5-linux-aarch64.tar.gz
   # extract file
   tar xvf julia-1.6.5-linux-x86_64.tar.gz
   # if $JULIA_DIR not exist
   export LD_LIBRARY_PATH=$PWD/julia-1.6.5/lib:$LD_LIBRARY_PATH
   # else
   export LD_LIBRARY_PATH=$JULIA_DIR/lib:$LD_LIBRARY_PATH
   ```

3. `Custom` 第一个入参指定用户书写的Julia函数需按照`file_name:module_name:func_name`格式指定，`file_name`需包含文件路径，建议使用绝对路径。
4. Julia代码文件需包含`module`，`module`内包含`function`，且`module`或`function`都以`end`结束。
5. Julia函数的输入输出顺序需与算子的输入输出顺序一致。
6. Julia函数的最终输出，即kernel output的赋值需要使用`.=`，否则结果无法写入内存。
7. Julia代码支持[Julia](https://docs.julialang.org/en/v1/)的常用语法，用户需自行保证语法正确，函数可正确执行。
8. 用户想在Julia文件内使用Julia的第三方软件包，需自行下载对应软件以确保能正常调用，可以通过 `import pkg; pkg.add("somepkg")`进行安装。
9. `julia array`在内存上是按照`column major`排列的，而`numpy array`是按照`row major`排列的。如果Julia和numpy做比较，非elemwise计算需考虑内存排布。在Julia函数中，可以通过如下代码示例进行`numpy array`和`julia array`的相互转换：

   ```julia
   function change_input_to_row_major(x)
       return permutedims(reshape(x, reverse(size(x))), length(size(x)):-1:1)
   end

   function change_output_to_row_major(x)
       return reshape(permutedims(x, length(size(x)):-1:1), size(x))
   end
   ```

   以矩阵乘为例：

   ```julia
   # julia array is column-major, numpy array is row-major
   # user should change julia or numpy's layout to keep same behavior
   #= EXAMPLE
   A[2,3]               B[3,4]               C[2,4]
   NUMPY:
   [[1, 2, 3]       [[1, 2, 3, 4]         [[38, 44, 50,  56]
    [4, 5, 6]]       [5, 6, 7, 8]          [83, 98, 113,128]]
                     [9,10,11,12]]
   JULIA:
   change_input_to_row_major:
   1.inputs read numpy data from memory:
   [[1, 3, 5]       [[1, 4, 7,10]
    [2, 4, 6]]       [2, 5, 8,11]
                     [3, 6, 9,12]]
   2.inputs after reshape(reverse(shape)):
   [[1, 4]          [[1, 5, 9]
    [2, 5]           [2, 6,10]
    [3, 6]]          [3, 7,11]
                     [4, 8,12]]
   3.inputs after transpose/permutedims:
   [[1, 2, 3]       [[1, 2, 3, 4]         [[38, 44, 50,  56]
    [4, 5, 6]]       [5, 6, 7, 8]          [83, 98, 113,128]]
                     [9,10,11,12]]
   change_output_to_row_major:
   1.output after transpose/permutedims:
                                          [[38, 83]
                                           [44, 98]
                                           [50,113]
                                           [56,128]
   2.output after reshape:
                                          [[38, 50, 83, 113]
                                           [44, 56, 98, 128]]
   3.output read numpy data from memory:
                                          [[38, 44, 50,  56]
                                           [83, 98,113, 128]]
   =#
   function foo!(x, y, z)
       x = change_input_to_row_major(x)
       y = change_input_to_row_major(y)
       z .= gemm(x, y, z)
       z .= change_output_to_row_major(z)
   end
   ```

更多完整的jullia类型自定义算子的例子可以参见MindSpore源码中的[用例](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/tests/st/graph_kernel/custom/test_custom_julia.py)。
