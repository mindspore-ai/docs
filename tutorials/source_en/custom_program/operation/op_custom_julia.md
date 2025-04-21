# Custom Operator with Third Party Frontend

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_en/custom_program/operation/op_custom_julia.md)

As one of the future development goals of MindSpore,  the fusion of AI and scientific computing draws more and more attention from the industry. Based on the flexibility of the representation, MindSpore custom operator also makes exploration on the scientific computing, and introduces the programming frontend for HPC to MindSpore via custom operator.

## Defining Custom Operator of julia Type

Julia is a high level general programming language which has high performance and is easy to use. Julia is firstly designed for scientific computing, and also gain the favor of general users due to its high effience.
The custom operator of julia type uses Julia to describe the internal calculation logic of the operator. The framework will automatically call this function during the network runtime.

Operator output shape and data type inference can be realized by defining Python functions to describe the inference logic of the operator output shape and the data type.

If the custom operator only supports specific input and output data types, you need to define the operator information. For the creation of operator information, please refer to [Registering the Operator Information](https://www.mindspore.cn/tutorials/en/br_base/custom_program/operation/op_custom_adv.html#registering-the-operator-information).

## Custom Operator Use Cases of julia Type

Takes the function of adding two input tensors as an example to introduce how to define a custom operator of julia type.

Firstly, users need to implement Julia functions via separate files, such as (add.jl):

```julia
# add.jl
module Add
# inputs: x, y, output: z, output should use .= to inplace assign
function add(x, y, z)
    z .= x + y
end
end
```

Secondly, refer to the Julia function written above in a custom operator in the network script, taking test_custom_julia.py as an example:

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_device(device_target="CPU")

if __name__ == "__main__":
    # Define custom operators of type julia
    op = ops.Custom("./add.jl:Add:add", out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="julia")
    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(ms.Tensor(x0), ms.Tensor(x1))
    print(output)
```

The following points need to be explained in this example:

- Use Python lambda functions to infer the output shape and data type, and pass them to the `out_shape` and `out_dtype` parameters of the `Custom` primitive. In this example, the lambda function indicates that the output shape and data type are the same as the information of the first input tensor.
- The operator information is not registered, so the operator information of the custom operator will be inferred from the inputs.

Execute case:

```bash
python test_custom_julia.py
```

The execution result is as follows:

```text
[[2. 2.]
 [4. 4.]]
```

Matters need attention:

1. User should make sure to download the correct version of Julia, that is, version >= 1.6.0.
2. User is required to set `julia/lib` to `LD_LIBRARY_PATH`, because the Julia C API called at runtime is obtained from `libjulia.so`, taking julia-1.6.5 as an example:

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

3. `Custom` operator's first arg `func` should keep format like `file_name:module_name:func_name`, `file_name` should include path, suggest using absolute path.
4. Julia file should include `module`, `module` include `function`, both ends with `end`.
5. The input and output order of the Julia function needs to be consistent with the input and output order of the operator.
6. The final output of the Julia function, i.e. assignment of kernel output, needs to use `.=`, otherwise the result cannot be written to memory.
7. Julia code supports [Julia](https://docs.julialang.org/en/v1/)'s common syntax, users need to ensure that the syntax is correct and the function can be executed correctly.
8. Users who want to use Julia's third-party software packages in Julia files need to download the corresponding software to ensure that they can call it correctly, which can be called through `import pkg; pkg.add ("somepkg")` to install.
9. `julia array` is `column major` arranged in memory, while `numpy array` is `row major`. If Julia and numpy are compared, non-elemwise calculations need to consider memory arrangement. In the Julia function, the conversion of `numpy array` and `julia array` can be performed by following the following code example:An example of MatMul:

     ```julia
    function change_input_to_row_major(x)
        return permutedims(reshape(x, reverse(size(x))), length(size(x)):-1:1)
    end

    function change_output_to_row_major(x)
        return reshape(permutedims(x, length(size(x)):-1:1), size(x))
    end
    ```

    Taking matrix multiplication as an example:

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

For more complete examples of julia-type custom operators, see the [use cases](https://gitee.com/mindspore/mindspore/blob/br_base/tests/st/graph_kernel/custom/test_custom_julia.py) in the MindSpore source code.
