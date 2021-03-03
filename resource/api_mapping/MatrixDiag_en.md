# Function Differences of MatrixDiag

PyTorch: Only 1D and 2D are supported. If the input is a 1D Tensor, a 2D diagonal matrix will be returned, and all elements in the returned matrix are set to 0 except the diagonals. If the input is a 2D Tensor, the value on the diagonal of the matrix will be returned. It also supports diagonal offsets specified by parameter `diagonal`.

MindSpore：Returns a diagonal matrix based on the given value, and k+1 dimensional diagonal matrix for k dimensional input.
