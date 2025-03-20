"""
This script run the onnx model, and save the input and output.

Usage:
    python onnx_demo.py --input_model <file_path> --save_path <save_path>

Arguments:
    --input_model: The ONNX model file.
    --save_path: Path to the data directory.
    --inputShape: Stable shape for inputs (e.g.[[1,2],[3],[4,5,6]]).
    --inDataFile: Input your specified data.
"""
import os
import argparse
import ast
import onnx
import numpy as np
import onnxruntime as rt

def generate_random_input_data_and_run_model(args):
    """
    generate random input data and save, and run model
    """
    # Init
    if not os.path.exists(os.path.join(args.savePath)):
        print(f"mkdir: {args.savePath}")
        os.makedirs(args.savePath)
    model = onnx.load(args.modelFile)
    graph = model.graph
    # org_output_num = len(graph.output)
    sess = rt.InferenceSession(args.modelFile)
    input_tensors = sess.get_inputs()
    input_dict = {}

    if args.inDataFile:  # User specifies input
        input_data_files = args.inDataFile.split(':')
        assert len(input_data_files) == len(input_tensors), "Shape of input data is not compatible with the model!"
        for i, input_tensor in enumerate(input_tensors):
            tensor_type = input_tensor.type[7:-1]
            if tensor_type == "float":
                tensor_type = "float32"
            if "int" in input_tensor.type:
                tensor_type = "int32"

            dtype = np.dtype(tensor_type)
            with open(input_data_files[i], "rb") as file:
                input_data = np.fromfile(file, dtype=dtype)
            if args.inputShape:
                stable_shape = ast.literal_eval(args.inputShape)
                shape_info = stable_shape[i]
                assert len(input_data) == np.prod(
                    shape_info), "Shape of input data is not compatible with the input shape!"
                input_data = input_data.reshape(shape_info)
            else:
                assert len(input_data) == np.prod(
                    input_tensor.shape), "Shape of input data is not compatible with the model!"
                input_data = input_data.reshape(input_tensor.shape)
            input_dict[input_tensor.name] = input_data
    else:  # Generate random input and save
        for input_tensor in input_tensors:
            input_info = {
                "input_name": input_tensor.name,
                "type": input_tensor.type,
                "shape": input_tensor.shape,
            }
            print(input_info)

        for i, input_tensor in enumerate(input_tensors):
            tensor_type = input_tensor.type[7:-1]
            if tensor_type == "float":
                tensor_type = "float32"

            shape_info = input_tensor.shape
            if args.inputShape:
                stable_shape = ast.literal_eval(args.inputShape)
                shape_info = stable_shape[i]

            # generate random data and save
            if "int" in tensor_type:
                input_data = np.random.uniform(low=0, high=20, size=shape_info).astype(tensor_type)
                # input_mindir_data = input_data.astype(np.int32)
                # input_data.astype(np.int32).flatten().tofile(os.path.join(args.savePath, f"input.bin{i}"))
            else:
                input_data = np.random.uniform(low=-1, high=1, size=shape_info).astype(tensor_type)
                # input_mindir_data = input_data.astype(tensor_type)
                # input_data.flatten().tofile(os.path.join(args.savePath, f"input.bin{i}"))

            input_dict[input_tensor.name] = input_data

    np.savez(os.path.join(args.savePath, f"input.npz"), **input_dict)

    # run model
    res = sess.run(None, input_dict)
    i = 0
    output_dict = {}
    for output in graph.output:
        output_dict[output.name] = res[i]
        i += 1
    np.savez(os.path.join(args.savePath, f"output.npz"), **output_dict)
    # return input_dict, res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ONNX model with random input data.")
    parser.add_argument('--modelFile', type=str, required=True, help="The ONNX model file.")
    parser.add_argument('--savePath', type=str, required=True, help="Path to the data directory.")
    parser.add_argument('--inputShape', type=str, help="Stable shape for inputs (e.g.[[1,2],[3],[4,5,6]]).")
    parser.add_argument('--inDataFile', type=str, help='Input your specified data.')
    args1 = parser.parse_args()

    generate_random_input_data_and_run_model(args1)
