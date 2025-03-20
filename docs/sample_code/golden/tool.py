"""
This module provides functionality to convert NPZ files into benchmark data format.
"""
import os
from collections import OrderedDict
import numpy as np

def save_bin(args):
    """
    Specific implementation
    """
    if not os.path.exists(args.savePath):
        print(f"mkdir:{args.savePath}")
        os.makedirs(args.savePath)

    try:
        input_dict = np.load(args.inputFile)
        print(f"Loaded inputs from {args.inputFile}: {list(input_dict.keys())}")
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error loading inputs: {e}")
        return
    i = 0
    input_dict = OrderedDict(input_dict)
    for key, input_data in input_dict.items():
        print(f"input {key}: shape={input_data.shape}, dtype={input_data.dtype}")
        if np.issubdtype(input_data.dtype, np.integer):
            input_data.astype(np.int32).flatten().tofile(os.path.join(args.savePath, f"input.bin{i}"))
        else:
            input_data.flatten().tofile(os.path.join(args.savePath, f"input.bin{i}"))
        i = i + 1

    try:
        output_dict = np.load(args.outputFile)
        print(f"Loaded outputs from {args.outputFile}: {list(output_dict.keys())}")
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error loading outputs: {e}")
        return

    opened = 0
    output_dict = OrderedDict(output_dict)
    output_file = os.path.join(args.savePath, 'model.out')
    for i, output_data in output_dict.items():
        print(f"output {i}: shape={output_data.shape}, dtype={output_data.dtype}")
        mode = 'w' if opened == 0 else 'a'
        if str(output_data.shape) == "[]":
            output_shape = [1]
        else:
            output_shape = output_data.shape

        with open(output_file, mode) as text_file:
            opened = 1
            if not output_shape:
                output_shape.append(len(output_data))

            text_file.write(f"{i} {len(output_data.shape)} ")
            text_file.write(" ".join([str(s) for s in output_data.shape]))
            text_file.write('\n')
            print(f"result shape: {len(output_data.flatten())}")

            for k in output_data.flatten():
                text_file.write(f"{k} ")
            text_file.write('\n')
