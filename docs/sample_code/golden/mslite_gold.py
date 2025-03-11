"""
This module provides a script to convert into benchmark format.
"""
import argparse
from tool import save_bin

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save inputs and outputs in specified format")
    parser.add_argument('--inputFile', type=str, required=True, help='Input your specified data.')
    parser.add_argument('--outputFile', type=str, required=True, help='Input your specified data.')
    parser.add_argument('--savePath', type=str, required=True, help="Path to the data directory.")
    args = parser.parse_args()

    save_bin(args)
