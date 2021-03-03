# Function Differences of FloorDiv

PyTorch: The result is rounded toward 0, not rounded down. For example, if the division is -0.9, the rounded result is 0.

MindSpore：The result is rounded down by FLOOR. For example, if the division is -0.9, the rounded result is -1.
