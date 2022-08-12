"""api generator filter"""
import os
import sys

def lite_class_filter(file):
    """contents keywords filter"""
    with open(file, "rb") as f:
        data = f.read()
    try:
        content = data.decode('utf-8')
    except UnicodeDecodeError:
        content = data.decode('GBK')
    if "MS_API" in content:
        print(content)
    elif "MIND_API" in content:
        print(content)
    elif "MS_CORE_API" in content:
        print(content)
    elif "MS_DECLARE_PARENT" in content:
        print(content)
    else:
        pass

if __name__ == "__main__":
    values = sys.argv[-1]
    if os.path.isfile(values):
        lite_class_filter(values)
