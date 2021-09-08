#!/bin/bash

set -e
make html

if [ $? -ne 0 ]; then
    echo "make html failed"
    exit
fi

cd build_zh_cn/html
python -m http.server
