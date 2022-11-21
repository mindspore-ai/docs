#!/bin/bash                                                                                                                                                                                                
if [ -d out ]; then
    rm -rf out
fi

mkdir out
cd out || exit

if [ -f "Makefile" ]; then
  make clean
fi

cmake .. \
    -DMINDSPORE_PATH="`pip show mindspore | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"
make

