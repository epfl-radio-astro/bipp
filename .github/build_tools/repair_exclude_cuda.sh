#!/bin/bash

options=""
for line in $(find /usr/local/cuda/lib64/ -name "lib*.so.*" -printf "%f\n");
do
    options="$options --exclude $line"
done

echo "$options"
