#!/bin/bash

set -e
set -x

export PYTHONHASHSEED=0

python -m pip install cibuildwheel
python -m cibuildwheel --output-dir wheelhouse
