#!/bin/bash
#
# usage: ./x.sh
#

clear

# settings
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# cython setup
python3 setup.py build_ext --inplace

# execute main python scripts
# python3 eval.py

# clear directory from cython setup files
# rm -rf __pycache__
# rm -rf build
# rm *.so

printf "#-------------------------------#\n"
printf "# SCRIPTS EXECUTED SUCCESSFULLY #\n"
printf "#-------------------------------#\n"
