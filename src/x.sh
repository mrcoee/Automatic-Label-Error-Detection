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

printf "#-------------------------------#\n"
printf "# SCRIPTS EXECUTED SUCCESSFULLY #\n"
printf "#-------------------------------#\n"
