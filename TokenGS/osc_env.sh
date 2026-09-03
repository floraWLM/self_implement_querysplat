#!/bin/bash

module purge
module load cuda/12.6.2

conda activate tokengs

export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CUDAHOSTCXX="$(command -v g++)"
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"

echo "=== TokenGS OSC Environment ==="
echo "CC=$CC"
echo "CXX=$CXX"
echo "CUDAHOSTCXX=$CUDAHOSTCXX"
echo "CUDA_HOME=$CUDA_HOME"
echo "nvcc=$(command -v nvcc)"
