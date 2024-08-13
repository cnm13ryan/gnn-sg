#!/bin/bash

# Function to check PyTorch version
function get_pytorch_version() {
    if poetry run python -c "import torch" &> /dev/null; then
        pytorch_version=$(poetry run python -c "import torch; print(torch.__version__)")
        pytorch_version=$(echo $pytorch_version | cut -d'.' -f1,2)
        echo "Detected PyTorch version: $pytorch_version"
    else
        echo "PyTorch is not installed in the current environment."
        exit 1
    fi
}

# Function to check NumPy version
function get_numpy_version() {
    if poetry run python -c "import numpy" &> /dev/null; then
        numpy_version=$(poetry run python -c "import numpy; print(numpy.__version__)")
        echo "Detected NumPy version: $numpy_version"
        if [[ $numpy_version != 2* ]]; then
            echo "NumPy version 2.x is required."
            exit 1
        fi
    else
        echo "NumPy is not installed in the current environment."
        exit 1
    fi
}

# Function to prompt user for CUDA version tag
function prompt_cuda_version() {
    echo "Select the CUDA version tag:"
    echo "1) CPU (cpu)"
    echo "2) CUDA 11.8 (cu118)"
    echo "3) CUDA 12.1 (cu121)"
    while true; do
        read -p "Enter the number corresponding to your choice (1/2/3): " choice
        case $choice in
            1 ) cuda_tag="cpu"; break;;
            2 ) cuda_tag="cu118"; break;;
            3 ) cuda_tag="cu121"; break;;
            * ) echo "Invalid choice. Please enter 1, 2, or 3.";;
        esac
    done
}

# Function to set TORCH_CUDA_ARCH_LIST environment variable
function set_cuda_arch_list() {
	export TORCH_CUDA_ARCH_LIST="7.5"  # Adjust based on your GPU compute capability (https://developer.nvidia.com/cuda-gpus)
}

# Function to install packages with the chosen CUDA version tag
function install_packages() {
    # Add PyTorch and its dependencies using poetry
    poetry add torch-scatter torch-sparse --source "https://data.pyg.org/whl/torch-${pytorch_version}+${cuda_tag}.html"
}

# Check PyTorch version
get_pytorch_version

# Check NumPy version
get_numpy_version

# Prompt user for CUDA version tag
prompt_cuda_version

# Set CUDA architecture list
set_cuda_arch_list

# Install packages
install_packages
