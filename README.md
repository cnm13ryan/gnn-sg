# Graph Neural Network Architectures for Systematic Generalization Problems

## Getting Started

This project is managed using [Poetry](https://python-poetry.org/) for dependency management and package installation. 

**Please note that this project can only be run on a Linux machine and requires Python 3.10.** 

If you are using an NVIDIA GPU, CUDA 12.1 is the maximum supported version for package compatibility.


### 1. Setting Up the Environment

1. **Python Version Management with `pyenv`:**
    - Ensure that `pyenv` is used to manage the Python version for this project, specifically Python 3.10. 
    - Configure Poetry to be controlled by `pyenv` to maintain consistent Python versions.

2. **Check System Configurations:**
    Before proceeding with the installation, verify the following:
    - **CUDA Version:** Ensure `nvcc --version` returns 12.1.
    - **Python Version:** Ensure `python --version` returns 3.10.
    - **Upgrade pip, setuptools, and wheel to latest versions:** 

This is important to avoid issues when installing `torch-sparse` and `torch-scatter`.

3. **Install Dependencies:**
    Poetry will automatically create a virtual environment and resolve package dependencies. To install everything, simply run:
    ```bash
    poetry install
    ```
    
    After installation, you can check the path of the virtual environment created by Poetry with:
    ```bash
    poetry env list --full-path
    ```
    
    You can manually activate this virtual environment using the following command:
    ```bash
    source <path_to_virtual_env>/bin/activate
    ```


### 2. Running the Project

To reproduce the results from the associated research paper or to run your experiments, navigate to the `src` directory and use the following command `python train.py experiments=<config_file_path>`. 

Modify the configuration directly from the command line. For example, to run the training on the RCC8 dataset for 10 epochs, use:
```bash
python train.py experiments=fb_model_rcc8 experiments.epochs=10
```


### 3. System Requirements

- **Operating System:** Linux
- **Python Version:** 3.10 (managed via `pyenv`)
- **PyTorch Version:** 2.3.0 (for torch-scatter and torch-sparse)
- **CUDA Version:** 12.1 (for cuda supported GPUs, compatible with pytorch version)


### 5. Citation

```
@misc{khalid2024systematicreasoning,
      title={Systematic Reasoning About Relational Domains With Graph Neural Networks},
      author={Irtaza Khalid and Steven Schockaert},
      year={2024},
      eprint={2407.17396},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={<https://arxiv.org/abs/2407.17396>},
}

```
