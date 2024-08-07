# Graph Neural Network architectures for systematic generalisation problems

## Getting started

Create a virtual environment and install the necessary packages below

```python3
python -m venv venv
```
```python3
source venv/bin/activate # for linux/macOS
```
```python3
pip install -r requirements.txt 
```
Also install the following packages using the link relevant for your hardware:
```
./install_torch_scatter_sparse.sh
```
See https://pypi.org/project/torch-sparse/ 


## To reproduce results in the paper just build and run the following command in `src`

```python
python train.py experiments=`pick a dataset model config file from configs/experiments`
```

The config can be easily adjusted from the command line using a dot file path notation with the `experiments.` prefix. For example, running for 10 epochs on the rcc8 dataset amounts to:
```
python train.py experiments=fb_model_rcc8 experiments.epochs=10
```

## Cite
If you find this work/code useful, please consider citing us:
```bibtex
@misc{khalid2024systematicreasoning,
      title={Systematic Reasoning About Relational Domains With Graph Neural Networks}, 
      author={Irtaza Khalid and Steven Schockaert},
      year={2024},
      eprint={2407.17396},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.17396}, 
}
```

