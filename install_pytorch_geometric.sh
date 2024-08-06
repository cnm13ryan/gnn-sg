#!/bin/bash
poetry run pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+${CUDA=cpu}.html
