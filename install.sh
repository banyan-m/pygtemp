pt_version=$(python -c 'import torch; print(torch.__version__)')

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${pt_version}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${pt_version}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${pt_version}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${pt_version}.html
pip install torch-geometric
pip install torch-geometric-temporal