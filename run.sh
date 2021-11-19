## run
CUDA_VISIBLE_DEVICES=1 python train.py

CUDA_VISIBLE_DEVICES=1 python eva_ep.py
CUDA_VISIBLE_DEVICES=0 python test_style.py

### requirement
conda install python=3.6
pip install scipy==1.2.2
pip install kornia
pip install scikit-image