# Installation
Our code is tested on the following environment.

## 1. Create conda environment
```bash
conda create -n gsworld python=3.8.16
conda activate gsworld
```

## 2. Install PyTorch
```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

## 3. Install packages from MMLab
```bash
pip install openmim
mim install mmcv==2.0.1
mim install mmdet==3.0.0
mim install mmsegmentation==1.0.0
mim install mmdet3d==1.1.1
```

## 4. Install other packages
```bash
pip install spconv-cu117
pip install timm
pip install git+https://github.com/NVIDIA/gpu_affinity
```

## 4. Install custom CUDA ops
```bash
cd model/encoder/gaussian_encoder/ops && pip install -e .
cd model/head/localagg && pip install -e .
```

## 5. (Optional) For visualization
```bash
pip install pyvirtualdisplay matplotlib==3.7.2 PyQt5 vtk==9.0.1 mayavi==4.7.3 configobj numpy==1.23.5
```