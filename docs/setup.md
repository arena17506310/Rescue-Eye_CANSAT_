# RESCUE-EYE 환경 세팅 (Phase 0)
# 이 문서는 최초 환경 구성 시에만 참조

---

## RTX 3080 데스크톱

```bash
# Conda 환경 생성
conda create -n rescue-eye python=3.10 -y && conda activate rescue-eye

# PyTorch + CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 기본 라이브러리
pip install open3d numpy opencv-python scipy

# COLMAP (Ubuntu 기준)
sudo apt-get install colmap
# 또는 pip install pycolmap

# DA3 (Depth Anything 3)
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
# 또는: git clone https://github.com/ByteDance-Seed/Depth-Anything-3 && pip install -e .

# SAM2
pip install git+https://github.com/facebookresearch/sam2.git

# Florence-2
pip install transformers  # Florence-2는 HuggingFace transformers로 로드
# from transformers import AutoProcessor, AutoModelForCausalLM

# 검증
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import open3d; print(open3d.__version__)"
python -c "import pycolmap; print('COLMAP OK')"
```

**완료 기준**: `torch.cuda.is_available() == True`

---

## 노트북 (지상국)

```bash
conda create -n rescue-eye-ground python=3.10 -y && conda activate rescue-eye-ground
pip install pyserial numpy opencv-python
```

---

## RPi Zero 2W

```bash
pip install picamera2 smbus2 RPi.GPIO numpy
```
