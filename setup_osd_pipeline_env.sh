#!/usr/bin/env bash
set -e

########################################
# CONFIG (edit if your paths differ)
########################################

# Where you installed Miniconda on RunPod
CONDA_ROOT="/workspace/miniconda3"

# Name of the env we’ve been using
ENV_NAME="osd_pipeline"

# Where the SpatialRGPT repo lives on the network volume
SPATIALRGPT_DIR="/workspace/SpatialRGPT"

PYTHON_VER="3.10"

########################################
# 1. Load conda
########################################

if [ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  echo "ERROR: ${CONDA_ROOT}/etc/profile.d/conda.sh not found."
  echo "Install Miniconda to ${CONDA_ROOT} first, then re-run this script."
  exit 1
fi

# Enable `conda` in this shell
. "${CONDA_ROOT}/etc/profile.d/conda.sh"

########################################
# 2. Create / activate env
########################################

if conda env list | grep -q "^${ENV_NAME} "; then
  echo "[INFO] Conda env '${ENV_NAME}' already exists. Using it."
else
  echo "[INFO] Creating conda env '${ENV_NAME}'..."
  conda create -y -n "${ENV_NAME}" python="${PYTHON_VER}"
fi

conda activate "${ENV_NAME}"

echo "[INFO] Using Python: $(which python)"

########################################
# 3. Core Python packages (torch, mmengine, etc.)
########################################

# Make sure we don’t hit the NumPy 2.x vs PyTorch binary issue
pip install --upgrade "pip"
pip install "numpy=1.26.0"

# Torch stack (matching what we used before)
pip install \
  "torch==2.3.0+cu121" \
  "torchvision==0.18.0" \
  "torchaudio==2.3.0" \
  --index-url https://download.pytorch.org/whl/cu121

# mmengine + mmcv
pip install -U openmim
mim install mmengine

# mmcv version we used
pip install "mmcv==2.0.0"

# Other libs from the README and our debugging
pip install \
  iopath \
  "pyequilib==0.3.0" \
  albumentations \
  einops \
  open3d \
  imageio \
  "matplotlib==3.8.4" \
  "transformers==4.37.2"

# Wis3D for 3D viz
pip install "https://github.com/zju3dv/Wis3D/releases/download/2.0.0/wis3d-2.0.0-py3-none-any.whl"

########################################
# 4. Detectron2 from source
########################################
# Assumes:
#   - nvcc is available (from RunPod image)
#   - CUDA version is compatible with torch 2.2.2+cu121
########################################

echo "[INFO] Installing detectron2 (this may take a while)..."
python -m pip install \
  "git+https://github.com/facebookresearch/detectron2.git" \
  --no-build-isolation

########################################
# 5. External repos for OSD pipeline
########################################

cd "${SPATIALRGPT_DIR}/dataset_pipeline"

mkdir -p osdsynth/external
cd osdsynth/external

########### Grounded-SAM (GSA) ###########
if [ ! -d "Grounded-Segment-Anything" ]; then
  echo "[INFO] Cloning Grounded-Segment-Anything..."
  git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
fi

cd Grounded-Segment-Anything

# Segment Anything
python -m pip install -e segment_anything

# GroundingDINO
pip install --no-build-isolation -e GroundingDINO

# RAM (Recognize Anything)
if [ ! -d "recognize-anything" ]; then
  git clone https://github.com/xinyu1205/recognize-anything.git
fi

pip install -r recognize-anything/requirements.txt
pip install --upgrade setuptools
pip install -e recognize-anything

########### Perspective Fields ###########
cd "${SPATIALRGPT_DIR}/dataset_pipeline/osdsynth/external"

if [ ! -d "PerspectiveFields" ]; then
  echo "[INFO] Cloning PerspectiveFields..."
  git clone https://github.com/jinlinyi/PerspectiveFields.git
fi

########################################
# 6. Download model weights (if needed)
########################################

cd "${SPATIALRGPT_DIR}/dataset_pipeline"

if [ -x "scripts/download_all_weights.sh" ]; then
  echo "[INFO] Downloading all weights (skip if already present)..."
  bash scripts/download_all_weights.sh
else
  echo "[WARN] scripts/download_all_weights.sh not found or not executable."
fi

########################################
# 7. Done
########################################

echo
echo "============================================================"
echo "[DONE] osd_pipeline environment is ready."
echo
echo "To use it in a new shell:"
echo "  source ${CONDA_ROOT}/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${SPATIALRGPT_DIR}/dataset_pipeline"
echo "  python run_template_qa.py --config configs/v2.py --input demo_images --vis True"
echo "============================================================"
echo
