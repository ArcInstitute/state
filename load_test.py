from __future__ import annotations

import os
import sys
import math
import logging
from pathlib import Path
from typing import Tuple, List, Dict
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent  # adjust if needed
sys.path.append(str(project_root))
sys.path.append(str(project_root / "vci_pretrain"))

# Import model class directly
from models.pertsets import PertSetsPerturbationModel

MODEL_DIR = Path(

)
DATA_PATH = Path(
    "/large_storage/ctc/userspace/aadduri/datasets/tahoe_45_ct/plate1.h5"
)
CELL_SET_LEN = 256   
CONTROL_SAMPLES = 50 
LAYER_IDX = 5
FIG_DIR = Path(__file__).resolve().parent / "figures" / "tahoe_21_256" 
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load model directly from checkpoint
#checkpoint_path = MODEL_DIR / "step=148000.ckpt"
checkpoint_path = MODEL_DIR / "checkpoints" / "step=124000.ckpt" # "step=step=104000-val_loss=val_loss=0.0472.ckpt"

if not checkpoint_path.exists():
    # Try other common checkpoint names
    for ckpt_name in ["best.ckpt", "last.ckpt", "epoch=*.ckpt"]:
        ckpt_files = list(MODEL_DIR.glob(f"checkpoints/{ckpt_name}"))
        if ckpt_files:
            checkpoint_path = ckpt_files[0]
            break
    else:
        raise FileNotFoundError(f"Could not find checkpoint in {MODEL_DIR}/checkpoints/")

logger.info(f"Loading model from checkpoint: {checkpoint_path}")
model = PertSetsPerturbationModel.load_from_checkpoint(str(checkpoint_path), strict=False)