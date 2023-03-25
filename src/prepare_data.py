import os

import numpy as np
from PIL import Image
from tqdm.contrib.concurrent import process_map
from scipy.special import softmax
from pathlib import Path

from metric_computations import compute_metrics
from config import cfg


def process_data(logits_fn):
    
    logits = np.load(os.path.join(cfg.LOGITS_DIR, logits_fn))
    logits = np.transpose(logits, (1,2,0))
    softmax_vs = softmax(logits, axis=-1)
    
    img_suffix = os.listdir(cfg.GT_MASKS_DIR)[0].split(".")[-1]
    fn_prefix = logits_fn.split(".")[0]
    gt_masks = Image.open(os.path.join(cfg.GT_MASKS_DIR, fn_prefix + "." + img_suffix))
    gt_masks = np.asarray(gt_masks)

    args = [softmax_vs, gt_masks, fn_prefix]
    compute_metrics(args)
    

def load_data():
    """Reads your data from your data root folder and calculate metrics"""
    Path(cfg.COMPONENTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(cfg.METRICS_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Process {cfg.DATASET} data:") 
    logit_fns = sorted(os.listdir(cfg.LOGITS_DIR))
    process_map(process_data, logit_fns, max_workers=cfg.NUM_WORKERS)