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
    
    gt_img_suffix = os.listdir(cfg.GT_MASKS_DIR)[0].split(".")[-1]
    fn_prefix = logits_fn.split(".")[0]
    gt_masks = Image.open(os.path.join(cfg.GT_MASKS_DIR, fn_prefix + "." + gt_img_suffix))
    gt_masks = np.asarray(gt_masks)
    
    perturbed_mask = None
    if cfg.BENCHMARK:
        pgt_img_suffix = os.listdir(cfg.PERTURBED_MASKS_DIR)[0].split(".")[-1]
        perturbed_mask = Image.open(os.path.join(cfg.PERTURBED_MASKS_DIR, fn_prefix + "." + pgt_img_suffix))
        perturbed_mask = np.asarray(perturbed_mask)

    args = [softmax_vs, gt_masks, fn_prefix, perturbed_mask]
    compute_metrics(args)


def load_data():
    """Reads your data from your data root folder and calculate metrics"""
    Path(cfg.COMPONENTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(cfg.METRICS_DIR).mkdir(parents=True, exist_ok=True)
    
    if cfg.BENCHMARK:
        Path(cfg.DIFF_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Process {cfg.DATASET} data:") 
    logit_fns = sorted(os.listdir(cfg.LOGITS_DIR))
    process_map(process_data, logit_fns, max_workers=cfg.NUM_WORKERS)