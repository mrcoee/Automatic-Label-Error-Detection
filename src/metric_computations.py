import os 
import pickle
import numpy as np

from config import cfg
from metrics import  compute_metrics_components

def compute_metrics(data):
    softmax_vs, gt_mask, fn_prefix, perturbed_mask = data 
    target_mask = gt_mask if not cfg.BENCHMARK else perturbed_mask
    metrics, components = compute_metrics_components(softmax_vs.copy(), target_mask.copy())
    np.save(os.path.join(cfg.COMPONENTS_DIR, fn_prefix + "_components.npy"), components)
    
    if perturbed_mask is not None:
        diff_mask = np.zeros_like(gt_mask, dtype="uint8")
        diff_mask[gt_mask != perturbed_mask] = 1
        np.save(os.path.join(cfg.DIFF_DIR, fn_prefix + "_diff_map.npy"), diff_mask)
    
    pickle.dump(metrics, open(os.path.join(cfg.METRICS_DIR, fn_prefix + "_metrics.p"), "wb"))
