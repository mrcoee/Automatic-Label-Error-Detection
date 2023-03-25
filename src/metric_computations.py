import os 
import pickle
import numpy as np

from config import cfg
from metrics import  compute_metrics_components

def compute_metrics(data):
    softmax_vs, gt_mask, fn_prefix = data 
    metrics, components = compute_metrics_components(softmax_vs.copy(), gt_mask.copy())
    np.save(os.path.join(cfg.COMPONENTS_DIR, fn_prefix + "_components.npy"), components)
    pickle.dump(metrics, open(os.path.join(cfg.METRICS_DIR, fn_prefix + "_metrics.p"), "wb"))
