import os 
import pickle

import numpy as np

from pathlib import Path
from tqdm.contrib.concurrent import process_map
from config import cfg
from inout import load_hdf5_data
from metrics import  compute_metrics_components




def compute_metrics(h5py_fn):

    data = load_hdf5_data(h5py_fn) 
    softmax_vs, gt_mask = data[:2]
        
    target_mask = gt_mask 
    metrics, components = compute_metrics_components(softmax_vs, target_mask)
    
    fn_prefix = h5py_fn.split(".")[0]
    np.save(os.path.join(cfg.COMPONENTS_DIR, fn_prefix + "_components.npy"), components)

    pickle.dump(metrics, open(os.path.join(cfg.METRICS_DIR, fn_prefix + "_metrics.p"), "wb"))


def compute_metrics_mp():
    """Calculate metrics of the prepared input data"""
    Path(cfg.METRICS_DIR).mkdir(parents=True, exist_ok=True)
    Path(cfg.COMPONENTS_DIR).mkdir(parents=True, exist_ok=True)

    print("calculating statistics")
    fns = [fn.split(".")[0] for fn in os.listdir(cfg.DATA_DIR)]
    process_map(compute_metrics, fns, max_workers=cfg.NUM_WORKERS)