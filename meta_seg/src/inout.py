import os
import h5py 
import math
import numpy as np

from config import cfg


def save_data(args):
    """Saved prepared data as hdf5 file"""
    softmax_vs, gt_masks, fn_prefix = args[:3]
    
    file_path = os.path.join(cfg.DATA_DIR, fn_prefix + ".hdf5")
    f = h5py.File( file_path, "w")
    f.create_dataset("probabilities", data=softmax_vs)
    f.create_dataset("ground_truth", data=gt_masks)
    f.close()


def load_hdf5_data(fn):
    """Load prepared data from hdf5 file"""
    h5py_fn = fn + ".hdf5"
    f_probs = h5py.File(os.path.join(cfg.DATA_DIR, h5py_fn), "r")

    softmax_vs = np.asarray(f_probs['probabilities'])
    gt_masks = np.asarray(f_probs['ground_truth'])
    softmax_vs = np.squeeze(softmax_vs)
    gt_masks = np.squeeze(gt_masks)
    
    output = [softmax_vs, gt_masks]
    return output
