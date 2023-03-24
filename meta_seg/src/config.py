import os
from easydict import EasyDict as edict

cfg = edict()

cfg.DATASET = "Cityscapes"      # Availabale in labels.py are Carla, Cityscape, Coco, PascalVOC
cfg.DATASET_DIR = "/home/marco/data/nvidia_val" #Root folder of your dataset

# Required input paths
cfg.NET_INPUT_DIR = os.path.join(cfg.DATASET_DIR, "net_input") # Required for visualizations
cfg.INFERENCE_OUTPUT_DIR = os.path.join(cfg.DATASET_DIR, "inference_output") # Segmentation masks generated by the net
cfg.GT_MASKS_DIR = os.path.join(cfg.DATASET_DIR, "gt_masks") # Ground truth segmentation masks
cfg.LOGITS_DIR = os.path.join(cfg.DATASET_DIR, "logits") # Ground truth segmentation masks

# Paths to store intermediate results. Only the root has to be set
cfg.INTERMEDIATE_DIR = "/home/marco/labelerror_detection/meta_seg/intermediate_results"

cfg.DATA_DIR = os.path.join(cfg.INTERMEDIATE_DIR, "data")
cfg.COMPONENTS_DIR = os.path.join(cfg.INTERMEDIATE_DIR, "components")  
cfg.METRICS_DIR = os.path.join(cfg.INTERMEDIATE_DIR, "metrics")  

# Visualization paths
cfg.VISUALIZATIONS_DIR = "/home/marco/labelerror_detection/meta_seg/visualizations"

# cfg.LABEL_ERROR_DIR = os.path.join(cfg.VISUALIZATIONS_DIR, "label_error_vis")
cfg.ERROR_PROPOSAL_DIR = os.path.join(cfg.VISUALIZATIONS_DIR, "error_proposals")
cfg.META_SEG_VIS_DIR = os.path.join(cfg.VISUALIZATIONS_DIR, "meta_seg")

cfg.NUM_WORKERS = 4 # Number of multiprocessing workers
cfg.random_seed = 1

cfg.SPLIT_RATIO = 0.5 # Value between 0 and 1. Determine how much of the dataset is used to train meta Seg.
cfg.CLASS_IDS = [6, 7, 11, 12, 13, 14, 15, 17, 18] # Class ids of classes in which we search for label errors. E.g. Carla [2, 7, 9, 13] Cityscapes [6, 7, 11, 12, 13, 14, 15, 17, 18] #Pascal list(range(1, 21)) #Coco list(range(1, 93)) 
cfg.MIN_ERROR_SIZE = 100 # Defines the minimum amount of pixel error proposal must have
cfg.NUM_PRPOSALS = 100 # Maximum amount of error proposals

