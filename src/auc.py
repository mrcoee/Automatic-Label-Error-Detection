import os
import sys
import logging
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import auc
from typing import Optional
from numba import njit
from PIL import Image
from glob import glob
from multiprocessing.dummy import Pool
from tqdm import tqdm

import labels
from config import cfg

@njit
def _split_by_ids(x: np.ndarray, y: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return (y.reshape(1, -1) == ids.reshape(-1, 1)) * x.reshape(1, -1)


@njit
def _faster_unique(x: np.ndarray):
    counts = np.bincount(x)
    uniq = np.nonzero(counts)[0]
    counts = counts[uniq]
    return uniq, counts


@njit
def _adjusted_ious(pr_labels: np.ndarray,
                   pr_segments: np.ndarray,
                   gt_labels: np.ndarray,
                   gt_segments: np.ndarray,
                   num_classes: int,
                   ignore_index: Optional[int] = None) -> np.ndarray:

    ids = np.arange(num_classes)
    if ignore_index is not None:
        ids = ids[ids != ignore_index]

        # The code below can be used to remove ignored areas from the prediction if this has not
        # already been done using the code currently in __init__
        # pr_labels = pr_labels.copy()
        # pr_labels.ravel()[gt_labels.ravel() == ignore_index] = ignore_index

    pr_split = _split_by_ids(pr_segments, pr_labels, ids).ravel()
    gt_split = _split_by_ids(gt_segments, gt_labels, ids).ravel()

    n_pr = pr_segments.max() + 1
    n_gt = gt_segments.max() + 1
    mult = max(n_pr, n_gt)

    # without numba, use the argument:
    #   pr_split + np.multiply(gt_split, mult, dtype=np.uint32)
    uniq, counts = _faster_unique(pr_split + gt_split * np.uint32(mult))

    div_, mod_ = np.divmod(uniq, mult)  # TODO: faster than (>>, &) ?
    union = np.bincount(mod_, weights=counts, minlength=n_pr)
    mask = div_ != 0
    inter = np.bincount(mod_[mask], weights=counts[mask], minlength=n_pr)

    # TODO: next line is a 'temporary hack' to fix double counting segments where the ground truth
    # segment is entirely covered by a prediction
    counts[mod_ != 0] = 0
    union += np.bincount(
        mod_[mask], weights=counts[np.searchsorted(uniq, div_[mask] * mult)], minlength=n_pr
    )

    union[union == 0] = np.nan

    return inter / union


def load_inputs(img_fn):
    e_map_img = Image.open(img_fn)
    pred_img = Image.open(img_fn.replace(cfg.ERROR_DIR, cfg.BENCHMARK_PROPOSAL_VIS_DIR).replace("label_errors", "proposals"))
    e_map = np.array(e_map_img)
    pred = np.array(pred_img)
    inps.append((e_map, pred))
    img_pbar.update(1)


def main():
    logging.basicConfig(filename=os.path.join("/home/marco/Automatic-Label-Error-Detection/benchmark_results.log"),
                    format='%(message)s',
                    filemode='w',
                    level=logging.INFO)

    global inps
    global img_pbar
    
    Dataset = getattr(labels, cfg.DATASET.capitalize())
    trainId2name = Dataset.trainId2name

    all_images = glob(os.path.join(cfg.ERROR_DIR, "*"))
    inps = []
    img_pbar = tqdm(total=len(all_images), desc='Loading images', file=sys.stdout, leave=False)
    img_pbar.refresh()
    img_pool = Pool(80)

    img_pool.map(load_inputs, all_images)
    img_pool.close()
    img_pool.join()

    results = {}

    for t in np.linspace(0, 1, 21):
        
        logging.info(f"Threshold: {t: .2f},\n\n")
        
        cls_results = [{"tp": 0, "fn": 0, "fp": 0, "precision": 0, "recall": 0, "f1": 0} for c in cfg.CLASS_IDS]
        tp = 0
        fp = 0
        fn = 0
        stat_pbar = tqdm(total=len(all_images), desc=f'Calculating stats to threshold {round(t, 2)}', file=sys.stdout, leave=False)
        structure = np.ones((3, 3), dtype=int)
        for item in inps:
            e_map = item[0]
            pred = item[1][:,:,0]
            cls_mask = item[1][:,:,1]
            
            class_e_map = np.zeros_like(e_map)
            pred_t = np.zeros_like(pred)
                
            for i, cls_id in enumerate(cfg.CLASS_IDS):
                
                class_e_map[:,:] = 0
                class_e_map[e_map==cls_id] = 1            

                pred_t[:,:] = 0
                pred_t[pred > t*255] = 1
                pred_t[cls_mask!=cls_id] = 0
                
                num_classes = 2
                e_segments, _ = label(class_e_map, structure)
                pred_segments, _ = label(pred_t, structure)

                res = _adjusted_ious(class_e_map, e_segments, pred_t, pred_segments, num_classes)
                cls_results[i]["tp"] += np.sum(res[1:] >= 0.25)
                cls_results[i]["fn"] += np.sum(res[1:] < 0.25)

                res = _adjusted_ious(pred_t, pred_segments, class_e_map, e_segments, num_classes)
                cls_results[i]["fp"] += np.sum(res[1:] < 0.25)
            stat_pbar.update(1)

        for i,_ in enumerate(cfg.CLASS_IDS):
            tp += cls_results[i]["tp"]
            fn += cls_results[i]["fn"]
            fp += cls_results[i]["fp"]
            
            if cls_results[i]["tp"] + cls_results[i]["fp"] > 0:
                cls_results[i]["precision"] = cls_results[i]["tp"] / (cls_results[i]["tp"] + cls_results[i]["fp"])
                
            if cls_results[i]["tp"] + cls_results[i]["fn"] > 0:
                cls_results[i]["recall"] = cls_results[i]["tp"] / (cls_results[i]["tp"] + cls_results[i]["fn"])
                
            if cls_results[i]["precision"] + cls_results[i]["recall"] > 0:
                cls_results[i]["f1"] = 2 * (cls_results[i]["precision"] * cls_results[i]["recall"]) / (cls_results[i]["precision"] + cls_results[i]["recall"])
            
            logging.info(f'Class: {trainId2name[cfg.CLASS_IDS[i]]},\n' \
                f'TP: {cls_results[i]["tp"]},\n' \
                f'FP: {cls_results[i]["fp"]},\n' \
                f'FN: {cls_results[i]["fn"]},\n' \
                f'Precision: {cls_results[i]["precision"]: .4f},\n' \
                f'Recall: {cls_results[i]["recall"]: .4f},\n' \
                f'f1: {cls_results[i]["f1"]: .4f}\n'
            )

        precision, f1, recall = 0, 0, 0

        if tp + fp > 0:
            precision = tp / (tp + fp)
        if tp + fn > 0:
            recall = tp / (tp + fn)
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            
        results[recall] = precision

        logging.info(f'\nOverall:\n' \
                    f'TP: {tp},\n' \
                    f'FP: {fp},\n' \
                    f'FN: {fn},\n' \
                    f'Precision: {precision: .4f},\n' \
                    f'Recall: {recall: .4f},\n' \
                    f'f1: {f1: .4f}\n' \
                    '---------------------------\n'
        )
            
    recall_list = []
    precision_list = []
    for key in sorted(results.keys()):
        recall_list.append(key)
        precision_list.append(results[key])
        
    recall_list.insert(0, 0)
    recall_list.append(1)
    precision_list.insert(0, 0)
    precision_list.append(precision_list[-1])
    auc_value = auc(recall_list, precision_list)
    logging.info(f'{auc_value=: .4f}')
    
    print('Job done!')


if __name__ == '__main__':
    main()

