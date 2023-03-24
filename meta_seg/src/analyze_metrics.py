import pickle
import os
import heapq
import numpy as np 

from PIL import Image
from pathlib import Path
from multiprocessing import Pool
from scipy.interpolate import interp1d

from regression import classification_fit_and_predict, classification_fit_and_predict
from visualization import proposal_vis

from config import cfg

def concatenate_metrics(metric_fns):
    metrics = None
    for fn in metric_fns:
        print("Concatenate file {}\r".format(fn))
        if metrics == None:
            metrics = pickle.load(open(os.path.join(cfg.METRICS_DIR, fn), "rb"))
        else:   
            m = pickle.load(open(os.path.join(cfg.METRICS_DIR, fn), "rb"))
            for key in metrics:
                metrics[key].extend(m[key])

    print("\nConnected components:", len(metrics['iou']) )
    print("Non-empty connected components:", np.sum( np.asarray(metrics['S_in']) != 0) )

    return metrics


def metrics_to_nparray(metrics, names, normalize=False, non_empty=True):

    I = range(len(metrics['S_in']))

    if non_empty == True:
        I = np.asarray(metrics['S_in']) > 0

    M = np.asarray([np.asarray(metrics[name])[I] for name in names])
    MM = M.copy()

    if normalize == True:
        for i in range(M.shape[0]):
            if names[i] != "class":
                M[i] = (np.asarray(M[i]) - np.mean(MM[i], axis=-1 )) / (np.std(MM[i], axis=-1) + 1e-10)
    M = np.squeeze(M.T)

    return M


def metrics_to_dataset(metrics, x_names, prob_names, non_empty):

    Xa = metrics_to_nparray(metrics, x_names, normalize=True, non_empty=non_empty)
    if len(Xa.shape) == 1:
        Xa = Xa.reshape(-1,1)
    if len(prob_names) > 0:
        classes = metrics_to_nparray(metrics, prob_names, normalize=True, non_empty=non_empty)
    else:
        classes = []
    perturbed_ya = metrics_to_nparray(metrics, ["iou"], normalize=False, non_empty=non_empty)
    perturbed_y0a = metrics_to_nparray(metrics, ["iou0"], normalize=False, non_empty=non_empty)

    return Xa, classes, perturbed_ya, perturbed_y0a


def evaluate():
    """
    Train the meta seg model on the calulated metrics and use it to predict label errors
    """

    #use a random seed so that the results are reproducible
    if "random_seed" in cfg:
        np.random.seed(cfg.random_seed)
            
    metric_fns = sorted(os.listdir(cfg.METRICS_DIR))
    inputs_len = len(metric_fns)
    
    assert cfg.SPLIT_RATIO >= 0 and cfg.SPLIT_RATIO <= 1, \
        f"SPLIT_RATIO between 0 and 1 expected, got: {cfg.SPLIT_RATIO}"
    
    assert int(cfg.SPLIT_RATIO * inputs_len) > 0, \
        f"No training input with a split ratio of {cfg.SPLIT_RATIO}. Set a higher value"
    
    np.random.shuffle(metric_fns)
    # Split and prepare the metrics for the model training
    train_metrics = concatenate_metrics(metric_fns[0:int(cfg.SPLIT_RATIO * inputs_len)])
    val_metrics = concatenate_metrics(metric_fns[int(cfg.SPLIT_RATIO * inputs_len):])
    print(f"Concatenated metrics of {inputs_len} images\n")
    
    m = interp1d([0, 19], [-4.2, 0.8])
    lambdas = [10 ** m(i).item() for i in range(20)]
    nclasses = np.max(train_metrics["class"]) + 1

    x_names = sorted([key for key in train_metrics if key not in ["class","iou","iou0"] and "cprob" not in key])
    prob_names = ["cprob" + str(i) for i in range(nclasses) if "cprob" + str(i) in train_metrics]

    Xa_train, train_classes, _, perturbed_y0a_train = metrics_to_dataset(train_metrics, x_names, prob_names, non_empty=True)
    if len(train_classes) > 0:
        Xa_train = np.concatenate((Xa_train, train_classes), axis=-1)

    Xa_val, val_classes, perturbed_ya_val, perturbed_y0a_val = metrics_to_dataset(val_metrics, x_names, prob_names, non_empty=True)
    if len(val_classes) > 0:
        Xa_val = np.concatenate((Xa_val, val_classes), axis=-1)

    if len(prob_names) > 0:
        x_names.extend(prob_names)
    
    print("Fitting regression model...")
    perturbed_y0a_val_pred, _ = classification_fit_and_predict(Xa_train, perturbed_y0a_train, lambdas, Xa_val)
    acc = np.mean(np.argmax(perturbed_y0a_val_pred,axis=-1)==perturbed_y0a_val)
    print(f"Accuracy of model: {100*acc:.1f}%\n")

    # Find the k most probable (according to the meta seg) label errors and push them into a min heap. The label error
    # with the lowest confidence is the first element in the list.
    # k is defined in the config file by cfg.NUM_PRPOSALS
    k = 0
    meta_seg_vis_args = []
    segment_heap = []
    val_fns = [fn.split("_")[0] for fn in metric_fns[int(cfg.SPLIT_RATIO * inputs_len):]]
    print("Searching for label errors...")
    for fn in val_fns:
        segments = np.load(os.path.join(cfg.COMPONENTS_DIR, fn + "_components.npy"))
        inf_ar = np.asarray(Image.open(os.path.join(cfg.INFERENCE_OUTPUT_DIR, fn + ".png")), dtype="uint8")
        iou_ar = np.zeros_like(segments)
        pos_segments = np.unique(segments[segments >= 0])
        for i, seg_ind in enumerate(pos_segments):
            cls_id = inf_ar[np.abs(segments) == seg_ind][0]

            if perturbed_ya_val[k+i] == 0: 
                if perturbed_y0a_val_pred[k+i, 0] > 0 and \
                    cls_id in cfg.CLASS_IDS:
                    
                    iou_ar[np.abs(segments) == seg_ind] = perturbed_y0a_val_pred[k+i, 0]*255
                    seg_size = len(np.where(np.abs(segments) == seg_ind)[0])

                    if seg_size >= cfg.MIN_ERROR_SIZE and len(segment_heap) < cfg.NUM_PRPOSALS:
                        heapq.heappush(segment_heap, (perturbed_y0a_val_pred[k+i, 0], fn, seg_ind, cls_id))

                    elif seg_size >= cfg.MIN_ERROR_SIZE and segment_heap[0][0] < perturbed_y0a_val_pred[k+i, 0] and len(segment_heap) == cfg.NUM_PRPOSALS:
                        heapq.heappop(segment_heap)
                        heapq.heappush(segment_heap, (perturbed_y0a_val_pred[k+i, 0], fn, seg_ind, cls_id))
                        
        meta_seg_vis_args.append((fn, perturbed_ya_val[k:k+len(pos_segments)], perturbed_y0a_val_pred[k:k+len(pos_segments)]))
        k += len(pos_segments)
      
    #Visualize label errors
    Path(cfg.ERROR_PROPOSAL_DIR).mkdir(parents=True, exist_ok=True)
    with Pool(cfg.NUM_WORKERS) as p:
        print("Visualize error proposals...")
        p.starmap(proposal_vis, segment_heap)
    
