
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label

from config import cfg

import labels


def benchmark_vis(iou_ar, fn):
    pred_pgt_ar = np.zeros_like(iou_ar)
    structure = np.ones((3, 3), dtype=int)
    segment_ar = label(iou_ar, structure)[0]
    for idx in np.unique(segment_ar):
        if idx > 0:
            segment = np.where(segment_ar == idx)
            if len(segment[0]) >= cfg.MIN_ERROR_SIZE:
                pred_pgt_ar[segment] = iou_ar[segment]

    inf_mask_suffix = os.listdir(cfg.GT_MASKS_DIR)[0].split(".")[-1]
    inf_ar = np.asarray(Image.open(os.path.join(cfg.INFERENCE_OUTPUT_DIR, fn + "." + inf_mask_suffix)), dtype="uint8")

    result = np.zeros(pred_pgt_ar.shape + (3,), dtype='uint8')
    result[:,:,0] = pred_pgt_ar
    result[:,:,1] = inf_ar

    Image.fromarray(result).save(os.path.join(cfg.BENCHMARK_PROPOSAL_VIS_DIR, fn + "_proposals.png"))
    
    
def label_error_vis(fn):
    diff_mask = np.load(os.path.join(cfg.DIFF_DIR, fn + "_diff_map.npy"))
    img_suffix = os.listdir(cfg.GT_MASKS_DIR)[0].split(".")[-1]
    gt_mask = np.array(Image.open(os.path.join(cfg.GT_MASKS_DIR, fn + "." + img_suffix)))

    structure = np.ones((3, 3), dtype=int)
    error_segments = label(diff_mask, structure)[0]
    for idx in np.unique(error_segments):
        if idx > 0:
            segment = np.where(error_segments == idx)
            if len(segment[0]) < cfg.MIN_ERROR_SIZE:
                diff_mask[segment] = 0

    diff_mask[diff_mask==1] = gt_mask[diff_mask==1]
    Image.fromarray(diff_mask).save(os.path.join(cfg.ERROR_DIR, fn + "_label_errors." + img_suffix))


def proposal_vis(iou, fn, seg_id, cls_id):
    """Visualization of one potential label error in image fn"""
    inf_mask_suffix = os.listdir(cfg.GT_MASKS_DIR)[0].split(".")[-1]
    inf_ar = np.asarray(Image.open(os.path.join(cfg.INFERENCE_OUTPUT_DIR, fn + "." + inf_mask_suffix)), dtype="uint8")
    
    gt_mask_suffix = os.listdir(cfg.GT_MASKS_DIR)[0].split(".")[-1]
    net_input_suffix = os.listdir(cfg.NET_INPUT_DIR)[0].split(".")[-1]
    gt_mask = np.array(Image.open(os.path.join(cfg.GT_MASKS_DIR, fn + "." + gt_mask_suffix)))
    net_input = np.array(Image.open(os.path.join(cfg.NET_INPUT_DIR, fn + "." + net_input_suffix)))
    
    Dataset = getattr(labels, cfg.DATASET.capitalize())
    trainId2color = Dataset.trainId2color
    trainId2name = Dataset.trainId2name
    num_classes = Dataset.num_classes

    gt_rgb = np.zeros(inf_ar.shape + (3,), dtype="uint8")
    pred_rgb = np.zeros(inf_ar.shape + (3,), dtype="uint8")
    for id in range(num_classes):
        gt_rgb[gt_mask==id] = trainId2color[id]
        pred_rgb[inf_ar==id] = trainId2color[id]

    rgb_blend_img = np.copy(net_input)
    components = np.load(os.path.join(cfg.COMPONENTS_DIR, fn + "_components.npy"))
    seg_idx = np.where(np.abs(components)==seg_id)

    bbox_0 = (max(np.min(seg_idx[1])-15, 0), max(np.min(seg_idx[0])-15, 0))
    bbox_1 = (min(np.max(seg_idx[1])+15, rgb_blend_img.shape[1]), min(np.max(seg_idx[0])+15, rgb_blend_img.shape[1]))

    rgb_blend_img[seg_idx] = 0.4*rgb_blend_img[seg_idx] + 0.6*pred_rgb[seg_idx]
    
    img_shape = rgb_blend_img.shape[0]

    net_input = Image.fromarray(net_input)
    rgb_blend_img = Image.fromarray(rgb_blend_img)
    gt_rgb_img = Image.fromarray(gt_rgb)
    pred_rgb_img = Image.fromarray(pred_rgb)
    
    font = ImageFont.truetype("arial.ttf", size=int(0.05*img_shape))

    draw = ImageDraw.Draw(rgb_blend_img)
    draw.rectangle([bbox_0, bbox_1], outline='red', width=4)
    draw.text((0, 0), "Predicted Label: " + trainId2name[int(cls_id)], fill='red', font=font)

    draw = ImageDraw.Draw(net_input)
    draw.rectangle([bbox_0, bbox_1], outline='red', width=4)
    draw.text((0, 0), "Ground Truth", fill='red', font=font)

    draw = ImageDraw.Draw(gt_rgb_img)
    draw.rectangle([bbox_0, bbox_1], outline='white', width=4)

    draw = ImageDraw.Draw(pred_rgb_img)
    draw.rectangle([bbox_0, bbox_1], outline='white', width=4)

    out_img = Image.new('RGB', (2*rgb_blend_img.width, 2*rgb_blend_img.height))
    out_img.paste(net_input, (0, 0))
    out_img.paste(rgb_blend_img, (0, net_input.height))
    out_img.paste(gt_rgb_img, (net_input.width, 0))
    out_img.paste(pred_rgb_img, (net_input.width, net_input.height))

    saved = False 
    i = 0
    while not saved:
        try:
            open(os.path.join(cfg.ERROR_PROPOSAL_DIR, fn + f"_proposal_{i}." + net_input_suffix))
            i += 1
        except IOError:
            out_img.save(os.path.join(cfg.ERROR_PROPOSAL_DIR, fn + f"_proposal_{i}." + net_input_suffix))
            saved = True