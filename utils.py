### utils.py
# Functions for logging, loading weights, assign learning rates.
###

import os, sys
import logging
import datetime
import torch
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


def loop_iterable(iterable):
    while True:
        yield from iterable


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def parameters_in_module(module):
    return sum([p.numel() for p in module.parameters()])

def get_parameter_count(params):
    total_count = 0
    for pg in params:
        for module in pg["params"]:
            total_count += sum([m.numel() for m in module])
    return total_count

def assign_learning_rate(optimizer, lr=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes


def add_weight_decay(params, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in params:
        if not param.requires_grad: continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list: no_decay.append(param)
        else: decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


def get_logger(logdir, test=False):
    """Function to build the logger.
    """
    logger = logging.getLogger('mylogger')
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    if test: ts = "test"
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    file_hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_hdlr.setFormatter(formatter)
    logger.addHandler(file_hdlr)
    logger.propagate = False
    stdout_hdlr = logging.StreamHandler(sys.stdout)
    stdout_hdlr.setFormatter(formatter)
    logger.addHandler(stdout_hdlr)
    logger.setLevel(logging.INFO)
    return logger


def cvt2normal_state(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
    module state_dict inplace, i.e. removing "module" in the string.
    """
    return {k.partition("module.")[-1]:v for k,v in state_dict.items()}
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k.replace('module.', '')
    #     new_state_dict[name] = v
    # return new_state_dict

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def rand_bbox_withcenter(size, lam, cx, cy):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def get_rand_boxes(size, batchsize):

    lam = np.random.beta(1,1)
    bbx1, bby1, bbx2, bby2 = rand_bbox(size, lam)
    boxes = torch.Tensor([bbx1, bby1, bbx2, bby2])
    boxes = boxes.repeat(batchsize,1).long()

    return boxes


def get_cam_images(featmap, gradient, image_size, cam="grad_cam", threshold=None, return_masks=False):

    if cam == "grad_cam":
        cam_weights = gradient.sum(-1).sum(-1)[:,:,None,None]
        cam_activations = F.relu(featmap*cam_weights).sum(1) + 1e-6
    elif cam == "HiRes_gradcam":
        cam_activations = F.relu(featmap*gradient).sum(1) + 1e-6
    else:
        raise NotImplementedError

    ## normalize features
    per_batch_maximum = cam_activations.max(-1)[0].max(-1)[0]
    per_batch_maximum = per_batch_maximum[:,None,None]
    cam_activations_normalized = cam_activations/per_batch_maximum

    ## generate image from smaller map.
    cam_activations_normalized = cam_activations_normalized[:,None]
    cam_activations_resized = F.interpolate(cam_activations_normalized, image_size, mode="bilinear", align_corners=True).squeeze(1)

    ## obtain bounding box from mask
    if threshold is None:
        threshold = torch.from_numpy(np.random.uniform(0.6,0.8,size=(len(cam_activations_normalized),1,1))).to(cam_activations_normalized)
    
    cam_mask = (cam_activations_resized > threshold).float()
    try:
        cam_boxes = masks_to_boxes(cam_mask)
    except:
        import pdb; pdb.set_trace()

    if not return_masks:
        return cam_boxes

    return cam_boxes, cam_mask

def paste_boxes_on_image(boxes_bg, boxes_fg, size):

    """
    Paste boxes_fg on boxes_bg after adjusting for size
    """
    
    im_w, im_h = size

    ## find the center coordinates of the source image
    cx_bg,cy_bg = (boxes_bg[:,0] + boxes_bg[:,2])//2 , (boxes_bg[:,1] + boxes_bg[:,3])//2

    ## find the center coordinates of the target image
    cx_fg,cy_fg = (boxes_fg[:,0] + boxes_fg[:,2])//2 , (boxes_fg[:,1] + boxes_fg[:,3])//2

    ## find all directions widths of boxes to be pasted from second image
    wx_left = cx_fg - boxes_fg[:,0]
    wx_right = boxes_fg[:,2] - cx_fg
    
    wy_top = cy_fg - boxes_fg[:,1]
    wy_bottom = boxes_fg[:,3] - cy_fg
        
    ## find the box coordinates from source image, clipped within image sizes
    bbx1, bbx2, bby1, bby2 = (cx_bg-wx_left).clamp(max=im_h,min=0.) , \
                             (cx_bg+wx_right).clamp(max=im_h,min=0), \
                              (cy_bg-wy_top).clamp(max=im_w,min=0), \
                            (cy_bg+wy_bottom).clamp(max=im_w, min=0)
    
    ## recompute all directions widths in second image since clipping might reduce the width
    wx_left = cx_bg - bbx1
    wx_right = bbx2 - cx_bg
    
    wy_top = cy_bg - bby1
    wy_bottom = bby2 - cy_bg
    
    ## recompute second image box coordinates. These are guaranteed to be within image sizes
    ## because new widhths can only be less than older widths
    bbx1_fg, bbx2_fg, bby1_fg, bby2_fg = (cx_fg-wx_left) , \
                                             (cx_fg+wx_right), \
                                              (cy_fg-wy_top), \
                                            (cy_fg+wy_bottom)
    
    box_coords_bg = torch.stack([bbx1, bby1, bbx2, bby2], dim=1).long()
    box_coords_fg = torch.stack([bbx1_fg, bby1_fg, bbx2_fg, bby2_fg], dim=1).long()
    
    return box_coords_bg, box_coords_fg

def area_of_bbox(boxes):

    return (boxes[:,2] - boxes[:,0])*(boxes[:,3] - boxes[:,1])


