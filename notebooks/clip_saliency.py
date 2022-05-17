import torch
import torch.nn.functional as F
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("RN50", device=device)

import numpy as np
from torchray.attribution.grad_cam import grad_cam
import streamlit as st
import matplotlib.pyplot as plt

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.transforms.functional as tr_F
from torchvision.transforms import InterpolationMode

rgb_mean = (0.48145466, 0.4578275, 0.40821073)
rgb_std = (0.26862954, 0.26130258, 0.27577711)
n_px = 224

def norm_tensor_to_np(arr, mean, std):
    """
    Parameters:
        - cfg: config node
        - arr: normalized floating point torch.Tensor of shape (3, H, W)
    """
    assert isinstance(arr, torch.Tensor)
    assert arr.shape[0] == 3
    assert len(arr) == 3
    ori_rgb_np = np.array(arr.permute((1, 2, 0)).cpu()) # H x W x 3
    ori_rgb_np = (ori_rgb_np * std) + mean
    assert ori_rgb_np.max() <= 1.1, "Max is {}".format(ori_rgb_np.max())
    ori_rgb_np[ori_rgb_np >= 1] = 1
    arr = (ori_rgb_np * 255).astype(np.uint8)
    return arr

# A generalized imshow helper function which supports displaying (CxHxW) tensor
def generalized_imshow(arr, mean=rgb_mean, std=rgb_std):
    '''
    Parameters
        - cfg: root yacs node of the YAML file
        - arr:
            normalized numpy array
            unnormalized pytorch tensor
    '''
    if isinstance(arr, torch.Tensor) and arr.shape[0] == 3:
        # TODO: check tensor type (float) to denormalize.
        arr = norm_tensor_to_np(arr, mean, std)
    plt.imshow(arr)
    plt.show()

def _convert_image_to_rgb(image):
    return image.convert("RGB")

# Original transform in CLIP
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize(rgb_mean, rgb_std),
    ])

def compute_params(W, H, n_px):
    '''
    Params
        - W: original width
        - H: original height
        - n_px: target square image width
    '''
    if W > H:
        target_H = int(n_px * (H * 1. / W))
        target_W = n_px
        H_pad = n_px - target_H
        W_pad = 0
    else:
        target_H = n_px
        target_W = int(n_px * (W * 1. / H))
        H_pad = 0
        W_pad = n_px - target_W
    left_pad = W_pad // 2
    right_pad = W_pad - left_pad
    top_pad = H_pad // 2
    bot_pad = H_pad - top_pad
    
    return target_H, target_W, left_pad, right_pad, top_pad, bot_pad

def resize_long_edge_to_res_and_pad(n_px):
    # Function factory
    def ret_func(image):
        W, H = image.size
        target_H, target_W, left_pad, right_pad, top_pad, bot_pad = compute_params(W, H, n_px)
        image = tr_F.resize(image, (target_H, target_W), interpolation=InterpolationMode.BICUBIC)
        # left, top, right and bottom borders respectively
        image = tr_F.pad(image, (left_pad, top_pad, right_pad, bot_pad))
        return image
    return ret_func

# Image preserving transform
def consistent_transform(n_px):
    return Compose([
        resize_long_edge_to_res_and_pad(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

preprocess=consistent_transform(n_px)

def get_saliency_map_helper(transformed_img, text_queries, restore_size=None):
    assert isinstance(text_queries, list)
    assert len(text_queries) == 1, "Multiple lengths not supported yet"

    text = clip.tokenize(text_queries).to(device)

    with torch.no_grad():
        image_features = model.encode_image(transformed_img)
        text_features = model.encode_text(text)
        image_features_norm = image_features.norm(dim=-1, keepdim=True)
        image_features_new = image_features / image_features_norm
        text_features_norm = text_features.norm(dim=-1, keepdim=True)
        text_features_new = text_features / text_features_norm
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features_new @ text_features_new.t()
        probs = logits_per_image.softmax(dim=-1).cpu().numpy().tolist()

    layer = st.sidebar.selectbox("select saliency layer", ['layer4.2.relu'], index=0)

    for i in range(len(text_queries)):
        text_prediction = (text_features_new[[i]] * image_features_norm)
        saliency = grad_cam(model.visual, transformed_img.type(model.dtype), text_prediction, saliency_layer=layer)
        if restore_size is not None:
            target_H, target_W, left_pad, right_pad, top_pad, bot_pad =\
                        compute_params(restore_size[1], restore_size[0], n_px)
            # restore to input to CLIP
            saliency = F.interpolate(saliency, (n_px, n_px), mode='bilinear')
            # cut padding portion
            saliency = saliency[:,:,top_pad:n_px-bot_pad,left_pad:n_px-right_pad]
            saliency = F.interpolate(saliency, restore_size, mode='bilinear')

    return saliency[0][0,].detach().cpu().numpy().astype(np.float32)

def get_saliency_map(image, text_queries):
    if isinstance(image, str): # path
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    W, H = image.size
    image = preprocess(image).unsqueeze(0).to(device)
    ret = get_saliency_map_helper(image, text_queries, restore_size=(H, W))
    return ret