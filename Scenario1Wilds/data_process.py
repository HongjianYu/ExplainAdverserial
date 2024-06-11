import sys

sys.argv = [""]
# sys.path.append("/Users/lindseywei/masksearch")
sys.path.append("/homes/gws/hjyu/MaskSearchDemo/Scenario1Wilds")

from masksearch import *
import argparse
import json
import pickle
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
# from utils import *
# from pytorch_grad_cam import (
#     AblationCAM,
#     EigenGradCAM,
#     GradCAM,
#     GradCAMPlusPlus,
#     HiResCAM,
#     LayerCAM,
#     RandomCAM,
# )
from pytorch_grad_cam.utils.image import show_cam_on_image
import wilds
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader
import shelve
import time

def data_process():

    # dir="/Users/lindseywei/masksearch/wilds/"
    dir = "/homes/gws/hjyu/MaskSearchDemo/Scenario1Wilds/"
    # Load the full dataset, and download it if necessary
    dataset = get_dataset(
        dataset="iwildcam",
        download=False,
        root_dir=dir
    )

    # Get the ID validation set
    id_val_data = dataset.get_subset(
        "id_val",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )

    ood_val_data = dataset.get_subset(
        "val",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )



    # Load from disk
    cam_map = shelve.open(dir + "id_ood_val_cam_map.shelve")
    with open(dir + "id_ood_val_pred.pkl", "rb") as f:
        pred_map = pickle.load(f)
    with open(dir + "id_ood_val_label.pkl", "rb") as f:
        label_map = pickle.load(f)



    id_total = 7314
    ood_total = 14961
    dataset_examples = []
    for distribution, image_total in zip(["id_val", "ood_val"], [id_total, ood_total]):
        for image_idx in range(1, 1 + image_total):
            dataset_examples.append(f"{distribution}_{image_idx}")

    hist_size = 16
    hist_edges = []
    bin_width = 256 // hist_size
    for i in range(1, hist_size):
        hist_edges.append(bin_width * i)
    hist_edges.append(256)

    available_coords = 64

    object_detection_map = load_object_region_index_in_memory(
        dataset_examples,
        f"{dir}id_ood_val_object_detection_map.shelve",
    )

    in_memory_index_suffix = np.load(
        f"{dir}id_ood_val_cam_hist_prefix_{hist_size}_in_memory_available_coords_{available_coords}_suffix.npy"
    )

    image_access_order = range(len(dataset_examples))

    return id_val_data, ood_val_data, label_map, pred_map, cam_map, object_detection_map, dataset_examples, in_memory_index_suffix, image_access_order