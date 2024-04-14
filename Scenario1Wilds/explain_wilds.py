# %%

# NOTE: see https://github.com/microsoft/vscode-jupyter/issues/1837 for sys.argv = [''] below
import sys

sys.argv = [""]
sys.path.append("/Users/lindseywei/masksearch")
                
from masksearch.masksearch import *
import argparse
import json
import pickle
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from utils import *
from pytorch_grad_cam import (
    AblationCAM,
    EigenGradCAM,
    GradCAM,
    GradCAMPlusPlus,
    HiResCAM,
    LayerCAM,
    RandomCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image
import wilds
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader
import shelve
import time


# Load the full dataset, and download it if necessary
dataset = get_dataset(
    dataset="iwildcam",
    download=True,
    root_dir="/Users/lindseywei/masksearch/wilds/"
)

# path = "/data/explain_wilds/models/best_model.pth"
# state = torch.load(path)["algorithm"]
# # print(state.keys())
# state_dict = {}
# for key in list(state.keys()):
#     state_dict[key.replace('model.', '')] = state[key]

# parser = argparse.ArgumentParser()
# config = parser.parse_args()
# with open('/home/ubuntu/model_selection/wilds/wilds_config.txt', 'r') as f:
#     config.__dict__ = json.load(f)

# model = initialize_model(config, 182)
# model.load_state_dict(state_dict)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()

"""
{'train': 'Train', 'val': 'Validation (OOD/Trans)',
    'test': 'Test (OOD/Trans)', 'id_val': 'Validation (ID/Cis)',
    'id_test': 'Test (ID/Cis)'}
"""

# Get the ID validation set
id_val_data = dataset.get_subset(
    "id_val",
    transform=transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    ),
)
id_val_loader = get_eval_loader("standard", id_val_data, batch_size=16)

ood_val_data = dataset.get_subset(
    "val",
    transform=transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    ),
)
ood_val_loader = get_eval_loader("standard", ood_val_data, batch_size=16)

# Load from disk
cam_map = shelve.open("./id_ood_val_cam_map.shelve")
with open("./id_ood_val_pred.pkl", "rb") as f:
    pred_map = pickle.load(f)
with open("./id_ood_val_label.pkl", "rb") as f:
    label_map = pickle.load(f)

cam_size_y = 448
cam_size_x = 448

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
print(hist_edges)
cam_size_y = 448
cam_size_x = 448

available_coords = 64

object_detection_map = load_object_region_index_in_memory(
    dataset_examples,
    "./id_ood_val_object_detection_map.shelve",
)

in_memory_index_suffix = np.load(
    f"./id_ood_val_cam_hist_prefix_{hist_size}_in_memory_available_coords_{available_coords}_suffix.npy"
)

# %%
# Filter query benchmark setup

toy_examples = []
for distribution, image_total in zip(["id_val", "ood_val"], [id_total, ood_total]):
    for image_idx in range(1, 1 + image_total):
        toy_examples.append(f"{distribution}_{image_idx}")

region = (50, 50, 150, 150)
threshold = 0.6
v = 10000
region_area_threshold = None
grayscale_threshold = int(threshold * 255)

# %%
# MaskSearch (vanilla): filter query

start = time.time()
count, area_images = get_images_based_on_area_filter("wilds", cam_map, object_detection_map, bin_width, hist_size, cam_size_y, cam_size_x, toy_examples, threshold, region, v, in_memory_index_suffix, region_area_threshold=region_area_threshold, ignore_zero_area_region=True, reverse=False, visualize=False, available_coords=available_coords, compression=None)  # type: ignore
end = time.time()
print("Skipped images:", count)
print("Skipped images:", sum(count), "out of", len(toy_examples))
print("Skipping ratio:", sum(count) / len(toy_examples))
print("(MaskSearch vanilla) Query time (cold cache):", end - start)

image_ids = sorted([x[1] for x in area_images])
print(image_ids)
# print(area_images)
# Write area_images to file
# images = sorted([x[1] for x in area_images])
# with open("images.txt", "w") as f:
#     f.write(str(images))

# %%
# Naive: filter query

start = time.time()
area_images = naive_get_images_satisfying_filter(
    cam_map,
    object_detection_map,
    cam_size_y,
    cam_size_x,
    toy_examples,
    threshold,
    region,
    v,
    region_area_threshold=region_area_threshold,
    ignore_zero_area_region=True,
    compression=None,
)
end = time.time()
print("(NumPy) Query time (cold cache):", end - start)

# print(area_images)
# images = sorted([x[1] for x in area_images])
# with open("images.std", "w") as f:
#     f.write(str(images))

# %%
# Setup for top-k subregion query

toy_examples = []
for distribution, image_total in zip(["id_val", "ood_val"], [id_total, ood_total]):
    for image_idx in range(1, 1 + image_total):
        toy_examples.append(f"{distribution}_{image_idx}")



region_area_threshold = 5000
region = (150, 150, 150, 150)
threshold = 0.8
# x, y, w, h
reverse = True
k = 25
selected_images = ["id_val_145", "id_val_146", "id_val_147", "id_val_148", "id_val_149", "id_val_150", "id_val_151", "id_val_152",
                    "id_val_153"]
object_region = []

for i in range(len(selected_images)):
    image_id = selected_images[i]
    x, y, w, h = get_object_region(
        object_detection_map, cam_size_y, cam_size_x, image_id
    )
    object_region.append([x,y,w,h])
    #mask = cam_map[image_id]
    #image = from_input_to_image(image_map[image_id])
    #cam_image = show_cam_on_image(image, mask, use_rgb=True, image_weight=0.)

print(object_region)

# %%
# MaskSearch: top-k subregion query processing with in-memory index without optimization

image_access_order = range(len(selected_images))

start = time.time()
count, images = get_max_area_in_subregion_in_memory_version(
    "wilds",
    (id_val_data, ood_val_data),
    label_map,
    pred_map,
    cam_map,
    object_detection_map,
    bin_width,
    cam_size_y,
    cam_size_x,
    #toy_examples,
    selected_images,
    threshold,
    #range,
    object_region,
    in_memory_index_suffix,
    image_access_order,
    early_stoppable=False,
    k=k,
    region_area_threshold=region_area_threshold,
    ignore_zero_area_region=True,
    reverse=True,
    visualize=True,
    available_coords=available_coords,
    compression=None,
)
end = time.time()
print("Skipped images:", count)
# print("Skipped images:", sum(count), "out of", len(toy_examples))
# print("Skipping ratio:", sum(count) / len(toy_examples))
print("(MaskSearch vanilla) Query time (cold cache):", end - start)
print("******")

# %%
# NumPy: top-k subregion query processing

start = time.time()
images_std = naive_get_max_metric(
    "wilds",
    (id_val_data, ood_val_data),
    cam_map,
    object_detection_map,
    label_map,
    pred_map,
    cam_size_y,
    cam_size_x,
    toy_examples,
    threshold=threshold,
    region=region,
    k=k,
    region_area_threshold=region_area_threshold,
    ignore_zero_area_region=True,
    compression=None,
    reverse=reverse,
    visualize=True,
)
end = time.time()
print("(NumPy) Query time (cold cache): ", end - start)


# %%
# Q4: Returns top-25 images with largest mean(CT_PX(mask, roi, (lb, ub)) (groupby image_id) for masks of two models, roi = object, (lb, ub) = (0.8, 1.0)
toy_examples = []
for distribution, image_total in zip(["id_val", "ood_val"], [id_total, ood_total]):
    for image_idx in range(1, 1 + image_total):
        toy_examples.append(f"{distribution}_{image_idx}")
region = "object"
thresholds = [0.8, 0.8]
k = 25
region_area_threshold = 10000
reverse = False

# %%
cam_maps = [
    cam_map,
    shelve.open("/data/explain_wilds/shelves/duplicate_10_id_ood_val_cam_map.shelve"),
]

in_memory_index_suffixes = [
    in_memory_index_suffix,
    np.load(
        f"/data/explain_wilds/npy/duplicate_10_id_ood_val_cam_hist_prefix_{hist_size}_in_memory_available_coords_{available_coords}_suffix.npy"
    ),
]

# %%
start = time.time()
res = get_max_udf_in_sub_region_in_memory(
    "wilds",
    get_max_mean_in_area_across_models_in_memory,
    object_detection_map,
    cam_size_x,
    cam_size_y,
    bin_width,
    toy_examples,
    cam_maps,
    thresholds,
    in_memory_index_suffixes,
    region=region,
    k=25,
    region_area_threshold=region_area_threshold,
    ignore_zero_area_region=True,
    reverse=False,
    visualize=True,
    available_coords=available_coords,
    compression=None,
)
end = time.time()
print("MaskSearch (vanilla) query time for max-mean aggregation queries:", end - start)
# %%
# NumPy: max-mean aggregation query

start = time.time()
std = naive_get_max_udf(
    get_area_mean_map,
    object_detection_map,
    cam_size_y,
    cam_size_x,
    toy_examples,
    cam_maps,
    thresholds,
    region,
    k,
    region_area_threshold,
    ignore_zero_area_region=True,
    compression=None,
    reverse=False,
    visualize=False,
)
end = time.time()
print("NumPy query time for max-mean aggregation queries:", end - start)

# %%
# Q4: Returns top-25 images with largest mean(CT_PX(mask, roi, (lb, ub)) (groupby image_id) for masks of two models, roi = object, (lb, ub) = (0.8, 1.0)
toy_examples = []
for distribution, image_total in zip(["id_val", "ood_val"], [id_total, ood_total]):
    for image_idx in range(1, 1 + image_total):
        
        toy_examples.append(f"{distribution}_{image_idx}")
region = "object"
thresholds = [0.8, 0.8]
k = 25
region_area_threshold = 10000
reverse = False
