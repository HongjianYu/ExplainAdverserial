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

def visualize_image_and_mask_side_by_side(image, mask, image_id, rect=None):
    fig, axs = plt.subplots(1, 2, figsize=(8, 5))  # 1 row, 2 columns

    # Display the image
    axs[0].imshow(image)
    axs[0].set_title(f"{image_id}: {label_map[image_id]}->{pred_map[image_id]}")

    # Display the mask
    axs[1].imshow(mask)
    axs[1].set_title(f"{image_id} Mask")

    if rect is not None:
        x, y, w, h = rect
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=5, edgecolor="b", facecolor="none"
        )
        axs[0].add_patch(rect)
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=5, edgecolor="y", facecolor="none"
        )
        axs[1].add_patch(rect)

    plt.tight_layout()
    plt.show()
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

misclassified_img_id = []
mis_label_set = {}

num_label = 0.0

for key in pred_map:
    pred_label = pred_map[key]
    true_label = label_map[key]
    if true_label == 146:
        num_label += 1
    if(pred_label != true_label):
        misclassified_img_id.append(key)
        if true_label in mis_label_set:
            
            mis_label_set[true_label].append(key)
           
        else:
            mis_label_set[true_label] = [key]

print(len(mis_label_set))
print(len(misclassified_img_id))

dev_acc = len(misclassified_img_id) / len(pred_map)
print("accuracy: ", dev_acc)



# ground truth label : a list of misclassified images w this ground truth label
sorted_mis_label_set = sorted(mis_label_set, key=lambda x: len(mis_label_set[x]), reverse=True)

# for label in sorted_mis_label_set:
#     print("&&" , label)
#     print(len(mis_label_set[label]))




cam_size_y = 448
cam_size_x = 448

id_total = 7314
ood_total = 14961
dataset_examples = []
for distribution, image_total in zip(["id_val", "ood_val"], [id_total, ood_total]):
    for image_idx in range(1, 1 + image_total):
        dataset_examples.append(f"{distribution}_{image_idx}")
       
print("dataset_examples length: ", len(dataset_examples))

hist_size = 16
hist_edges = []
bin_width = 256 // hist_size
for i in range(1, hist_size):
    hist_edges.append(bin_width * i)
hist_edges.append(256)
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
print(len(sorted_mis_label_set))
image_map = wilds_random_access_images(
    id_val_data, ood_val_data, misclassified_img_id
)

label=146
print(len(mis_label_set[label]))
print(num_label)
print(len(mis_label_set[label]) / num_label)


#%%
# present the misclassified images & saliency maps w given ground truth label
for i in range(25):
    label = sorted_mis_label_set[i]
    print(label)

#%%
for i in range(1):
    label = 146
    if label == 1: continue
    img_set = mis_label_set[label]
    cnt = 0
    for image_id in img_set:
        x, y, w, h = get_object_region(
            object_detection_map, cam_size_y, cam_size_x, image_id
        )
        mask = cam_map[image_id]
        if image_id not in image_map: 
            print("not in:", image_id)
            continue
            
        image = from_input_to_image(image_map[image_id])
        cam_image = show_cam_on_image(image, mask, use_rgb=True, image_weight=0.)
        visualize_image_and_mask_side_by_side(image, cam_image, image_id, rect=(x, y, w, h))
        cnt += 1
        if cnt > 64: break


# %%
# Setup for top-k subregion query

toy_examples = []
for distribution, image_total in zip(["id_val", "ood_val"], [id_total, ood_total]):
    for image_idx in range(1, 1 + image_total):
        toy_examples.append(f"{distribution}_{image_idx}")

print("toy_examples length: ", len(toy_examples))
region_area_threshold = 5000
region = "object"
threshold = 0.8
reverse = True
k = 25

# MaskSearch: top-k subregion query processing with in-memory index without optimization

image_access_order = range(len(toy_examples))

start = time.time()
count, images, augmented_img, orig_img = get_max_area_in_subregion_in_memory_version(
    "wilds",
    (id_val_data, ood_val_data),
    label_map,
    pred_map,
    cam_map,
    object_detection_map,
    bin_width,
    cam_size_y,
    cam_size_x,
    toy_examples,
    threshold,
    region,
    in_memory_index_suffix,
    image_access_order,
    early_stoppable=False,
    k=k,
    region_area_threshold=region_area_threshold,
    ignore_zero_area_region=True,
    reverse=reverse,
    visualize=True,
    available_coords=available_coords,
    compression=None,
)

augmented_img *= 255
orig_img *= 255
end = time.time()
print("Skipped images:", count)
# print("Skipped images:", sum(count), "out of", len(toy_examples))
# print("Skipping ratio:", sum(count) / len(toy_examples))
print("(MaskSearch vanilla) Query time (cold cache):", end - start)
print(len(images))
print("augmented_num: ", len(augmented_img))


def visualize_mask(mask, image_id, rect=None):
    fig, axs = plt.subplots(1, 1, figsize=(8, 5))  

    # Display the mask
    axs.imshow(mask)
    axs.set_title(f"{image_id} Mask")

    if rect is not None:
        x, y, w, h = rect
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=5, edgecolor="b", facecolor="none"
        )
        axs.add_patch(rect)
    plt.tight_layout()
    plt.show()


for img in augmented_img:
    plt.axis('off')
    plt.imshow(img)

#%%
plt.axis('off')
plt.imshow(augmented_img[24])

# def show_images(images):
#     num_images = len(images)
#     fig, axes = plt.subplots(1, num_images, figsize=(10, 4))  # Adjust figsize as needed
#     for i, img in enumerate(images):
#         axes[i].axis('off')
#         axes[i].imshow(img)
        
#     plt.show()

# show_images(augmented_img)


# %%
# filter query with MaskSearch

count, images = get_images_satisfying_filter("wilds",
    cam_map,
    object_detection_map,
    bin_width,
    hist_size,
    cam_size_y,
    cam_size_x,
    toy_examples,
    threshold,
    region,
    v,
    region_area_threshold=region_area_threshold,
    ignore_zero_area_region=True,
    in_memory_index_suffix,
    visualize=True,
    available_coords=available_coords,
    compression=None,)




# %%
# print(images)
tot = len(images)
print(tot)
# images = images[:tot]
cnt = 0
plt.figure(figsize=(8, 10))

freq_class = {}
for j in range(tot):
    metric, area, image_id = images[j]
    label = label_map[image_id]
    freq_class[label] = freq_class.get(label, 0) + 1

freq_class = sorted(freq_class.items(), key = lambda x:x[1], reverse=True)
print(freq_class)
target = 47
selected_images = []


for j in range(tot):
    cnt += 1
    metric, area, image_id = images[j]

    if not isinstance(region, tuple):
        x, y, w, h = get_object_region(
            object_detection_map, cam_size_y, cam_size_x, image_id
        )
    else:
        x, y, w, h = region

    if w == 0 or h == 0:
        continue

    if dataset == "imagenet":
        image = image_map[image_id].reshape(3, 224, 224)
    else:
        image = image_map[image_id].reshape(3, 448, 448)
    cam = cam_map[image_id]

    ax = plt.subplot((tot + 4) // 5, 5, cnt)
    plt.xticks([], [])
    plt.yticks([], [])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')

    image = from_input_to_image(image)
    cam_image = show_cam_on_image(image, cam, use_rgb=True, image_weight=1.)
    plt.imshow(cam_image)
    if(label_map[image_id] == target):
        selected_images.append(image_id)
    plt.title(f"{image_id}: {label_map[image_id]}->{pred_map[image_id]}")
    # rect = patches.Rectangle(
    #     (x, y), w, h, linewidth=5, edgecolor="b", facecolor="none"
    # )
    # ax.add_patch(rect)

    plt.tight_layout()
    plt.show()

print(selected_images)
# %%

def visualize_image(image, image_id, rect=None):
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.imshow(image)
    plt.title(f"{image_id}: {label_map[image_id]}->{pred_map[image_id]}")
    if rect is not None:
        x, y, w, h = rect
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=5, edgecolor="b", facecolor="none"
        )
        ax.add_patch(rect)
    plt.tight_layout()
    plt.show()


region = "object"
cnt = 0
class_ids = [101]
for image_id in dataset_examples:
    # print(label_map[image_id], pred_map[image_id])
    # if label_map[image_id] not in class_ids:
    #     continue
    if isinstance(region, tuple):
        x, y, w, h = region
    else:
        x, y, w, h = get_object_region(
            object_detection_map, cam_size_y, cam_size_x, image_id
        )
    if w == 0 or h == 0:
        continue
    if pred_map[image_id] != label_map[image_id]:
        image_map = wilds_random_access_images(
            id_val_data, ood_val_data, [image_id]
        )
        visualize_image(from_input_to_image(image_map[image_id]), image_id, (x, y, w, h))
        cnt += 1
        if cnt == 100:
            break

# %%




for i in range(len(selected_images)):
    mask = cam_map[selected_images[i]]
    fake_image = np.zeros((448, 448, 3))
    cam_image = show_cam_on_image(fake_image, mask, use_rgb=True, image_weight=0.)
    plt.figure
    plt.imshow(cam_image)
    plt.title(f"{selected_images[i]}: {label_map[selected_images[i]]}->{pred_map[selected_images[i]]}")
    plt.show()
    plt.close()



image_map = wilds_random_access_images(
    id_val_data, ood_val_data, selected_images
)

for i in range(len(selected_images)):
    image_id = selected_images[i]
    x, y, w, h = get_object_region(
        object_detection_map, cam_size_y, cam_size_x, image_id
    )
    mask = cam_map[image_id]
    image = from_input_to_image(image_map[image_id])
    cam_image = show_cam_on_image(image, mask, use_rgb=True, image_weight=0.)
    visualize_image_and_mask_side_by_side(image, cam_image, image_id, rect=(x, y, w, h))

# %%

toy_examples = []
for distribution, image_total in zip(["id_val", "ood_val"], [id_total, ood_total]):
    for image_idx in range(1, 1 + image_total):
        image_id = f"{distribution}_{image_idx}"
        if label_map[image_id] == 113:
            toy_examples.append(image_id)

region_area_threshold = 0
region = "object"
threshold = 0.8
reverse = True
k = 25

image_access_order = range(len(toy_examples))

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
    toy_examples,
    threshold,
    region,
    in_memory_index_suffix,
    image_access_order,
    early_stoppable=False,
    k=k,
    region_area_threshold=region_area_threshold,
    ignore_zero_area_region=True,
    reverse=reverse,
    visualize=False,
    available_coords=available_coords,
    compression=None,
)
end = time.time()
print("Skipped images:", count)
# print("Skipped images:", sum(count), "out of", len(toy_examples))
# print("Skipping ratio:", sum(count) / len(toy_examples))
print("(MaskSearch vanilla) Query time (cold cache):", end - start)

# %%
print(images)

selected_images = [x[-1] for x in images]
print(selected_images)
image_map = wilds_random_access_images(
    id_val_data, ood_val_data, selected_images
)

for i in range(len(selected_images)):
    image_id = selected_images[i]
    print(image_id)
    x, y, w, h = get_object_region(
        object_detection_map, cam_size_y, cam_size_x, image_id
    )
    mask = cam_map[image_id]
    image = from_input_to_image(image_map[image_id])
    cam_image = show_cam_on_image(image, mask, use_rgb=True, image_weight=0.)
    visualize_image_and_mask_side_by_side(image, cam_image, image_id, rect=(x, y, w, h))

# %%
