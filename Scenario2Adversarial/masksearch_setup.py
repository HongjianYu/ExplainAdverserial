# %%
import sys
sys.argv = [""]
from masksearch import *
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
from tqdm.notebook import tqdm
import argparse
import json
# import pickle
import shelve
import matplotlib.pyplot as plt
import time

# %%
# Set main path to scenario root directory (i.e. Scenario2Adversarial)
main = Path(".").resolve()
main

# %%
class ImagenettePath(datasets.Imagenette):
    def __getitem__(self, idx):
        path, label = self._samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label, path

transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((400, 600))])
dataset = ImagenettePath(main/"data", size='full',
                         split='val', transform=transform, download=False)

def convert(input_image, multiply=False, BGR=False):
    multiplier = 255.0 if multiply else 1.0
    image = input_image * multiplier
    if BGR:
        image = image[:, :, ::-1]
    return image

# %%
cam_map = shelve.open(str(main) + "/serialized/cam_map")
image_map = shelve.open(str(main) + "/serialized/image_map")
correctness_map = shelve.open(str(main) + "/serialized/correctness_map")
attack_map = shelve.open(str(main) + "/serialized/attack_map")

# %%
# with tqdm(total=len(image_map)) as pbar:
#     for i in range(len(image_map)):
#         idx = f"{i}"
#         image, cam = image_map[idx], cam_map[idx]
#         cam_image = show_cam_on_image(image, cam)
#         cv2.imwrite(str(main/"cam_images"/f"{i}.JPEG"), cam_image)
#         pbar.update(1)

# %%
# with tqdm(total=len(image_map)) as pbar:
#     for i in range(len(image_map)):
#         idx = f"{i}"
#         image = convert(image_map[idx], multiply=True, BGR=True)
#         cv2.imwrite(str(main/"pure_images"/f"{i}.JPEG"), image)
#         pbar.update(1)

# %%
image_total = len(dataset)
dataset_examples = []
for i in range(image_total):
    dataset_examples.append(f"{i}")

hist_size = 16
hist_edges = []
bin_width = 256 // hist_size
for i in range(1, hist_size):
    hist_edges.append(bin_width * i)
hist_edges.append(256)
cam_size_y = 400
cam_size_x = 600

available_coords = 20

in_memory_index_suffix = np.load(
    main/f"npy/imagenet_cam_hist_prefix_{hist_size}_available_coords_{available_coords}_np_suffix.npy"
)

# %%
region_area_threshold = 5000
region = (0, 0, 600, 400)
lv = 0.2
uv = 0.4
reverse = False
k = 20

image_access_order = range(len(dataset_examples))

start = time.time()
count, images = get_max_area_in_subregion_in_memory_version(
    "imagenet",
    image_map,
    correctness_map,
    attack_map,
    cam_map,
    None,
    bin_width,
    cam_size_y,
    cam_size_x,
    hist_size,
    dataset_examples,
    lv,
    uv,
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

print(images)
end = time.time()
print("Skipped images:", count)
print("(MaskSearch vanilla) Query time (cold cache):", end - start)

# %%
