# %%
import sys
sys.argv = [""]
from masksearch import *
from torchvision import datasets, transforms
from pytorch_grad_cam.utils.image import show_cam_on_image
from pathlib import Path
from PIL import Image
import argparse
import json
import pickle
import shelve
import matplotlib.pyplot as plt
import time

# %%
main = Path("/homes/gws/hjyu/MaskSearchDemo/Scenario2Adversarial").resolve()
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

# %%
cam_map = shelve.open(str(main) + "/serialized/cam_map")
image_map = shelve.open(str(main) + "/serialized/image_map")
correctness_map = shelve.open(str(main) + "/serialized/correctness_map")
attack_map = shelve.open(str(main) + "/serialized/attack_map")

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
