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

# %%
cam_map = shelve.open("./serialized/cam_map")
with open("./serialized/image_data.pkl", "rb") as f:
    image_map = pickle.load(f)
with open("./serialized/correctness_data.pkl", "rb") as f:
    correctness_map = pickle.load(f)
with open("./serialized/attack_data.pkl", "rb") as f:
    attack_map = pickle.load(f)

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

available_coords = 14

in_memory_index_suffix = np.load(
    f"./npy/trial_imagenet_cam_hist_prefix_{hist_size}_available_coords_{available_coords}_memmap_suffix.npy",
    allow_pickle=True
)

# %%
