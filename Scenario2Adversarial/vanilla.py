# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from pathlib import Path
from PIL import Image
import cv2
# from pytorch_grad_cam import CAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
import pickle
import shelve
import heapq
from operator import itemgetter
import os
import sys
import time


# %%
# Set main path to scenario root directory
main = Path(".").resolve()
main


# %%
# # Load checkpoint
# checkpoint = torch.load(main/"checkpoints/resnet50_imagenette.pth")
# model = resnet50()

# print("CUDA Available: ", torch.cuda.is_available())
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# model.fc = nn.Linear(model.fc.in_features, 10)
# model.load_state_dict(checkpoint)
# model.eval()


# %%
dataset_name = "data"
num_processes = mp.cpu_count() // 2
num_processes

class ImagenettePath(datasets.Imagenette):
    def __getitem__(self, idx):
        path, label = self._samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label, path

# def fgsm_attack(image, epsilon, data_grad):
#     sign_data_grad = data_grad.sign()
#     perturbed_image = image + epsilon * sign_data_grad
#     perturbed_image = torch.clamp(perturbed_image, 0, 1)
#     return perturbed_image

def convert(input_image, multiply=False, BGR=False):
    multiplier = 255.0 if multiply else 1.0
    image = np.moveaxis(input_image.detach().cpu().numpy() * multiplier, 0, 2)
    if BGR:
        image = image[:, :, ::-1]
    return image


# %%
download = False
transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((400, 600))])
dataset = ImagenettePath(main/dataset_name, size='full',
                         split='val', transform=transform, download=download)
chunk = len(dataset) // num_processes
loader = torch.utils.data.DataLoader(dataset, batch_size=chunk, shuffle=False, num_workers=0)


# %%
# def process_attack(images, targets, paths, criterion, epsilon):
#     for image, target, path in zip(images, targets, paths):
#         image, target = image[None, :, :, :], torch.tensor([target]).to(device)
#         image.requires_grad = True
#         y_hat = model(image)
#         prediction = y_hat.argmax(1)
#         if prediction.item() != target.item():
#             continue
#         loss = criterion(y_hat, target)
#         model.zero_grad()
#         loss.backward()
#         perturbed_image = fgsm_attack(image, epsilon, image.grad)
#         path_split = path.split("/")
#         path_attacking = main/dataset_name/"imagenette2"/"val"/path_split[-2]/(path_split[-1][:-5] + "_attacked.JPEG")
#         cv2.imwrite(str(path_attacking), convert(perturbed_image[0], multiply=True, BGR=True))


# %%
# processes = []
# criterion = nn.CrossEntropyLoss()
# epsilon = 0.05

# for images, targets, paths in loader:
#     torch.set_num_threads(1)
#     model.share_memory()
#     p = mp.Process(target=process_attack, args=(images.to(device), targets.to(device), paths, criterion, epsilon))
#     processes.append(p)
#     p.start()
# for p in processes:
#     p.join()


# %%
serialized_folder_name = "serialized"


# %%
# transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((400, 600))])
# dataset = ImagenettePath(main/dataset_name, size='full',
#                          split='val', transform=transform, download=False)
# batch_size = len(dataset) // num_processes
# loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# %%
# def process_serialized(images, targets, paths, q, i):
#     images_local = []
#     cams_local = []
#     correctness_local= []
#     attack_local = []

#     if i < len(images):
#         for image, target, path in zip(images[i:i+1], targets[i:i+1], paths[i:i+1]):
#             target_layer = model.layer4[-1]
#             cam = CAM(model=model,  target_layer=target_layer, use_cuda=torch.cuda.is_available())
#             image, target = image[None, :, :, :], torch.tensor([target]).to(device)

#             image.requires_grad = False
#             prediction = model(image).argmax(1)

#             image.requires_grad = True
#             grayscale_cam = cam(input_tensor=image, target_category=target.item(), method="gradcam")
#             converted_image = convert(image[0])

#             images_local.append(converted_image)
#             cams_local.append(grayscale_cam)
#             correctness_local.append(prediction.item() == target.item())
#             attack_local.append("attack" in path)

#     q.put((images_local, cams_local, correctness_local, attack_local))
#     print(f"{os.getpid()} Complete")


# %%
# images_global = []
# cams_global = []
# correctness_global = []
# attack_global = []

# with tqdm(range(batch_size), desc=f"Load Saliency Maps", total=batch_size) as tq:
#     for i in tq:
#         processes = []
#         q = mp.Queue()

#         for images, targets, paths in loader:
#             torch.set_num_threads(1)
#             model.share_memory()
#             p = mp.Process(target=process_serialized, args=(images.to(device), targets.to(device), paths, q, i))
#             processes.append(p)
#             p.start()
#         for p in processes:
#             images_local, cams_local, correctness_local, attack_local = q.get()
#             images_global.extend(images_local)
#             cams_global.extend(cams_local)
#             correctness_global.extend(correctness_local)
#             attack_global.extend(attack_local)
#         for p in processes:
#             p.join()

# image_file = open(serialized_folder_name + "/image_data.pkl", "wb")
# cam_file = open(serialized_folder_name + "/cam_data.pkl", "wb")
# correctness_file = open(serialized_folder_name + "/correctness_data.pkl", "wb")
# attack_file = open(serialized_folder_name + "/attack_data.pkl", "wb")

# pickle.dump(images_global, image_file, pickle.HIGHEST_PROTOCOL)
# pickle.dump(cams_global, cam_file, pickle.HIGHEST_PROTOCOL)
# pickle.dump(correctness_global, correctness_file, pickle.HIGHEST_PROTOCOL)
# pickle.dump(attack_global, attack_file, pickle.HIGHEST_PROTOCOL)

# image_file.close()
# cam_file.close()
# correctness_file.close()
# attack_file.close()


# %%
# image_map = shelve.open(serialized_folder_name + "/image_map")
# image_file = open(serialized_folder_name + "/image_data.pkl", "rb")
# image_data = pickle.load(image_file)
# for i, image in enumerate(image_data):
#     image_map[f"{i}"] = image
# image_file.close()
# image_map.close()

# cam_map = shelve.open(serialized_folder_name + "/cam_map")
# cam_file = open(serialized_folder_name + "/cam_data.pkl", "rb")
# cam_data = pickle.load(cam_file)
# for i, cam in enumerate(cam_data):
#     cam_map[f"{i}"] = cam
# cam_file.close()
# cam_map.close()

# correctness_map = shelve.open(serialized_folder_name + "/correctness_map")
# correctness_file = open(serialized_folder_name + "/correctness_data.pkl", "rb")
# correctness_data = pickle.load(correctness_file)
# for i, correctness in enumerate(correctness_data):
#     correctness_map[f"{i}"] = correctness
# correctness_file.close()
# correctness_map.close()

# attack_map = shelve.open(serialized_folder_name + "/attack_map")
# attack_file = open(serialized_folder_name + "/attack_data.pkl", "rb")
# attack_data = pickle.load(attack_file)
# for i, attack in enumerate(attack_data):
#     attack_map[f"{i}"] = attack
# attack_file.close()
# attack_map.close()


# %%
def compute_dispersion(cam, threshold=(0.3, 0.45)):
    if isinstance(threshold, tuple):
        assert len(threshold) == 2
        return ((cam > threshold[0]) & (cam <= threshold[1])).sum()
    else:
        return (cam > threshold).sum()


# %%
image_file = open(serialized_folder_name + "/image_data.pkl", "rb")
cam_file = open(serialized_folder_name + "/cam_data.pkl", "rb")
correctness_file = open(serialized_folder_name + "/correctness_data.pkl", "rb")
attack_file = open(serialized_folder_name + "/attack_data.pkl", "rb")

image_data = pickle.load(image_file)
cam_data = pickle.load(cam_file)
correctness_data = pickle.load(correctness_file)
attack_data = pickle.load(attack_file)

start = time.time()

# Compute dispersion scores
dispersion_data = []
for cam in cam_data:
    dispersion_data.append(compute_dispersion(cam, threshold=(0.2, 0.4)))

visualize = False

# Load top-k results
k = 20
top_k = heapq.nlargest(k, zip(image_data, cam_data, correctness_data, attack_data, dispersion_data), key=itemgetter(4))

# if visualize:
#     for i, (image, cam, correctness, attack, dispersion) in enumerate(top_k):
#         cam_display = show_cam_on_image(image, cam)
#         plt.figure()
#         plt.imshow(cam_display)
#         plt.axis('off')
#         plt.show()
#         cv2.imwrite(str(main/"numpy_results"/f"cam_display_{i + 1}.JPEG"), cam_display)
#         print(f"dispersion={dispersion}, classification={correctness}, attack={attack}")

end = time.time()

print(f"Misclassification rate in {len(correctness_data)} images: {1 - sum(correctness_data) / len(correctness_data)}")
print(f"Misclassification rate in top {k}: {1 - sum([entry[2] for entry in top_k]) / len(top_k)}")
print(f"Attack rate in {len(attack_data)} images: {sum(attack_data) / len(attack_data)}")
print(f"Attack rate in top {k}: {sum([entry[3] for entry in top_k]) / len(top_k)}")

print("Naive query time:", end - start)

image_file.close()
cam_file.close()
correctness_file.close()
attack_file.close()

# %%
