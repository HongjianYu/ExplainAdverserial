from pathlib import Path
from tqdm.notebook import tqdm
import pickle
import shelve

main = Path("/homes/gws/hjyu/MaskSearchDemo/Scenario2Adversarial").resolve()
serialized_folder_name = "serialized"

image_map = shelve.open(serialized_folder_name + "/image_map")
image_file = open(serialized_folder_name + "/image_data.pkl", "rb")
image_data = pickle.load(image_file)
with tqdm(total=len(image_data), desc="Creating image map") as pbar:
    for i, image in enumerate(image_data):
        image_map[f"{i}"] = image
        pbar.update(1)
image_file.close()
image_map.close()

cam_map = shelve.open(serialized_folder_name + "/cam_map")
cam_file = open(serialized_folder_name + "/cam_data.pkl", "rb")
cam_data = pickle.load(cam_file)
with tqdm(total=len(cam_data), desc="Creating cam map") as pbar:
    for i, cam in enumerate(cam_data):
        cam_map[f"{i}"] = cam
        pbar.update(1)
cam_file.close()
cam_map.close()

correctness_map = shelve.open(serialized_folder_name + "/correctness_map")
correctness_file = open(serialized_folder_name + "/correctness_data.pkl", "rb")
correctness_data = pickle.load(correctness_file)
with tqdm(total=len(correctness_data), desc="Creating correctness map") as pbar:
    for i, correctness in enumerate(correctness_data):
        correctness_map[f"{i}"] = correctness
        pbar.update(1)
correctness_file.close()
correctness_map.close()

attack_map = shelve.open(serialized_folder_name + "/attack_map")
attack_file = open(serialized_folder_name + "/attack_data.pkl", "rb")
attack_data = pickle.load(attack_file)
with tqdm(total=len(attack_data), desc="Creating attack map") as pbar:
    for i, attack in enumerate(attack_data):
        attack_map[f"{i}"] = attack
        pbar.update(1)
attack_file.close()
attack_map.close()
