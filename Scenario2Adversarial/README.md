## Scenario 2 Setup

### [Download the prescribed zip files here (~30GB)](https://drive.google.com/drive/folders/1UGKixXdXw0oGPmhF-3aXiYW-BiNbbptu?usp=sharing) to replace the placeholder directories
```
cam_images
npy
pure_images
serialized
```


### Install packages in a separate environment
```
conda create --name masksearch_s2 --file ./Scenario2Adversarial/packageslist.txt
```

### Initialize (Additional ~30GB space required)
```
npm install
python ./Scenario2Adversarial/prepare_serialized.py
```

### Start fe/be
```
npm start
python ./GUI/backend/scenario2.py
```
