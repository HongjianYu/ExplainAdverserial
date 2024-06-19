## Scenario 2 Setup

### [Download the prescribed zip files here](https://drive.google.com/drive/folders/1UGKixXdXw0oGPmhF-3aXiYW-BiNbbptu?usp=sharing) to replace the placeholder directories
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

### Initialize
```
npm install
python ./Scenario2Adversarial/prepare_serialized.py
```

### Start
```
npm start  # localhost:3000
python ./GUI/backend/scenario2.py
```
