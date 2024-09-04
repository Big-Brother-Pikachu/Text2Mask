# Utilize Stable Diffusion for Segmentation
We propose a simple but effective way to distill the attention information in diffusion models for segmentation.

## 1. Reqirements
```
# create conda env
conda env create -f environment.yaml
conda activate ldm

# install pydensecrf from source
git clone https://github.com/lucasb-eyer/pydensecrf
cd pydensecrf
python setup.py install
```

## 2. Preparing Datasets
### PASCAL VOC 2012
Set up PASCAL VOC dataset following [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch/blob/master/data/datasets/voc12/README.md).
The structure of `datasets/VOCdevkit` should be organized as follows:
```
VOCdevkit
└── VOC2012
    ├── Annotations
    ├── ImageSets
    │   ├── Segmentation
    │   └── SegmentationAug
    │       ├── test.txt
    │       ├── train_aug.txt
    │       ├── train.txt
    │       ├── trainval_aug.txt
    │       ├── trainval.txt
    │       └── val.txt
    ├── JPEGImages
    ├── SegmentationObject
    ├── SegmentationClass
    └── SegmentationClassAug
        └── 2007_000032.png
```

### MS COCO 2014
Download MS COCO images from the [official website](https://cocodataset.org/#download).
Download semantic segmentation annotations for the MS COCO dataset at [here](https://drive.google.com/file/d/1pRE9SEYkZKVg0Rgz2pi9tg48j7GlinPV/view?usp=sharing).
The structure of `datasets/COCO` are suggested to be organized as follows:
```
COCO
└── COCO2014
    ├── Annotations
    ├── ImageSets
    │	└── SegmentationClass
    │	    ├── train.txt
    │       └── val.txt
    └── JPEGImages
	├── train2014
	└── val2014
```

## 3. Preparing Pre-trained Models
The Stable Diffusion v1.4 model should be downloaded from [Huggingface](https://huggingface.co/CompVis/stable-diffusion) and be put to `models/ldm/stable-diffusion-v1/` as `model.ckpt`.
The CLIP pre-trained model will be downloaded automatically when running the codes.

## 4. Generate Pseudo Masks
We put the running commands in `run_scripts.sh`.
```
bash run_scripts.sh
```
After running, the generated pseudo masks are saved in `datasets/VOCdevkit/VOC2012/PseudoLabelMysample` or `datasets/COCO/COCO2014/PseudoLabelMysample`.

## 5. Train Segmentation Model
To train DeepLab v2, we refer to [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch). The ImageNet pre-trained model can be found in [AdvCAM](https://github.com/jbeomlee93/AdvCAM).
