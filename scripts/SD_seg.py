import argparse, os, sys, glob
from ast import Pass
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import torchvision
import time
import datetime
import shutil
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import pydensecrf
from pydensecrf.utils import unary_from_softmax, unary_from_labels, create_pairwise_bilateral
from pydensecrf.densecrf import DenseCRF2D

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
import os.path as osp
import joblib
import multiprocessing
import copy, random, clip


def scoremap2bbox(scoremap, threshold, multi_contour_eval=False):
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY)
    contours = cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes), len(contours)


class BaseDataset(torch.utils.data.Dataset):
    """
    Base dataset with extra annotations
    """

    def __init__(self, root, dataset_name="VOC", year=2012, image_set='train_aug', **kwargs):
        if dataset_name == "VOC":
            self.root = osp.join(root, "VOCdevkit")
        elif dataset_name == "COCO":
            self.root = osp.join(root, "COCO")
        self.dataset_name = dataset_name
        self.year = year
        self.image_set = image_set
        self.files = []
        self._set_files()

    def _set_files(self):
        if self.dataset_name == "VOC":
            self.root = osp.join(self.root, "VOC{}".format(self.year))

            if self.image_set in ["train", "train_aug", "trainval", "trainval_aug", "val"]:
                file_list = osp.join(
                    self.root, "ImageSets/SegmentationAug", self.image_set + ".txt"
                )
                file_list = tuple(open(file_list, "r"))
                file_list = [id_.rstrip().split(" ") for id_ in file_list]
                image_names, target_names = list(zip(*file_list))
                self.images = [osp.join(self.root, x[1:]) for x in image_names]
                self.targets = [osp.join(self.root, x[1:]) for x in target_names]
            else:
                raise ValueError("Invalid split name: {}".format(self.image_set))
            
            # seg_prompts = [
            #     "background", "aeroplane", "bicycle", "bird avian", "boat", "bottle", "bus", "car", "cat", "chair seat", "cow","dining table", 
            #     "dog","horse", "motorbike", "person with clothes, people, human", "potted plant", "sheep", "sofa", "train", "tv monitor screen"
            #     ]
            self.seg_prompts = [
                "background", "aeroplane", "bicycle", "bird avian", "boat", "bottle", "bus", "car", "cat", "chair seat", "cow","dining table", 
                "dog","horse", "motorbike", "person with clothes", "potted plant", "sheep", "sofa", "train", "tv monitor screen"
                ]
            # seg_prompts = [
            #     "aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, dining table, dog, horse, motorbike, person, potted plant, sheep, sofa, train, and tv monitor."
            #     ]
        elif self.dataset_name == "COCO":
            self.root = osp.join(self.root, "COCO{}".format(self.year))

            if self.image_set in ["train", "val"]:
                file_list = osp.join(
                    self.root, "ImageSets/SegmentationClass", self.image_set + ".txt"
                )
                file_list = tuple(open(file_list, "r"))
                file_list = [id_.rstrip().split(" ") for id_ in file_list]
                self.images = [osp.join(self.root, "JPEGImages", f"{self.image_set}{self.year}", x[0] + '.jpg') for x in file_list]  # [2000:]
                self.targets = [osp.join(self.root, "Annotations", x[0].split("_")[-1] + '.png') for x in file_list]  # [2000:]
            else:
                raise ValueError("Invalid split name: {}".format(self.image_set))
            
            # self.seg_prompts = ['background', 'person with clothes,people,human','bicycle','car','motorbike','aeroplane',
            #         'bus','train','truck','boat','traffic light',
            #         'fire hydrant','stop sign','parking meter','bench','bird avian',
            #         'cat','dog','horse','sheep','cow',
            #         'elephant','bear','zebra','giraffe','backpack,bag',
            #         'umbrella,parasol','handbag,purse','necktie','suitcase','frisbee',
            #         'skis','sknowboard','sports ball','kite','baseball bat',
            #         'glove','skateboard','surfboard','tennis racket','bottle',
            #         'wine glass','cup','fork','knife','dessertspoon',
            #         'bowl','banana','apple','sandwich','orange',
            #         'broccoli','carrot','hot dog','pizza','donut',
            #         'cake','chair seat','sofa','pottedplant','bed',
            #         'diningtable','toilet','tvmonitor screen','laptop','mouse',
            #         'remote control','keyboard','cell phone','microwave','oven',
            #         'toaster','sink','refrigerator','book','clock',
            #         'vase','scissors','teddy bear','hairdrier,blowdrier','toothbrush',
            #         ]
            self.seg_prompts = ['background', 'person with clothes','bicycle','car','motorbike','aeroplane',
                    'bus','train','truck','boat','traffic light',
                    'fire hydrant','stop sign','parking meter','bench','bird avian',
                    'cat','dog','horse','sheep','cow',
                    'elephant','bear','zebra','giraffe','backpack, bag',
                    'umbrella, parasol','handbag, purse','necktie','suitcase','frisbee',
                    'skis','sknowboard','sports ball','kite','baseball bat',
                    'glove','skateboard','surfboard','tennis racket','bottle',
                    'wine glass','cup','fork','knife','dessert spoon',
                    'bowl','banana','apple','sandwich','orange',
                    'broccoli','carrot','hot dog','pizza','donut',
                    'cake','chair seat','sofa','potted plant','bed',
                    'dining table','toilet','tv monitor screen','laptop','mouse',
                    'remote control','keyboard','cell phone','microwave','oven',
                    'toaster','sink','refrigerator','book','clock',
                    'vase','scissors','teddy bear','hairdrier, blowdrier','toothbrush',
                    ]
    
    def __len__(self):
        return len(self.images)


class MySegmentation(BaseDataset):
    @property
    def masks(self):
        return self.targets

    @property
    def palette(self):
        target = Image.open(self.masks[0]).convert("P")
        return(target.getpalette())

    def __getitem__(self, index: int):
        bg_mean = (123, 117, 104)
        ignore_label = 255
        times = 64

        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index]).convert("P")

        w, h = img.size
        ratio = max(h / w, w / h)
        if ratio > 16 / 7:
            real_size = 64 * 16
        elif ratio > 15 / 8:
            real_size = 64 * 15
        elif ratio > 14 / 8:
            real_size = 64 * 14
        elif ratio > 13 / 9:
            real_size = 64 * 13
        elif ratio > 12 / 10:
            real_size = 64 * 12
        else:
            real_size = 64 * 11
        if w > h:
            w_upsize, h_upsize = real_size, int(real_size*h/w)
            width = real_size
            if h_upsize % times == 0:
                height = h_upsize
            else:
                height = h_upsize - h_upsize % times + times
            left = 0
            # left = (height - h_upsize) // 2
            up = 0
            right = 0
            bottom = height - h_upsize - up
        else:
            w_upsize, h_upsize = int(real_size*w/h), real_size
            height = real_size
            if w_upsize % times == 0:
                width = w_upsize
            else:
                width = w_upsize - w_upsize % times + times
            left = 0
            up = 0
            # up = (width - w_upsize) // 2
            right = width - w_upsize - left
            bottom = 0
        pad_img = torchvision.transforms.Pad((left, up, right, bottom), fill=bg_mean)
        pad_target = torchvision.transforms.Pad((left, up, right, bottom), fill=ignore_label)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((h_upsize, w_upsize)),
            pad_img, 
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize(0.5, 0.5)
        ])
        img = transform(img)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        target = transform(target)

        return img, target, torch.tensor([w, h, left, up, right, bottom, width, height])


class Logger(object):
    def __init__(self, stdout, filename):
        self.logfile = filename
        self.terminal = stdout
 
    def write(self, message):
        self.terminal.write(message)
        if self.logfile != "":
            try:
                self.log = open(self.logfile, 'a')
                self.log.write(message)
                self.log.close()
            except:
                pass
 
    def flush(self):
        pass


class PCA(object):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        n = X.shape[0]
        self.mean = torch.mean(X, axis=0)
        X = X - self.mean
        covariance_matrix = 1 / n * torch.matmul(X.T, X)
        eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        self.proj_mat = eigenvectors[:, 0:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return X.matmul(self.proj_mat)


class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = unary_from_softmax(probmap)
        # U = pydensecrf.utils.unary_from_labels(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q


def crf_inference_label(img, labels, t=10, n_labels=21, gt_prob=0.7):

    h, w = img.shape[:2]

    d = DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)


def crf_inference_label_feat(feat, labels, t=10, n_labels=21, gt_prob=0.7, sdims=50, schan=2):

    h, w = feat.shape[:2]

    d = DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)

    pairwise_energy = create_pairwise_bilateral(sdims=(sdims,sdims), schan=schan, img=np.ascontiguousarray(np.copy(feat)), chdim=2)
    d.addPairwiseEnergy(pairwise_energy, compat=10)

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, class_name):
    n_class = len(class_name)
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    valid = hist.sum(axis=1) > 0  # added
    acc_cls = np.diag(hist)[valid] / hist.sum(axis=1)[valid]
    acc_cls = np.nanmean(acc_cls)
    iu_valid = (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)) > 0  # added
    iu = np.zeros(np.diag(hist).shape)
    iu.fill(np.nan)
    iu[iu_valid] = np.diag(hist)[iu_valid] / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))[iu_valid]
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    # cls_iu = dict(zip(range(n_class), iu))
    cls_iu = dict(zip(class_name, iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False, device="cuda:0"):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model = model.to(device)
    model.eval()
    return model


def crf(n_jobs, save_npy_name, dataset, seg_prompts):
    """
    CRF post-processing on pre-computed logits
    """

    # Configuration
    torch.set_grad_enabled(False)
    print("# jobs:", n_jobs)

    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )
    mean_bgr = (104.008, 116.669, 122.675)

    folder_name = "PseudoLabelMysample"
    save_crf_name = dataset.images[0].split("/")
    save_crf_name = save_crf_name[0] + "/" + save_crf_name[1] + "/" + save_crf_name[2] + "/" + save_crf_name[3] + "/" + folder_name
    os.makedirs(save_crf_name, exist_ok=True)

    # Process per sample
    def process(i):
        if dataset.dataset_name == "VOC":
            image_id = dataset.images[i].split("/")[-1]
        elif dataset.dataset_name == "COCO":
            image_id = dataset.images[i].split("/")[-1].split("_")[-1]
        image = cv2.imread(dataset.images[i], cv2.IMREAD_COLOR).astype(np.float32)
        gt_label = np.asarray(Image.open(dataset.masks[i]), dtype=np.int32)
        # Mean subtraction
        image -= mean_bgr

        cam_dict = np.load(os.path.join(save_npy_name, image_id.replace('jpg', 'npy')), allow_pickle=True).item()
        cams = cam_dict['high_res']
        prob = cams

        image = image.astype(np.uint8)
        prob = postprocessor(image, prob)

        label = np.argmax(prob, axis=0)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        label = keys[label]
        eval_label = copy.deepcopy(label)
        confidence = np.max(prob, axis=0)
        label[confidence < 0.95] = 255
        cv2.imwrite(os.path.join(save_crf_name, image_id.replace('jpg', 'png')), label.astype(np.uint8))

        return eval_label.astype(np.uint8), gt_label.astype(np.uint8)

    # CRF in multi-process
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
           [joblib.delayed(process)(i) for i in range(len(dataset))]
    )
    
    preds, gts = zip(*results)

    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    score = scores(gts, preds, seg_prompts)
    print(score)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--noise_steps",
        type=int,
        nargs='+',  
        default=[500],
        help="number of add noise steps",
    )
    parser.add_argument(
        '--layers', 
        type=int, 
        nargs='+',  
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
        help="number of attn layers",
    )
    parser.add_argument(
        '--self_layers', 
        type=int, 
        default=13, 
        help="number of self_attn layers",
    )
    parser.add_argument(
        '--feat_layers', 
        type=int, 
        nargs='+',  
        default=[0, 4, 7, 10, 13, 16, 19, 22, 25], 
        help="number of feat layers",
    )
    parser.add_argument(
        '--sdims', 
        type=float, 
        default=50, 
        help="scale of position channels",
    )
    parser.add_argument(
        '--schan', 
        type=float, 
        default=2, 
        help="scale of feat channels",
    )
    parser.add_argument(
        '--n_components', 
        type=int, 
        default=10, 
        help="number of feat dims",
    )
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.05, 
        help="attention threshold",
    )
    parser.add_argument(
        '--fg_th', 
        type=float, 
        default=0.2, 
        help="foreground threshold",
    )
    parser.add_argument(
        '--bg_th', 
        type=float, 
        default=0.05, 
        help="background threshold",
    )
    parser.add_argument(
        "--sample_times",
        type=int,
        default=1,
        help="how many times to add noises",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="the used gpu",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--npy_folder",
        type=str,
        help="where to save the npy",
        default="PseudoLabelES"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="run experiments on which dataset",
        default="VOC"
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)
    device = torch.device("cuda:" + str(opt.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # load Stable Diffusion
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}", device=device)

    # prepare dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(576),
        torchvision.transforms.CenterCrop((512, 512)), 
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize(0.5, 0.5)
    ])
    target_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(576),
        torchvision.transforms.CenterCrop((512, 512)), 
        torchvision.transforms.ToTensor()
    ])
    if opt.dataset == "VOC":
        VOC2012_val = MySegmentation("./datasets/", dataset_name="VOC", year= '2012', image_set='train', aug=True, transform=transform, target_transform=target_transform)
    elif opt.dataset == "COCO":
        VOC2012_val = MySegmentation("./datasets/", dataset_name="COCO", year= '2014', image_set='train', aug=True, transform=transform, target_transform=target_transform)
    dataloader = torch.utils.data.DataLoader(VOC2012_val, batch_size = opt.n_samples)

    folder_name = opt.npy_folder
    save_name = VOC2012_val.images[0].split("/")
    save_name = save_name[0] + "/" + save_name[1] + "/" + save_name[2] + "/" + save_name[3] + "/" + folder_name
    os.makedirs(save_name, exist_ok=True)
    seg_prompts = VOC2012_val.seg_prompts
    class_nums = (torch.max(clip.tokenize(seg_prompts), dim=1)[1] - 1)[1:].view(-1, 1)
    clims_bg_prompts = [
        'tree', 'river',
        'sea', 'lake', 'water',
        'railway', 'railroad', 'track',
        'stone', 'rocks'
        ]
    clipes_bg_prompts = [
        'ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad','keyboard','helmet',
        'cloud','house','mountain','ocean','road','rock','street','valley','bridge','sign'
        ]
    my_bg_prompts = [
        'tree', 'river',
        'sea', 'lake', 'water',
        'railway', 'railroad', 'track',
        'stone', 'rocks',
        'wine glass', 'painting', 'wall'
        ]
    clipes_bg_prompts_COCO = ['ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad','helmet',
                        'cloud','house','mountain','ocean','road','rock','street','valley','bridge',
                        ]

    if opt.dataset == "VOC":
        CAA_threshold = 0.2
        palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,
					 64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,
					 0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,
					 64,64,0,  192,64,0,  64,192,0, 192,192,0]
        background_prompts = [["background"]] + [clims_bg_prompts] * 3 + [['river', 'sea', 'lake', 'water', 'mountain']] + [clims_bg_prompts] * 5\
        + [["grass", 'stone', 'rocks']] + [clims_bg_prompts] * 8 + [['railway', 'railroad', 'track']] + [["keyboard", "wall"]]
    elif opt.dataset == "COCO":
        CAA_threshold = 0.5
        palette = VOC2012_val.palette
        background_prompts = [clims_bg_prompts] * len(seg_prompts)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    sample_times = opt.sample_times

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # log folder
    dateTime_p = datetime.datetime.now()
    dateTime_p = datetime.datetime.strftime(dateTime_p, '%Y-%m-%d-%H-%M-%S')
    param = "-".join([str(x) for x in opt.noise_steps]) + "_" + "-".join([str(x) for x in opt.layers]) + "_" + str(opt.threshold)
    output_dir = os.path.join(outpath, f"{dateTime_p}_{param}_{opt.gpu:01}")
    os.makedirs(output_dir, exist_ok=True)
    files = ['scripts/SD_seg.py', 'ldm/models/diffusion/plms.py', 'ldm/models/diffusion/ddpm.py', 'ldm/modules/diffusionmodules/openaimodel.py', 'ldm/modules/attention.py', "run_scripts.sh"]
    for f in files:
        shutil.copy(f, os.path.join(output_dir, f.split('/')[-1]))
    stdout_backup = sys.stdout
    sys.stdout = Logger(stdout_backup, os.path.join(output_dir, 'run.log'))
    print('--------args----------')
    for k in list(vars(opt).keys()):    
        print('%s: %s' % (k, vars(opt)[k]))
    print('--------args----------\n')

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                all_attn = []
                gd_truth = []
                for i, (images, labels, origin_sizes) in tqdm(
                    enumerate(dataloader), total=np.ceil(len(VOC2012_val) / batch_size), leave=False
                    ):
                    images = images.to(device)
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    z_samples_ddim = model.get_first_stage_encoding(model.encode_first_stage(images))
                    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).cpu()
                    sampler.make_schedule(ddim_num_steps=50, ddim_eta=opt.ddim_eta, verbose=False)

                    steps = opt.noise_steps
                    choices = torch.unique(labels) * 255
                    choices = choices[(choices > 0) * (choices < 255)]
                    un_step_attn = []
                    step_attn = []
                    step_feat = []

                    while len(all_attn) < len(choices):
                        all_attn.append([])
                        gd_truth.append([])

		    # text query
                    seg_prompt = "a photo including "
                    contras_prompt = ""
                    begin_index = 1
                    location_begin = 3
                    class_indexes = []
                    bg_prompt = []
                    for nums, choice in enumerate(choices):
                        bg_prompt.extend(background_prompts[int(choice)])
                        if nums < len(choices) - 1:
                            seg_prompt = seg_prompt + seg_prompts[int(choice)] + ", "
                        else:
			    # append background prompts
                            unique_bg_prompt = list(set(bg_prompt))
                            unique_bg_prompt.sort(key = bg_prompt.index)
                            my_bg_prompt = ", ".join(unique_bg_prompt[:-1]) + f", and {unique_bg_prompt[-1]}."
                            seg_prompt = seg_prompt + seg_prompts[int(choice)] + ", " + my_bg_prompt
                        location_end = location_begin + class_nums[int(choice) - 1][0]
                        class_indexes.append(list(range(location_begin, location_end)))
                        location_begin = location_end + 1
                    c_seg = [seg_prompt] * batch_size
                    tokens = model.cond_stage_model.tokenizer(c_seg, truncation=True, max_length=77, return_length=True,
                                    return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
                    c_seg = model.get_learned_conditioning(c_seg)
                    con_seg = [contras_prompt] * batch_size
                    con_seg = model.get_learned_conditioning(con_seg)
                    for j in range(sample_times):
                        for step in steps:
			    # add noise to the latent
                            ts = torch.full((opt.n_samples,), step, device=device, dtype=torch.long)
                            img_diff = sampler.model.q_sample(z_samples_ddim, ts)

                            outs = sampler.p_sample_plms(img_diff, c_seg, ts, index=0,
                                        unconditional_guidance_scale=opt.scale,
                                        unconditional_conditioning=uc,
                                        old_eps=[], t_next=ts, return_attns=True, return_feats=False)
                            _, _, e_t = outs
			    # extract Self-Attn
                            self_attn, _, height, width = e_t[1][opt.self_layers]
                            _, con_selfattn_map = self_attn.chunk(2)
                            un_step_attn.append(con_selfattn_map.unsqueeze(1).cpu())
			    # extract Cross-Attn
                            for class_num in class_indexes:
                                un_word_attn = []
                                word_attn = []
                                for index in opt.layers:
                                    _, attn_map, _, _ = e_t[1][index]
                                    un_attn_map, con_attn_map = attn_map.chunk(2)

                                    attn_map = torch.mean(torch.mean(con_attn_map[:, :, :, :, [2]], dim=1), dim=-1, keepdim=True)
                                    attn_map = rearrange(attn_map, 'b h w t -> (b t) h w')
                                    attn_map = attn_map.unsqueeze(1)
                                    attn_map = torch.nn.functional.interpolate(attn_map, (height, width), mode='bilinear', align_corners=False)
                                    attn_map = rearrange(attn_map[:, 0, :, :], '(b t) h w -> b h w t', b=con_attn_map.shape[0]).unsqueeze(1)

                                    un_word_attn.append(attn_map.cpu())
                                    con_attn_map = con_attn_map[:, :, :, :, begin_index:]
                                    attn_map = torch.mean(torch.mean(con_attn_map[:, :, :, :, class_num], dim=1), dim=-1, keepdim=True)
                                    attn_map = rearrange(attn_map, 'b h w t -> (b t) h w')
                                    attn_map = attn_map.unsqueeze(1)
                                    attn_map = torch.nn.functional.interpolate(attn_map, (height, width), mode='bilinear', align_corners=False)
                                    attn_map = rearrange(attn_map[:, 0, :, :], '(b t) h w -> b h w t', b=con_attn_map.shape[0]).unsqueeze(1)

                                    word_attn.append(attn_map.cpu())
                                step_attn.append(torch.mean(torch.cat(word_attn, dim=1), dim=1).unsqueeze(1))
                    if len(step_attn) == 0:
                        continue

                    attn_map = torch.cat(step_attn, dim=-1).view(*step_attn[0].shape[:4], len(steps)*sample_times, -1).squeeze(1).permute(0, 3, 1, 2, 4)
                    self_attn_map = torch.cat(un_step_attn, dim=1)

                    attn_map = (attn_map - torch.amin(attn_map, dim=[2, 3], keepdim=True)) / (torch.amax(attn_map, dim=[2, 3], keepdim=True) - torch.amin(attn_map, dim=[2, 3], keepdim=True))

		    # CAA to obtain SelfCross
                    class_aware_attns = []
                    for class_aware in range(attn_map.shape[-1]):
                        sample_aware_attns = []
                        for sample_idx in range(attn_map.shape[1]):
                            class_aware_attn = attn_map[:, :, :, :, [class_aware]]
                            class_aware_attn = class_aware_attn[:, [sample_idx], :, :, :]
                            box, cnt = scoremap2bbox(scoremap=class_aware_attn[0, 0, :, :, 0].numpy(), threshold=CAA_threshold, multi_contour_eval=True)
                            aff_mask = torch.zeros_like(class_aware_attn[0, 0, :, :, 0])
                            for i_ in range(cnt):
                                x0_, y0_, x1_, y1_ = box[i_]
                                aff_mask[y0_:y1_, x0_:x1_] = 1

                            aff_mask = aff_mask.view(1, 1, class_aware_attn.shape[-3] * class_aware_attn.shape[-2])
                            trans_mat = self_attn_map[:, sample_idx, :, :].float() * aff_mask

                            class_aware_attn_flat = class_aware_attn.view(class_aware_attn.shape[0], -1, class_aware_attn.shape[-1])
                            class_aware_attn = torch.einsum('b l j, b j s -> b l s', trans_mat, class_aware_attn_flat.float()).view(*class_aware_attn.shape)
                            sample_aware_attns.append(class_aware_attn)
                        class_aware_attns.append(torch.cat(sample_aware_attns, dim=1))
                    attn_map = torch.cat(class_aware_attns, dim=-1)
		    # normalize SelfCross
                    attn_map = (attn_map - torch.amin(attn_map, dim=[2, 3], keepdim=True)) / (torch.amax(attn_map, dim=[2, 3], keepdim=True) - torch.amin(attn_map, dim=[2, 3], keepdim=True))

		    # attain pseudo masks
                    attn_map = torch.mean(attn_map, dim=1).unsqueeze(1)
                    attn_map = torch.cat([torch.pow(2 * opt.threshold - torch.max(attn_map, dim=-1)[0].unsqueeze(-1), 2), attn_map], dim=-1)
                    class_map = torch.max(attn_map, dim=-1)[1].unsqueeze(-1)

                    label_map = torch.zeros_like(class_map)
                    for ids, choice in enumerate(choices):
                        label_map[class_map == (ids+1)] = int(choice)
                    class_map = label_map

		    # save before-dCRF results
                    for batch_idx, origin_size in enumerate(origin_sizes):
                        high_class_map = torchvision.transforms.Resize((images.shape[-2], images.shape[-1]), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(class_map[batch_idx, :, :, :, 0])
                        w, h, left, up, right, bottom, width, height = origin_size
                        pseudo_label = torchvision.transforms.Resize((h, w), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(high_class_map[:, up:(height-bottom), left:(width-right)])
                        all_attn[len(choices)-1].append(pseudo_label.int().numpy())
                        label = labels[batch_idx]
                        gd_truth[len(choices)-1].append((label * 255).int().numpy())

                        high_attn_map = torchvision.transforms.Resize((images.shape[-2], images.shape[-1]))(attn_map[batch_idx].permute(3, 0, 1, 2))
                        cam = high_attn_map[:, :, up:(height-bottom), left:(width-right)]
                        high_res = torchvision.transforms.Resize((h, w))(cam).permute(1, 0, 2, 3)[0]
                        save_name = VOC2012_val.masks[i * batch_size + batch_idx].split("/")
                        save_name = save_name[0] + "/" + save_name[1] + "/" + save_name[2] + "/" + save_name[3] + "/" + folder_name + "/" + save_name[5].replace('png', 'npy')
                        np.save(save_name, {"keys": (choices.cpu()).int() - 1, "high_res": high_res.numpy()})

                toc = time.time()
                print("Time: ", (toc - tic) / len(VOC2012_val))

		# before-dCRF evaluation results
                gds = []
                attns = []
                for i in range(len(gd_truth)):
                    if len(gd_truth[i]) == 0:
                        continue
                    evaluation = scores(gd_truth[i], all_attn[i], seg_prompts)
                    print(i, len(gd_truth[i]))
                    print(evaluation)
                    gds.extend(gd_truth[i])
                    attns.extend(all_attn[i])
                evaluation = scores(gds, attns, seg_prompts)
                print(evaluation)

		# after-dCRF evaluation results
                folder_name = opt.npy_folder
                save_name = VOC2012_val.images[0].split("/")
                save_name = save_name[0] + "/" + save_name[1] + "/" + save_name[2] + "/" + save_name[3] + "/" + folder_name
                n_jobs =multiprocessing.cpu_count()
                crf(n_jobs, save_name, VOC2012_val, seg_prompts)
                shutil.rmtree(save_name)

                if not opt.skip_grid:
                    grid_count += 1


if __name__ == "__main__":
    main()
