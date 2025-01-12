
import os
import h5py
from io import BytesIO
from PIL import Image
from glob import glob
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score

import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import pytorch_lightning as pl

BASE_DATA_DIR = "/kaggle/input/isic-2024-challenge/"
BASE_IMG_DIR = os.path.join(BASE_DATA_DIR, "train-image/image")
df_train_meta = pd.read_csv(BASE_DATA_DIR + "train-metadata.csv")
df_test_meta  = pd.read_csv(BASE_DATA_DIR + "test-metadata.csv")

na_cols = ["sex", "age_approx", "anatom_site_general"]
df_train_meta[na_cols] = df_train_meta[na_cols].fillna(df_train_meta[na_cols].mode().iloc[0])
df_test_meta[na_cols] = df_test_meta[na_cols].fillna(df_train_meta[na_cols].mode().iloc[0])

num_cols = [
    'age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 
    'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 
    'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 
    'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB',
    'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM',
    'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
    'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL',
    'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',
    'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',
]
cat_cols = ["sex", "anatom_site_general", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple"]
train_cols = num_cols

df_train_meta[train_cols] = df_train_meta[train_cols].astype(float)
df_test_meta[train_cols] = df_test_meta[train_cols].astype(float)

file_hdf = os.path.join(BASE_DATA_DIR + "test-image.hdf5")
fp_hdf = h5py.File(file_hdf, mode="r")

def read_image(isic_id):
    # From: https://www.kaggle.com/code/motono0223/isic-pytorch-inference-baseline-image-only
    return Image.open(BytesIO(fp_hdf[isic_id][()]))


class ISICDataset(Dataset):
    def __init__(self, df_meta, transforms=None):
        self.df_meta = df_meta
        self.transforms = transforms

    def __len__(self):
        return len(self.df_meta)

    def __getitem__(self, idx):
        row = self.df_meta.iloc[idx]
        meta = torch.from_numpy(row[train_cols].values.astype(float)).float()
        # Returns the same image as cv2.cvtColor(cv2.imread(...), cv2.COLOR_BGR2RGB)
        img = np.array(read_image(row.isic_id))
        if self.transforms:
            img = self.transforms(image=img)["image"]
        return img, meta

img_size = 384
valid_transforms = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


class ISICModel(nn.Module):
    def __init__(self, model_name, in_meta_features, inner_feature_shape=256, meta_dropout_p=0.0, pretrained=False):
        super(ISICModel, self).__init__()
        self.image_backbone = timm.create_model(model_name, pretrained=pretrained)
        if "resnet" in model_name:
            image_out_features = self.image_backbone.fc.in_features
            self.image_backbone.fc = nn.Linear(image_out_features, 1)
        elif "efficientnet" in model_name:
            image_out_features = self.image_backbone.classifier.in_features
            self.image_backbone.classifier = nn.Linear(image_out_features, 1)

    def forward(self, inputs):
        x, meta = inputs
        x = self.image_backbone(x)
        return x


class ISICModule(pl.LightningModule):
    def __init__(
        self, 
        model_name="efficientnet_b0", 
        in_meta_features=34, 
        inner_feature_shape=256, 
        meta_dropout_p=0.0, 
        pretrained=False
    ):
        super().__init__()
        self.model = ISICModel(model_name, in_meta_features, inner_feature_shape=inner_feature_shape, meta_dropout_p=meta_dropout_p, pretrained=False)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.validation_targets = []
        self.validation_preds = []
        
    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        images, meta, labels = batch
        outputs = self.model((images, meta))
        loss = self.loss_fn(outputs, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, meta, labels = batch
        outputs = self.model((images, meta))
        loss = self.loss_fn(outputs, labels)
        labels = labels.detach().cpu()# .numpy().tolist()
        outputs = torch.sigmoid(outputs.detach().cpu())# .numpy().tolist()
        self.validation_targets.extend(labels)
        self.validation_preds.extend(outputs)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
    def on_validation_epoch_end(self):
        targets = torch.stack(self.validation_targets)
        outputs = torch.stack(self.validation_preds)
        outputs = torch.stack([targets, outputs], dim=1).squeeze(-1).numpy()
        df_sub = pd.DataFrame(outputs, columns=["label", "prediction"])
        score = comp_score(df_sub[["label"]], df_sub[["prediction"]], "")
        self.log("val_score", score, prog_bar=True)
        self.validation_targets.clear()
        self.validation_preds.clear()
        return score
    


ckpt_paths = sorted(glob("/kaggle/input/isic-2024-fails/epoch-2*"))
model_name = "efficientnet_b0"
in_meta_features = len(train_cols)
dropout_p = 0.1
inner_feature_shape = 256
test_dataset = ISICDataset(df_test_meta, valid_transforms)
dl_test = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=3, pin_memory=True)
models = [ISICModule.load_from_checkpoint(ckpt_path, model_name=model_name).eval() for ckpt_path in ckpt_paths]
all_preds = []
with torch.no_grad():
    for imgs, meta in dl_test:
        imgs, meta = imgs.to("cuda"), meta.to("cuda")
        preds = [torch.sigmoid(model((imgs, meta))).detach().cpu().numpy() for model in models]
        preds = np.mean(preds, 0)
        all_preds.extend(preds[:, 0])
        
df_sub = pd.read_csv("/kaggle/input/isic-2024-challenge/sample_submission.csv")
df_sub["target"] = all_preds
df_sub.to_csv("submission.csv", index=False)
