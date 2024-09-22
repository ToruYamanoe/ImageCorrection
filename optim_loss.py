from pyparsing import autoname_elements
from email.policy import default
import torch
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from models.blur import BlurModel
from PIL import Image
from argparse import ArgumentParser
import numpy as np
import pyocr
import sys
import torchmetrics
from torchvision import transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt
import easyocr
from models.vdsr import VDSR
from models.srcnn import SRCNN
from models.swinir import SwinIR
from models.unet import UNet
from models.BSRGAN import RRDBNet
from models.RCAN import RCAN 
from models.SRResNet import _NetG
from models.new_loss import Legibility_Loss
from datasets import ImageDataModule

reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

import pytorch_lightning as pl
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.io import read_image
import pandas as pd
import os
from typing import Optional
from PIL import Image
import numpy as np
import optuna



class LitImageCorrection(pl.LightningModule):
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitImageCorrection")
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--model", type=str, default="srcnn", choices=["srcnn", "vdsr", "swinir", "unet","rcan","resnet"])
        parser.add_argument("--sp", type=int, default=64)
        parser.add_argument("--sphere", type=float, default=0.8)
        parser.add_argument("--cylinder", type=float, default=0.0)
        parser.add_argument("--axis", type=int, default=0)
        parser.add_argument("--radius", type=float, default=1.5)
        parser.add_argument("--img_shape", type=int, nargs=3, default=[3, 500, 500])
        parser.add_argument('--scale', type=int, default=1)
        parser.add_argument('--num_features', type=int, default=64)
        parser.add_argument('--num_rg', type=int, default=10)
        parser.add_argument('--num_rcab', type=int, default=20)
        parser.add_argument('--reduction', type=int, default=16)
        return parent_parser

    def __init__(self, args, alpha, beta):
        super().__init__()
        self.model_name = args.model

        if args.model == "srcnn":
            self.correction_model = SRCNN(img_shape=self.img_shape)
        elif args.model == "vdsr":
            self.correction_model = VDSR(img_shape=self.img_shape, init_weights=True)
        elif args.model == "swinir":
            self.correction_model = SwinIR(img_size=self.img_shape[-1], upscale=1, window_size=8)
        elif args.model == "unet":
            self.correction_model = UNet(n_channels=args.img_shape[0])
        elif args.model == "rcan":
            self.correction_model = RCAN(args=args)
        elif args.model == "resnet":
            self.correction_model = _NetG()
        else:
            raise ValueError("invalid model name")
        self.alpha = alpha
        self.beta = beta
        self.img_shape = args.img_shape
        self.sphere = args.sphere
        self.cylinder = args.cylinder
        self.axis = args.axis
        self.radius = args.radius
        self.sp = args.sp
        self.blur_model = BlurModel(S=self.sphere, C=self.cylinder, A=self.axis, R=self.radius, sp=self.sp, img_shape=self.img_shape)
        self.lr = args.lr
        self.negate_loss = False
        self.wide_range = args.wide_range
        if self.wide_range:
            self.lower_value = -1
        else:
            self.lower_value = 0
        self.train_psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0)
        self.valid_psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0)
        # self.valid_ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0)

        self.itr = 0
        self.step = 0

    def denormalize(self, x):
        return 0.5 * x + 0.5

    def forward(self, x):
        x = self.correction_model(x)
        x.clamp_(self.lower_value, 1.0)
        return x

    def on_train_start(self):
        self.logger.log_hyperparams({
            'hp/sphere': self.sphere,
            'hp/cylinder': self.cylinder,
            'hp/axis': self.axis,
            'hp/radius': self.radius,
            'hp/sp': self.sp
        })
        # self.logger.experiment.add_text("model", self.model_name)
        # self.logger.experiment.add_text("loss", self.loss_name)

    def training_step(self, batch, batch_idx):
        imgs = batch

        corrected_imgs = self.correction_model(imgs)
        corrected_imgs.clamp_(self.lower_value, 1.0)

        blurred_corrected_imgs = self.blur_model(corrected_imgs)
        blurred_corrected_imgs.clamp_(self.lower_value, 1.0)

        blurred_imgs = self.blur_model(imgs)
        blurred_imgs.clamp_(self.lower_value, 1.0)

        loss = Legibility_Loss(blurred_corrected_imgs, imgs,alpha=self.alpha,beta=self.beta)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        # 学習率をログに記録
        self.log('current_lr', current_lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_psnr(blurred_corrected_imgs, imgs)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_psnr', self.train_psnr, on_epoch=True)

        self.step += 1
        # return loss
        return {'loss': loss, 'blr_crr': blurred_corrected_imgs.detach(), 'imgs': imgs.detach(), 'crr': corrected_imgs.detach(), 'blr': blurred_imgs.detach()}

    def training_step_end(self, outputs):
        pass

    def training_epoch_end(self, outputs):
        img = torch.cat([outputs[0]['imgs'][:4], outputs[0]['blr'][:4], outputs[0]['blr_crr'][:4], outputs[0]['crr'][:4]])
        if self.wide_range:
            img = self.denormalize(img)
        grid = make_grid(img, 4)
        img.detach()
        self.logger.experiment.add_image('img(train)',grid.detach(), self.itr)
        self.itr += 1

    def validation_step(self, batch, batch_idx):
        imgs = batch

        corrected_imgs = self.correction_model(imgs)
        corrected_imgs.clamp_(self.lower_value, 1.0)

        blurred_corrected_imgs = self.blur_model(corrected_imgs)
        blurred_corrected_imgs.clamp_(self.lower_value, 1.0)

        blurred_imgs = self.blur_model(imgs)
        blurred_imgs.clamp_(self.lower_value, 1.0)

        loss = Legibility_Loss(blurred_corrected_imgs, imgs,alpha=self.alpha,beta=self.beta)

        self.valid_psnr(blurred_corrected_imgs, imgs)
        self.log('valid_loss', loss.item(), on_step=True, on_epoch=True, logger=True)
        self.log('valid_psnr', self.valid_psnr, on_epoch=True)

        self.step += 1
        return {'loss': loss, 'blr_crr': blurred_corrected_imgs.detach(), 'imgs': imgs.detach(), 'crr': corrected_imgs.detach(), 'blr_imgs': blurred_imgs.detach()}
        def validation_step_end(self, outputs):
            pass

    def validation_epoch_end(self, outputs):
        
        # print('outputs:'+str(outputs))

        img = torch.cat([outputs[0]['imgs'][:4], outputs[0]['blr_imgs'][:4], outputs[0]['blr_crr'][:4], outputs[0]['crr'][:4]])
        
        if self.wide_range:
            img = self.denormalize(img)
        img.detach()
        grid = make_grid(img, 4)
        self.logger.experiment.add_image('img(valid)', grid.detach(), self.itr-1)
        # self.itr += 1
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'valid_loss'  # スケジューラが参照する性能指標
            }
        }


def train_and_evaluate_model(alpha, beta):
    parser = ArgumentParser()

    parser.add_argument("--loggr", type=str, dest="logger",default="tensorboard", choices=["tensorboard", "mlflow"])
    parser.add_argument("--expname", type=str, default="default")
    parser.add_argument("--grad_clip", type=float, default=0.01)

    parser = LitImageCorrection.add_model_specific_args(parser)
    parser = ImageDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    # チェックポイントファイルのパスを設定
    checkpoint_path = r"C:\Users\kwlab\Documents\yamanoue\image_correction\lightning_logs\default\resnet\checkpoints\epoch=499-step=386500.ckpt"
# # モデルのロード
    model = LitImageCorrection.load_from_checkpoint(checkpoint_path=checkpoint_path, args=args)

    # model = LitImageCorrection(args)

    dataset = ImageDataModule.from_argparse_args(args)

    if args.logger == "tensorboard":
        logger = TensorBoardLogger('lightning_logs', default_hp_metric=False, name=args.expname)
    elif args.logger == "mlflow":
        logger = MLFlowLogger(tracking_uri="")
    else:
        raise ValueError("invalid logger name")

    es_cb = EarlyStopping('valid_loss_epoch', patience=25,)

    cp_cb =ModelCheckpoint('results/', 'ckpt-{epoch:4d}', monitor='valid_loss_epoch', save_top_k=3)


    trainer = pl.Trainer.from_argparse_args(args, logger=logger,precision=16, auto_select_gpus=True,  gradient_clip_val=args.grad_clip/args.lr)
 
    trainer.fit(model, dataset)


    # 各評価指標の初期化
    levenshtein_distances = []

    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]




    

    # モデルを評価モードに設定
    model.eval()

    # BlurModel の初期化
    blur = BlurModel(S=0.8)
    test_dataset = []
    with open('./datasets_characters/test.txt', 'r') as file:
        for line in file:
            image_path = './datasets_characters/' + line.strip()  # 各行から改行文字を取り除く
            test_dataset.append(image_path)
    for original_img_path in test_dataset:

        input_image = Image.open(original_img_path)

        # 画像をPyTorchテンソルに変換
        input_image_tensor = torch.from_numpy(np.array(input_image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # 学習済みモデルに画像を適用
        with torch.no_grad():
            output_image_tensor = model(input_image_tensor)

        # 補正された画像をPIL Imageに変換
        output_image = Image.fromarray((output_image_tensor[0] * 255).byte().permute(1, 2, 0).numpy())


        # 画像を開く
        recorrected_image = output_image

        # 画像をNumPy配列に変換
        recorrected_image = np.array(recorrected_image)

        # 画像をPyTorchテンソルに変換
        recorrected_image = torch.from_numpy(recorrected_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # 再度ぼかしを適用
        recorrected_image_tensor = blur(recorrected_image)
        recorrected_image = Image.fromarray((recorrected_image_tensor[0] * 255).byte().permute(1, 2, 0).numpy())


        # OCRでテキスト抽出
        image1 = np.array(input_image)
        image2 = np.array(recorrected_image)

        txt1 = reader.readtext(image1,detail=0)
        txt2 = reader.readtext(image2,detail=0)

        # Levenshtein距離の計算
        levenshtein_dist = sum([levenshtein_distance(s1, s2) for s1, s2 in zip(txt1,txt2)])
        levenshtein_distances.append(levenshtein_dist)

    # 各評価指標の平均値を計算
    average_levenshtein = sum(levenshtein_distances) / len(levenshtein_distances)

    return average_levenshtein




def objective(trial):
    # Optunaで探索するハイパーパラメータの範囲を指定
    alpha = trial.suggest_float('alpha', 0.01, 1.0)
    beta = trial.suggest_float('beta', 0.00, 1.0)

    # ここでモデルの評価を行う関数を実行
    result = train_and_evaluate_model(alpha, beta)

    # 最適化の目的はこのresultを最小化（または最大化）すること
    return result

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')  # 目的関数を最小化
    study.optimize(objective, n_trials=50)  # 100回のトライアルを実行

    # 最適なパラメータを出力
    print(study.best_params)

