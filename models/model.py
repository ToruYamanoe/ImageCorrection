from email.policy import default
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import make_grid
import pytorch_lightning as pl
import torchmetrics
import os
import numpy as np
import pandas as pd
import pyocr

from .srcnn import SRCNN
from .vdsr import VDSR
from .swinir import SwinIR
from .unet import UNet
from .blur import BlurModel
from .BSRGAN import RRDBNet
from .RCAN import RCAN 
from .SRResNet import _NetG
from .loss import total_variation_loss
from .new_loss import Adjusted_MSE_Loss,Adjusted_PSNR_Loss,Adjusted_SSIM_Loss,Legibility_Loss


class LitImageCorrection(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitImageCorrection")
        parser.add_argument("--loss", type=str, choices=["mse", "l1", "mae", "psnr", "ssim", "tv","original","original2","original3"], default="l1")
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--sp", type=int, default=64)
        parser.add_argument("--model", type=str, default="srcnn", choices=["srcnn", "vdsr", "swinir", "unet","rcan","resnet"])
        parser.add_argument("--sphere", type=float, default=1.0)
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

    # def __init__(self, model, blur, loss:str=None, lr=1e-4):
    def __init__(self, args):
        super().__init__()

        self.img_shape = args.img_shape
        self.model_name = args.model

        # self.correction_model = model
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


        self.sphere = args.sphere
        self.cylinder = args.cylinder
        self.axis = args.axis
        self.radius = args.radius
        self.sp = args.sp
        self.blur_model = BlurModel(S=self.sphere, C=self.cylinder, A=self.axis, R=self.radius, sp=self.sp, img_shape=self.img_shape)

        self.lr = args.lr
        self.negate_loss = False

        self.loss_name = args.loss
        if self.loss_name == 'mse':
            self.loss_fn = F.mse_loss
        elif self.loss_name == 'l1' or self.loss_name == 'mae':
            self.loss_fn = F.l1_loss
        elif self.loss_name == 'psnr':
            self.loss_fn = torchmetrics.functional.peak_signal_noise_ratio
            # self.loss_fn = torchmetrics.PeakSignalNoiseRatio
            self.negate_loss = True
        elif self.loss_name == 'ssim':
            self.loss_fn = torchmetrics.functional.structural_similarity_index_measure
            self.negate_loss = True
        elif self.loss_name == 'tv':
            self.loss_fn = F.l1_loss
        elif self.loss_name == 'original':
            self.loss_fn = Adjusted_MSE_Loss

        elif self.loss_name == 'original2':
            self.loss_fn = Adjusted_SSIM_Loss
        elif self.loss_name == 'original3':
            self.loss_fn = Legibility_Loss
        else:
            raise ValueError("invalid loss name")

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
        self.logger.experiment.add_text("model", self.model_name)
        self.logger.experiment.add_text("loss", self.loss_name)

    def training_step(self, batch, batch_idx):
        imgs = batch

        corrected_imgs = self.correction_model(imgs)
        corrected_imgs.clamp_(self.lower_value, 1.0)

        blurred_corrected_imgs = self.blur_model(corrected_imgs)
        blurred_corrected_imgs.clamp_(self.lower_value, 1.0)

        blurred_imgs = self.blur_model(imgs)
        blurred_imgs.clamp_(self.lower_value, 1.0)

        if self.loss_name == 'tv':
            tv_loss = total_variation_loss(corrected_imgs, 10)
            loss = self.loss_fn(blurred_corrected_imgs, imgs)
        elif self.negate_loss:
            loss = 1-self.loss_fn(blurred_corrected_imgs, imgs)
        else:
            loss = self.loss_fn(blurred_corrected_imgs, imgs)

        if self.loss_name == 'tv':
            self.log('tv_loss', tv_loss.item(), on_epoch=True)
            self.log('l1_loss', loss.item(), on_epoch=True)
            loss += tv_loss
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
        self.logger.experiment.add_image('img(train)', grid.detach(), self.itr)
        self.itr += 1

    def validation_step(self, batch, batch_idx):
        imgs = batch

        corrected_imgs = self.correction_model(imgs)
        corrected_imgs.clamp_(self.lower_value, 1.0)

        blurred_corrected_imgs = self.blur_model(corrected_imgs)
        blurred_corrected_imgs.clamp_(self.lower_value, 1.0)

        blurred_imgs = self.blur_model(imgs)
        blurred_imgs.clamp_(self.lower_value, 1.0)

        if self.loss_name == 'tv':
            tv_loss = total_variation_loss(corrected_imgs, 10)
            loss = self.loss_fn(blurred_corrected_imgs, imgs)
        elif self.negate_loss:
            loss = 1-self.loss_fn(blurred_corrected_imgs, imgs, data_range=1.0)
        else:
            loss = self.loss_fn(blurred_corrected_imgs, imgs)

        if self.loss_name == 'tv':
            self.log('tv_loss', tv_loss.item(), on_epoch=True)
            self.log('l1_loss', loss.item(), on_epoch=True)
            loss += tv_loss

        self.valid_psnr(blurred_corrected_imgs, imgs)
        # self.valid_ssim(blurred_corrected_imgs, imgs)
        self.log('valid_loss', loss.item(), on_step=True, on_epoch=True, logger=True)
        self.log('valid_psnr', self.valid_psnr, on_epoch=True)
        # self.log('valid_ssim', self.valid_ssim, on_epoch=True)

        self.step += 1
        return {'loss': loss, 'blr_crr': blurred_corrected_imgs.detach(), 'imgs': imgs.detach(), 'crr': corrected_imgs.detach(), 'blr_imgs': blurred_imgs.detach()}
        # return loss

    def validation_step_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        
        # print('outputs:'+str(outputs))

        img = torch.cat([outputs[0]['imgs'][:4], outputs[0]['blr_imgs'][:4], outputs[0]['blr_crr'][:4], outputs[0]['crr'][:4]])
        
        if self.wide_range:
            img = self.denormalize(img)
        grid = make_grid(img, 4)
        img.detach()
        self.logger.experiment.add_image('img(valid)', grid.detach(), self.itr-1)
        # self.itr += 1
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # ReduceLROnPlateauスケジューラの設定例
        # ここでは、'valid_loss'が5エポック改善されない場合、学習率を0.1倍に減衰させる設定です。
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'valid_loss'  # スケジューラが参照する性能指標
            }
        }
