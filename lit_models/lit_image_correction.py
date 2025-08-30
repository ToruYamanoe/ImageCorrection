import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import make_grid
import pytorch_lightning as pl
import torchmetrics

from models.correction.srcnn import SRCNN
from models.correction.vdsr import VDSR      
from models.correction.unet import UNet
from models.blur.blur import BlurModel       
from models.loss import zernike_cycle_loss   

class LitImageCorrection(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitImageCorrection")
        parser.add_argument("--loss", type=str, choices=["l1","mse","psnr","ssim"], default="psnr")
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--sp", type=int, default=64)
        parser.add_argument("--model", type=str, default="unet", choices=["unet","srcnn","vdsr"])
        parser.add_argument("--img_shape", type=int, nargs=3, default=[3, 500, 500])
        parser.add_argument("--wide_range", action="store_true")
        return parent_parser

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()  # ログに便利

        self.img_shape = args.img_shape
        self.model_name = args.model

        # 画像補正モデル（(x, z) を受け取れる実装にしておく）
        if args.model == "srcnn":
            self.correction_model = SRCNN(img_shape=self.img_shape)
        elif args.model == "vdsr":
            self.correction_model = VDSR(img_shape=self.img_shape, init_weights=True)
        elif args.model == "unet":
            self.correction_model = UNet(n_channels=args.img_shape[0])  # ← UNet側も (x, z) 対応版に
        else:
            raise ValueError("invalid model name")

        # 視覚再現モデル（zはforward時に渡す）
        self.blur_model = BlurModel(sp=args.sp, img_shape=self.img_shape)

        # 学習設定
        self.loss_name = args.loss
        self.lr = args.lr
        self.lower_value = -1 if args.wide_range else 0

        # 指標
        self.train_psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0)
        self.valid_psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0)
        self.itr = 0
        self.step = 0

    def denormalize(self, x):
        return 0.5 * x + 0.5

    def forward(self, x, z):
        x = self.correction_model(x, z)  # ← カンマ！(x, z)
        x.clamp_(self.lower_value, 1.0)
        return x

    def on_train_start(self):
        # 余計なhpログを簡素化
        self.logger.experiment.add_text("model", self.model_name)
        self.logger.experiment.add_text("loss", self.loss_name)

    def training_step(self, batch, batch_idx):
        imgs, z = batch  # ← (画像, ゼルニケ) を受け取る
        corrected_imgs = self.correction_model(imgs, z).clamp(self.lower_value, 1.0)

        # 自作サイクルロス（補正→視覚再現と元画像の誤差）
        loss = zernike_cycle_loss(corrected_imgs, imgs, z, self.blur_model, loss_type=self.loss_name)

        # ログ
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('current_lr', current_lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # 参考用PSNR（視覚再現後で）
        blurred_corrected_imgs = self.blur_model(corrected_imgs, z).clamp(self.lower_value, 1.0)
        self.train_psnr(blurred_corrected_imgs, imgs)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_psnr', self.train_psnr, on_epoch=True)

        self.step += 1
        return {'loss': loss,
                'imgs': imgs.detach(),
                'blr_crr': blurred_corrected_imgs.detach(),
                'crr': corrected_imgs.detach()}

    def training_epoch_end(self, outputs):
        img = torch.cat([outputs[0]['imgs'][:4], outputs[0]['blr_crr'][:4], outputs[0]['crr'][:4]])
        if self.lower_value < 0: img = self.denormalize(img)
        grid = make_grid(img, 4)
        self.logger.experiment.add_image('img(train)', grid.detach(), self.itr)
        self.itr += 1

    def validation_step(self, batch, batch_idx):
        imgs, z = batch
        corrected_imgs = self.correction_model(imgs, z).clamp(self.lower_value, 1.0)
        loss = zernike_cycle_loss(corrected_imgs, imgs, z, self.blur_model, loss_type=self.loss_name)

        blurred_corrected_imgs = self.blur_model(corrected_imgs, z).clamp(self.lower_value, 1.0)
        self.valid_psnr(blurred_corrected_imgs, imgs)
        self.log('valid_loss', loss.item(), on_step=True, on_epoch=True, logger=True)
        self.log('valid_psnr', self.valid_psnr, on_epoch=True)

        self.step += 1
        return {'loss': loss,
                'imgs': imgs.detach(),
                'blr_crr': blurred_corrected_imgs.detach(),
                'crr': corrected_imgs.detach()}

    def validation_epoch_end(self, outputs):
        img = torch.cat([outputs[0]['imgs'][:4], outputs[0]['blr_crr'][:4], outputs[0]['crr'][:4]])
        if self.lower_value < 0: img = self.denormalize(img)
        grid = make_grid(img, 4)
        self.logger.experiment.add_image('img(valid)', grid.detach(), self.itr-1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': 'valid_loss'}}
