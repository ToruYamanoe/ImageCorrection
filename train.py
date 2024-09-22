from pyparsing import autoname_elements
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# from pytorch_lightning.callbacks.gpu_stats_monitor import GPUStatsMonitor
# from pytorch_lightning.plugins import DDPPlugin
# from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from models.blur import BlurModel
from models.model import LitImageCorrection
from datasets import ImageDataModule

from argparse import ArgumentParser

class ClearCudaCashe(pl.Callback):
    def on_epoch_end(self,trainer,pl_module):
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--loggr", type=str, dest="logger",default="tensorboard", choices=["tensorboard", "mlflow"])
    parser.add_argument("--expname", type=str, default="default")
    parser.add_argument("--grad_clip", type=float, default=0.01)

    parser = LitImageCorrection.add_model_specific_args(parser)
    parser = ImageDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    # チェックポイントファイルのパスを設定
    checkpoint_path = "./pre_trained.ckpt"

    # LitImageCorrectionインスタンスをチェックポイントからロード
    model = LitImageCorrection.load_from_checkpoint(checkpoint_path, args=args)
    # model = LitImageCorrection(args)
    # dataset = ImageDataModule(args.data_path, args.anotation_path, batch_size=args.batch_size, num_workers=12)
    # dataset = ImageDataModule('../datasets/', '../datasets/MSRA-TD500/td500_train.csv', batch_size=4, num_workers=12)
    dataset = ImageDataModule.from_argparse_args(args)

    if args.logger == "tensorboard":
        logger = TensorBoardLogger('lightning_logs', default_hp_metric=False, name=args.expname)
    elif args.logger == "mlflow":
        logger = MLFlowLogger(tracking_uri="")
    else:
        raise ValueError("invalid logger name")

    es_cb = EarlyStopping('valid_loss_epoch', patience=25,)
    # lrmon_cb = LearningRateMonitor('epoch')
    # gpu_cb = GPUStatsMonitor()
    cp_cb =ModelCheckpoint('results/', 'ckpt-{epoch:4d}', monitor='valid_loss_epoch', save_top_k=3)


    clear_cuda_cashe = ClearCudaCashe()
    trainer = pl.Trainer.from_argparse_args(args, logger=logger,precision=16, auto_select_gpus=True,  gradient_clip_val=args.grad_clip/args.lr)

    
    trainer.fit(model, dataset)
