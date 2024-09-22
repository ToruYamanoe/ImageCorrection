from pyparsing import autoname_elements
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from models.blur import BlurModel
from models.model import LitImageCorrection
from datasets import ImageDataModule
from PIL import Image
from argparse import ArgumentParser
import numpy as np
import pyocr
import sys
import torchmetrics
from torch.nn import functional as F
import matplotlib.pyplot as plt


# 各評価指標の初期化
mse_metric = torchmetrics.MeanSquaredError()
psnr_metric = torchmetrics.PeakSignalNoiseRatio()
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

from argparse import Namespace


# 必要な引数を設定
args_dict = {
    "model": "vdsr",
    "loss": "original3",
    "lr": 1e-4,
    "sp": 64,
    "sphere": 0.8,
    "cylinder": 0.0,
    "axis": 0,
    "radius": 1.5,
    "img_shape": [3, 500, 500],
    "wide_range": False
}
args = Namespace(**args_dict)

# チェックポイントのパス
checkpoint_path = r"C:\Users\kwlab\Documents\yamanoue\image_correction\lightning_logs\default\MSE+OCR\checkpoints\epoch=210-step=10128.ckpt"
# モデルのロード
model = LitImageCorrection.load_from_checkpoint(checkpoint_path=checkpoint_path, args=args)

# モデルを評価モードに設定
model.eval()

# BlurModel の初期化
blur = BlurModel(S=args_dict["sphere"])
test_dataset = []
with open('./dataset/train.txt', 'r') as file:
    for line in file:
        image_path = './dataset/' + line.strip()  # 各行から改行文字を取り除く
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

    # 補正された画像を保存または表示
    # output_image.show()
    # output_image.save('output_image.jpg')  # または画像を表示するコード

    # 画像を開く
    recorrected_image = output_image

    # 画像をNumPy配列に変換
    recorrected_image = np.array(recorrected_image)

    # 画像をPyTorchテンソルに変換
    recorrected_image = torch.from_numpy(recorrected_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 再度ぼかしを適用
    recorrected_image_tensor = blur(recorrected_image)
    recorrected_image = Image.fromarray((recorrected_image_tensor[0] * 255).byte().permute(1, 2, 0).numpy())

    # MSEとPSNRの計算
    mse_metric(input_image_tensor.reshape(-1), recorrected_image_tensor.reshape(-1))
    psnr_metric(input_image_tensor, recorrected_image_tensor)

    # OCRでテキスト抽出
    image1 = np.array(input_image)
    image2 = np.array(recorrected_image)
    pyocr.tesseract.TESSERACT_CMD = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("OCRツールが見つかりませんでした")
        sys.exit(1)

    tools = pyocr.get_available_tools()

    tool = tools[0]
    txt1 = tool.image_to_string(
        Image.fromarray(np.uint8(image1)),
        lang='eng',
        builder=pyocr.builders.TextBuilder(tesseract_layout=6)
    )
    txt2 = tool.image_to_string(
        Image.fromarray(np.uint8(image2)),
        lang='eng',
        builder=pyocr.builders.TextBuilder(tesseract_layout=6)
    )
    # Levenshtein距離の計算
    levenshtein_dist = levenshtein_distance(txt1, txt2)
    levenshtein_distances.append(levenshtein_dist)

# 各評価指標の平均値を計算
average_mse = mse_metric.compute()
average_psnr = psnr_metric.compute()
average_levenshtein = sum(levenshtein_distances) / len(levenshtein_distances)

print("original\n\n")
print(f"Average MSE: {average_mse}")
print(f"Average PSNR: {average_psnr}")
print(f"Average Levenshtein Distance: {average_levenshtein}")