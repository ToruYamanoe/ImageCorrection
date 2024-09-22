from pyparsing import autoname_elements
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# from pytorch_lightning.callbacks.gpu_stats_monitor import GPUStatsMonitor
from pytorch_lightning.plugins import DDPPlugin
# from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from models.blur import BlurModel
from models.model import LitImageCorrection
from datasets import ImageDataModule
from PIL import Image
from argparse import ArgumentParser
import numpy as np
import pyocr
import sys
import matplotlib.pyplot as plt
import easyocr
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory




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
    "loss": "original",
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
# checkpoint_path = r"C:\Users\kwlab\Documents\yamanoue\image_correction\lightning_logs\default\MSE+OCR\checkpoints\epoch=210-step=10128.ckpt"
checkpoint_path = r"C:\Users\kwlab\Documents\yamanoue\image_correction\lightning_logs\default\Tanaka_data\version_1\checkpoints\epoch=171-step=105092.ckpt"
# モデルのロード
model = LitImageCorrection.load_from_checkpoint(checkpoint_path=checkpoint_path, args=args)

# モデルを評価モードに設定
model.eval()

# BlurModel の初期化
blur = BlurModel(S=args_dict["sphere"])


# 実際の画像を読み込み
# input_image = Image.open(r"C:\Users\kwlab\Documents\yamanoue\image_correction\datasets\ArT\test_part1_images\gt_20.jpg")  # または他の画像ファイルのパス
input_image = Image.open('./datasets_characters/test/image_302.png')
# input_image = Image.open('./dataset/train/image_060.png')

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
recorrected_image = blur(recorrected_image)

# 出力画像をPIL Imageに変換
recorrected_image = Image.fromarray((recorrected_image[0] * 255).byte().permute(1, 2, 0).numpy())

# オリジナル画像のぼかし
blurred_image = np.array(input_image)
blurred_image = torch.from_numpy(blurred_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
blurred_image = blur(blurred_image)
blurred_image = Image.fromarray((blurred_image[0] * 255).byte().permute(1, 2, 0).numpy())

# # 画像を表示
# blurred_image.show()  # オリジナル画像のぼかし
# recorrected_image.show()  # 補正された画像の再ぼかし


# 画像データの読み込みまたは生成
image1 = np.array(input_image)
image2 = np.array(recorrected_image)
image3 = np.array(blurred_image)
image4 = np.array(output_image)


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

txt3 = tool.image_to_string(
    Image.fromarray(np.uint8(image3)),
    lang='eng',
    builder=pyocr.builders.TextBuilder(tesseract_layout=6)
)

text1 = reader.readtext(image1, detail = 0)
text2 = reader.readtext(image2, detail = 0)
text3 = reader.readtext(image3, detail = 0)


print('original:\n'+'text:\n'+txt1+'\n\n'+'levenshtain:\n'+str(levenshtein_distance(txt1,txt1))+'\n')
print('recrr+brr:\n'+txt2+'\n\n'+'levenshtain:\n'+str(levenshtein_distance(txt1,txt2))+'\n')
print('brr only:\n'+txt3+'\n\n'+'levenshtain:\n'+str(levenshtein_distance(txt1,txt3))+'\n')

print("--------Easy OCR--------------\n\n")

print(text1)
print(text2)
print(text3)


plt.figure(figsize=(15, 5))  # フィギュアのサイズを設定
plt.subplot(2, 2, 1)  # 1行3列の1番目のプロット
plt.imshow(image1)
plt.title("Original")
plt.axis('off')  # 目盛りを非表示にする

plt.subplot(2, 2, 2)  # 1行3列の2番目のプロット
plt.imshow(image2)
plt.title(f"Blurred After Recorrected({args_dict['loss']})")
plt.axis('off')  # 目盛りを非表示にする

plt.subplot(2, 2, 3)  # 1行3列の3番目のプロット
plt.imshow(image3)
plt.title("Blurred Only")
plt.axis('off')  # 目盛りを非表示にする

plt.subplot(2, 2, 4)  # 1行3列の3番目のプロット
plt.imshow(image4)
plt.title("Recorrect Only")
plt.axis('off')  # 目盛りを非表示にする

plt.tight_layout()  # レイアウトの調整
plt.show()