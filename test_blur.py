from models.blur import BlurModel
import numpy as np
from PIL import Image
import torch
import pyocr
import sys
import matplotlib.pyplot as plt



# BlurModel の初期化
blur = BlurModel(S=0.8)


input_image = Image.open('./dataset/image_16.png')

# 画像をPyTorchテンソルに変換
input_image_tensor = torch.from_numpy(np.array(input_image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0



# オリジナル画像のぼかし
blurred_image = np.array(input_image)
blurred_image = torch.from_numpy(blurred_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
blurred_image = blur(blurred_image)
blurred_image = Image.fromarray((blurred_image[0] * 255).byte().permute(1, 2, 0).numpy())


# 画像データの読み込みまたは生成
image1 = np.array(input_image)

image2 = np.array(blurred_image)



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

pyocr.tesseract.TESSERACT_CMD = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("OCRツールが見つかりませんでした")
    sys.exit(1)

tools = pyocr.get_available_tools()





print('input\n'+txt1+'\n\n'+'blured\n'+txt2)

plt.figure(figsize=(15, 5))  # フィギュアのサイズを設定
plt.subplot(1, 2, 1)  # 1行3列の1番目のプロット
plt.imshow(image1)
plt.title("Original")
plt.axis('off')  # 目盛りを非表示にする

plt.subplot(1, 2, 2)  # 1行3列の2番目のプロット
plt.imshow(image2)
plt.title("Blurred")
plt.axis('off')  # 目盛りを非表示にする
plt.tight_layout()  # レイアウトの調整
plt.show()