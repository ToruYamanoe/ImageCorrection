import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import os
import random
import string
import numpy as np


def add_noise(img):
    """ ノイズを追加する関数 """
    arr = np.array(img)

    noise = np.random.randint(-5, 50, arr.shape)
    img = PIL.Image.fromarray(np.clip(arr + noise, 0, 255).astype('uint8'))
    return img

# 画像生成ループ
canvasSize = (500, 500)
ttfontname = './datasets_characters/font/ubuntu.ttf'
initial_fontsize = 80
font_step = 8
left_margin = canvasSize[0] // 25

for i in range(600):
    img = PIL.Image.new('RGB', canvasSize, (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)))  # 白背景
    img = add_noise(img)
    draw = PIL.ImageDraw.Draw(img)

    current_fontsize = initial_fontsize
    y = 0
    line_count = 0  # 行数をカウントする変数
    # while y < canvasSize[1] - 1 * initial_fontsize:  # 最後の1行を描画しないようにする
    while y < canvasSize[1]:
        textRGB = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        font = PIL.ImageFont.truetype(ttfontname, current_fontsize)
        random_text = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(10, 20)))
        textWidth = draw.textlength(random_text, font=font)
        textHeight = current_fontsize
        textTopLeft = (left_margin, y)
        draw.text(textTopLeft, random_text, fill=textRGB, font=font)
        y += textHeight + 10
        line_count += 1
        current_fontsize -= font_step
        if current_fontsize < 25:
            break

    filename = f'image_{i + 1:03d}.png'
    img.save(os.path.join('dataset/train/', filename))

print('Images have been saved.')



