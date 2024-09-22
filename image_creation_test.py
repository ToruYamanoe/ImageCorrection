from bs4 import BeautifulSoup
import urllib.request as req
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import textwrap
import os 
# url = "https://www.gutenberg.org/cache/epub/71739/pg71739-images.html"

# # urlopen()でデータを取得
# res = req.urlopen(url)
# # BeautifulSoup()で解析
# soup = BeautifulSoup(res, 'html.parser')

# # print(soup)
# div_elements = soup.find_all('div')
# text = '\n'.join([elem.get_text() for elem in div_elements])

# テキストを出力
text = 'Neural Network'
print(text)



# 使うフォント，サイズ，描くテキストの設定
ttfontname = './data/ubuntu.ttf'
fontsize = 36


# 画像サイズ，背景色，フォントの色を設定
canvasSize = (500, 500)
backgroundRGB = (255, 255, 255)
textRGB = (0, 0, 0)

# 文字を描く画像の作成
img = PIL.Image.new('RGB', canvasSize, backgroundRGB)
draw = PIL.ImageDraw.Draw(img)

# フォントの設定
font = PIL.ImageFont.truetype(ttfontname, fontsize)

# テキストを折り返す
wrapper = textwrap.TextWrapper(width=22)  # 22文字ごとに折り返す例
text_lines = wrapper.wrap(text)

# テキストを描画
y = (canvasSize[1] - sum(draw.textsize(line, font=font)[1] for line in text_lines)) // 2  # テキストを中央に配置する計算
for line in text_lines:
    textWidth, textHeight = draw.textsize(line, font=font)
    textTopLeft = ((canvasSize[0] - textWidth) // 2, y)  # テキストを中央に配置
    draw.text(textTopLeft, line, fill=textRGB, font=font)
    y += textHeight

img.save("image.png")