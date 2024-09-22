
import torch
import torch.nn.functional as F
from torchvision import transforms
import pyocr
import sys
import torchmetrics


def Legibility_Loss(original_img, corrected_img,alpha=1.0,beta=0.5):

    def Image_Legibility_Loss(brr_img, img):


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
        
        # OCRツールの設定
        pyocr.tesseract.TESSERACT_CMD = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            print("OCR tools not found")
            sys.exit(1)
        tool = tools[0]

        # 画像をPIL形式に変換
        img = transforms.ToPILImage()((img[0].cpu() * 255).byte().permute(1, 2, 0).numpy())
        brr_img = transforms.ToPILImage()((brr_img[0].cpu() * 255).byte().permute(1, 2, 0).numpy())

        # OCRでテキスト抽出
        y_true = tool.image_to_string(img, lang='eng', builder=pyocr.builders.TextBuilder(tesseract_layout=6)).lower()
        y_pred = tool.image_to_string(brr_img, lang='eng', builder=pyocr.builders.TextBuilder(tesseract_layout=6)).lower()

        # テキストの一致度に基づいて報酬を計算
        # ここでは例として、一致する文字数に基づく報酬を使用します。
        loss = levenshtein_distance(y_true,y_pred)/ (max(len(y_true), len(y_pred))+1)
        return loss
    
    mse_loss = F.mse_loss(corrected_img, original_img)
    ocr_loss = Image_Legibility_Loss(corrected_img, original_img)
    adjusted_loss = alpha*mse_loss + beta*ocr_loss  # 報酬が高いほど損失を減らす
    return adjusted_loss
    



def Adjusted_MSE_Loss(original_img, corrected_img):

    def Image_Legibility_Reward(brr_img, img):


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
        
        # OCRツールの設定
        pyocr.tesseract.TESSERACT_CMD = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            print("OCR tools not found")
            sys.exit(1)
        tool = tools[0]

        # 画像をPIL形式に変換
        img = transforms.ToPILImage()((img[0].cpu() * 255).byte().permute(1, 2, 0).numpy())
        brr_img = transforms.ToPILImage()((brr_img[0].cpu() * 255).byte().permute(1, 2, 0).numpy())

        # OCRでテキスト抽出
        y_true = tool.image_to_string(img, lang='eng', builder=pyocr.builders.TextBuilder(tesseract_layout=6)).lower()
        y_pred = tool.image_to_string(brr_img, lang='eng', builder=pyocr.builders.TextBuilder(tesseract_layout=6)).lower()

        # テキストの一致度に基づいて報酬を計算
        # ここでは例として、一致する文字数に基づく報酬を使用します。
        reward = 1 - levenshtein_distance(y_true,y_pred)/ max(len(y_true), len(y_pred))
        return reward
    
    mse_loss = F.mse_loss(corrected_img, original_img)
    reward = Image_Legibility_Reward(corrected_img, original_img)
    adjusted_loss = mse_loss * (2 - reward)  # 報酬が高いほど損失を減らす
    return adjusted_loss


def Adjusted_PSNR_Loss(original_img, corrected_img):
    def Image_Legibility_Reward(brr_img, img):


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
        
        # OCRツールの設定
        pyocr.tesseract.TESSERACT_CMD = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            print("OCR tools not found")
            sys.exit(1)
        tool = tools[0]

        # 画像をPIL形式に変換
        img = transforms.ToPILImage()((img[0].cpu() * 255).byte().permute(1, 2, 0).numpy())
        brr_img = transforms.ToPILImage()((brr_img[0].cpu() * 255).byte().permute(1, 2, 0).numpy())

        # OCRでテキスト抽出
        y_true = tool.image_to_string(img, lang='eng', builder=pyocr.builders.TextBuilder(tesseract_layout=6))
        y_pred = tool.image_to_string(brr_img, lang='eng', builder=pyocr.builders.TextBuilder(tesseract_layout=6))

        # テキストの一致度に基づいて報酬を計算
        # ここでは例として、一致する文字数に基づく報酬を使用します。
        reward = 1 - levenshtein_distance(y_true,y_pred)/ max(len(y_true), len(y_pred))
        return reward
    
    psnr = torchmetrics.functional.structural_similarity_index_measure(corrected_img, original_img)
    reward = Image_Legibility_Reward(corrected_img, original_img)
    adjusted_loss = (1 - psnr / 100.0) * weight* (1 - reward)  # 報酬が高いほど損失を減らす
    return adjusted_loss


def Adjusted_SSIM_Loss(original_img, corrected_img):

    def Image_Legibility_Reward(brr_img, img):


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
        
        # OCRツールの設定
        pyocr.tesseract.TESSERACT_CMD = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            print("OCR tools not found")
            sys.exit(1)
        tool = tools[0]

        # 画像をPIL形式に変換
        img = transforms.ToPILImage()((img[0].cpu() * 255).byte().permute(1, 2, 0).numpy())
        brr_img = transforms.ToPILImage()((brr_img[0].cpu() * 255).byte().permute(1, 2, 0).numpy())

        # OCRでテキスト抽出
        y_true = tool.image_to_string(img, lang='eng', builder=pyocr.builders.TextBuilder(tesseract_layout=6))
        y_pred = tool.image_to_string(brr_img, lang='eng', builder=pyocr.builders.TextBuilder(tesseract_layout=6))

        # テキストの一致度に基づいて報酬を計算
        # ここでは例として、一致する文字数に基づく報酬を使用します。
        reward = 1 - levenshtein_distance(y_true,y_pred)/ max(len(y_true), len(y_pred))
        return reward
    
    ssim_value = torchmetrics.functional.structural_similarity_index_measure(original_img, corrected_img)
    reward = Image_Legibility_Reward(original_img,corrected_img)

    loss = 0.5 * (1 - ssim_value) + 0.5 * (1 - reward)
    return loss



