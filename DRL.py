import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from PIL import Image
import pyocr
from torchvision import transforms
import os
import numpy as np
import pandas as pd
import pyocr
import random
import argparse

from models.srcnn import SRCNN
from models.vdsr import VDSR
from models.swinir import SwinIR
from models.unet import UNet
from models.blur import BlurModel

import glob

# 画像が存在するディレクトリのパスを指定
image_directory = "./dataset"

# そのディレクトリ内の全画像ファイルのパスを取得
image_paths = glob.glob(f"{image_directory}/*.png")  


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


# コマンドライン引数の取得
def get_args():
    parser = argparse.ArgumentParser(description="Image Correction Model Training")
    parser.add_argument("--loss", type=str, choices=["mse", "l1", "mae", "psnr", "ssim", "tv","original"], default="l1")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sp", type=int, default=64)
    parser.add_argument("--model", type=str, default="srcnn", choices=["srcnn", "vdsr", "swinir", "unet"])
    parser.add_argument("--sphere", type=float, default=1.0)
    parser.add_argument("--cylinder", type=float, default=0.0)
    parser.add_argument("--axis", type=int, default=0)
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument("--img_shape", type=int, nargs=3, default=[3, 500, 500])
    parser.add_argument("--wide_range", action="store_true", help="Use wide range for ...")

    return parser.parse_args()

args = get_args()



class ImageCorrectionEnv(gym.Env):
    def __init__(self, model_name, blur_model, img_shape, image_paths, wide_range=args.wide_range,device='cpu'):
        super().__init__()
        self.image_paths = image_paths
        self.img_shape = img_shape
        self.blur_model = blur_model
        self.wide_range = wide_range
        pyocr.tesseract.TESSERACT_CMD = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        self.tools = pyocr.get_available_tools()
        self.tool = self.tools[0] if self.tools else None

        # モデルの選択
        if model_name == "srcnn":
            self.correction_model = SRCNN(img_shape).to(device)
        elif model_name == "vdsr":
            self.correction_model = VDSR(img_shape, init_weights=True).to(device)
        elif model_name == "swinir":
            self.correction_model = SwinIR(img_size=img_shape[-1], upscale=1, window_size=8).to(device)
        elif model_name == "unet":
            self.correction_model = UNet(n_channels=img_shape[0]).to(device)
        else:
            raise ValueError("Invalid model name")

    def step(self, action):
        # Adopt correction-model & blur-model
        corrected_img = self.correction_model(action)
        blurred_corrected_img = self.blur_model(corrected_img)

        # use OCR to extract text
        y_pred = self.extract_text(blurred_corrected_img)
        y_true = self.extract_text(action)

        # calculate reward
        reward = self.calculate_reward(y_true, y_pred)
        new_state = blurred_corrected_img
        done = levenshtein_distance(y_true,y_pred) <= 1

        return new_state, reward, done, {}

    def extract_text(self, img_tensor):
        # TensorをPIL Imageに変換
        img = transforms.ToPILImage()(img_tensor.cpu().squeeze(0).byte())
        # OCRでテキスト抽出
        
        tools = pyocr.get_available_tools()
        text = self.tool.image_to_string(img, lang='eng', builder=pyocr.builders.TextBuilder(tesseract_layout=6))
        return text

    def calculate_reward(self, y_true, y_pred):
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0

        match_count = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        reward = match_count / max(len(y_true), len(y_pred))
        return reward

    def reset(self):
        initial_state = self.load_new_image()
        return initial_state

    def load_new_image(self):
        image_path = random.choice(self.image_paths)
        image = Image.open(image_path)
        image = self.transform_image(image)
        return image

    def transform_image(self, image):
        transform = transforms.Compose([
            transforms.Resize(self.img_shape[1:]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)



##Agent
class CorrectionAgent:
    def __init__(self, env, model, lr=0.001, device='cpu'):
        self.env = env
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update(state, action, reward)

    def choose_action(self, state):
        state = state.unsqueeze(0).to(self.device)  # ステートをデバイスに移動
        action = self.model(state)
        return action.squeeze(0)

    def update(self, original_img, action, reward):
        original_img = original_img.to(self.device)  # 元の画像をデバイスに移動
        action = action.to(self.device)  # アクションをデバイスに移動
        mse_loss = F.mse_loss(action, original_img)
        adjusted_loss = mse_loss * (1 - reward)
        self.optimizer.zero_grad()
        adjusted_loss.backward()
        self.optimizer.step()


#main train_loop
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blur_model_args = {
        'S': args.sphere,
        'C': args.cylinder,
        'A': args.axis,
        'R': args.radius,
        'sp': args.sp,
        'img_shape': args.img_shape
    }
    blur_model = BlurModel(**blur_model_args)
    env = ImageCorrectionEnv(args.model, blur_model,  args.img_shape, image_paths,wide_range=args.wide_range)
    

    model = SRCNN(args.img_shape).to(device)
    agent = CorrectionAgent(env, model, device=device)

    num_episodes=500
    for episode in range(num_episodes):
        original_img = env.reset()  # 元の画像
        done = False
        total_reward = 0
        step_count = 0  # ステップ数のカウント

        while not done:
            action = agent.choose_action(original_img)
            next_state, reward, done, _ = env.step(action)
            agent.update(original_img, action, reward)
            total_reward += reward
            step_count += 1

            # 進行状況の表示
            if step_count % 100 == 0:
                print(f"エピソード {episode}, ステップ {step_count}, 総報酬: {total_reward}")

        print(f"エピソード: {episode}, 総報酬: {total_reward}, ステップ数: {step_count}")

    torch.save(model.state_dict(), "image_correction_model.pth")

if __name__ == "__main__":
    main()