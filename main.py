import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger

from models.blur import BlurModel
from models.model import LitImageCorrection
from datasets import ImageDataModule

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

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

@st.cache_resource
def load_models():
    """
    Load the image correction model and the blur model.
    """
    checkpoint_path = "./lightning_logs/default/MSE+OCR/checkpoints/epoch=210-step=10128.ckpt"
    try:
        model = LitImageCorrection.load_from_checkpoint(checkpoint_path=checkpoint_path, args=args)
        model.eval()
    except Exception as e:
        st.error(f"モデルの読み込み中にエラーが発生しました: {e}")
        st.stop()

    try:
        blur_model = BlurModel(S=args_dict["sphere"])
        blur_model.eval()
    except Exception as e:
        st.error(f"ぼかしモデルの初期化中にエラーが発生しました: {e}")
        st.stop()

    return model, blur_model

def preprocess_image(uploaded_file):
    """
    Preprocess the uploaded image for the model.
    """
    try:
        image = Image.open(uploaded_file)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        st.error(f"画像の処理中にエラーが発生しました: {e}")
        st.stop()

def postprocess_tensor(tensor):
    """
    Convert a tensor back to a PIL image.
    """
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = torch.clamp(tensor, 0.0, 1.0)
    tensor = tensor.permute(1, 2, 0).numpy() * 255.0
    tensor = tensor.astype(np.uint8)
    return Image.fromarray(tensor)

def main():
    # ページ設定とスタイルのカスタマイズ
    st.set_page_config(
        page_title="屈折異常画像補正アプリ",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # カスタムCSSの適用
    st.markdown(
        """
        <style>
        .title {
            font-size: 2.5rem;
            text-align: center;
            color: #4CAF50;
            margin-bottom: 20px;
        }
        .description {
            font-size: 1.2rem;
            text-align: center;
            color: #555555;
            margin-bottom: 40px;
        }
        .image-caption {
            font-size: 1rem;
            text-align: center;
            color: #333333;
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # タイトルと説明
    st.markdown('<div class="title">🔍 屈折異常の為の画像補正アプリ</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">この技術は屈折異常を持つ人が裸眼で見られるようにニューラルネットワークを用いて画像を補正する技術です。2023年度の山之上暢の修論の成果物です。</div>', unsafe_allow_html=True)

    # サイドバーでの画像ソース選択
    img_source = st.sidebar.radio(
        "画像のソースを選択してください",
        ("画像をアップロード", "画像を撮影")
    )

    # ファイルアップローダーまたはカメラ入力
    if img_source == "画像をアップロード":
        uploaded_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg", "jpeg"])
    else:
        uploaded_file = st.camera_input("カメラで撮影")

    if uploaded_file is not None:
        # ローディングインジケーターの開始
        with st.spinner('画像を処理中...'):
            # モデルの読み込み
            model, blur_model = load_models()

            # 画像の前処理
            input_image = preprocess_image(uploaded_file)
            input_image_tensor = torch.from_numpy(np.array(input_image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

            # ぼかしモデルの適用
            with torch.no_grad():
                blurred_tensor = blur_model(input_image_tensor)
            blurred_image = postprocess_tensor(blurred_tensor)

            # 補正モデルの適用
            with torch.no_grad():
                corrected_tensor = model(input_image_tensor)
            corrected_image = postprocess_tensor(corrected_tensor)

            # # 再度ぼかしモデルの適用
            # with torch.no_grad():
            #     recorrected_tensor = blur_model(corrected_tensor)
            # recorrected_image = postprocess_tensor(recorrected_tensor)

        # 画像の表示
        st.markdown("### 画像の比較")
        cols = st.columns(3)
        with cols[0]:
            st.image(input_image,  use_column_width=True)
            st.markdown('<div class="image-caption">入力画像</div>', unsafe_allow_html=True)
        with cols[1]:
            st.image(blurred_image,  use_column_width=True)
            st.markdown('<div class="image-caption">ぼかした入力画像</div>', unsafe_allow_html=True)
        with cols[2]:
            st.image(corrected_image, use_column_width=True)
            st.markdown('<div class="image-caption">AIで補正された画像</div>', unsafe_allow_html=True)
        # with cols[2]:
        #     st.image(recorrected_image, use_column_width=True)
        #     st.markdown('<div class="image-caption">AIで補正した画像をぼかした画像</div>', unsafe_allow_html=True)

        # 説明文や追加情報の表示
        st.markdown("""
        ---
        **補正プロセスについて:**
        - **入力画像:** 入力画像。                    
        - **ぼかした入力画像:** 入力画像に屈折異常を模擬したぼかしを適用。
        - **補正された画像:** ニューラルネットワークによって補正された画像。
        """)
    else:
        st.info("画像をアップロードするか、カメラで撮影してください。")

if __name__ == "__main__":
    main()
