import torch

from models.blur import BlurModel
from models.model import LitImageCorrection

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


@st.cache_resource
def load_models():
    """
    Load the blur model.
    """
    try:
        blur_model = BlurModel(S=args_dict["sphere"])
        blur_model.eval()
    except Exception as e:
        st.error(f"ぼかしモデルの初期化中にエラーが発生しました: {e}")
        st.stop()

    return blur_model

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

    # ユーザー入力フォームの追加
    st.sidebar.header("補正パラメータの入力")
    with st.sidebar.form("parameters_form"):
        sp = st.number_input(
            label="SP",
            min_value=0,
            max_value=1000,
            value=64,
            step=1,
            help="空間周波数を指定します。"
        )

        sphere = st.number_input(
            label="Sphere",
            min_value=-10.0,
            max_value=10.0,
            value=0.8,
            step=0.1,
            help="球面度数を入力します。"
        )

        cylinder = st.number_input(
            label="Cylinder",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            help="円柱度数を入力します。"
        )

        axis = st.number_input(
            label="Axis",
            min_value=0,
            max_value=180,
            value=0,
            step=1,
            help="円柱軸の方向を入力します。"
        )

        radius = st.number_input(
            label="Radius",
            min_value=0.1,
            max_value=10.0,
            value=1.5,
            step=0.1,
            help="半径を入力します。"
        )

        submitted = st.form_submit_button("パラメータを適用")

    if submitted:
        # 入力されたパラメータを更新
        args.sp = sp
        args.sphere = sphere
        args.cylinder = cylinder
        args.axis = axis
        args.radius = radius

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
            blur_model = load_models()

            # 画像の前処理
            input_image = preprocess_image(uploaded_file)
            input_image_tensor = torch.from_numpy(np.array(input_image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

            # ぼかしモデルの適用
            with torch.no_grad():
                blurred_tensor = blur_model(input_image_tensor)
            blurred_image = postprocess_tensor(blurred_tensor)

        # 画像の表示
        st.markdown("### 画像の比較")
        cols = st.columns(2)
        with cols[0]:
            st.image(input_image,  use_column_width=True)
            st.markdown('<div class="image-caption">入力画像</div>', unsafe_allow_html=True)
        with cols[1]:
            st.image(blurred_image, use_column_width=True)
            st.markdown('<div class="image-caption">資格再現した画像</div>', unsafe_allow_html=True)


        # 説明文や追加情報の表示
        st.markdown("""
        ---
        **補正プロセスについて:**
        - **ぼかした入力画像:** 入力画像に屈折異常を模擬したぼかしを適用。
        """)
    else:
        st.info("画像をアップロードするか、カメラで撮影してください。")

if __name__ == "__main__":
    main()
