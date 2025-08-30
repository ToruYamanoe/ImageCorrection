import torch
import torch.nn.functional as F
import torchmetrics

# PSNR / SSIM (torchmetrics)
_psnr_fn = torchmetrics.functional.peak_signal_noise_ratio
_ssim_fn = torchmetrics.functional.structural_similarity_index_measure

def zernike_cycle_loss(corr_imgs, imgs, z, blur_model, loss_type="l1"):
    """
    corr_imgs: 補正モデルの出力 (補正後画像)
    imgs: クリーン画像
    z: ゼルニケ係数 (バッチごとに渡す)
    blur_model: BlurModel instance
    loss_type: "l1", "mse", "psnr", "ssim"
    """
    # 出力画像を BlurModel に通す（視覚再現）
    blurred_corr = blur_model(corr_imgs, z)

    if loss_type == "l1":
        return F.l1_loss(blurred_corr, imgs)
    elif loss_type == "mse":
        return F.mse_loss(blurred_corr, imgs)
    elif loss_type == "psnr":
        # 高い方が良いので 1 - 値 にして minimization 互換にする
        return 1.0 - _psnr_fn(blurred_corr, imgs, data_range=1.0)
    elif loss_type == "ssim":
        # SSIM も高い方が良い → 1 - 値
        return 1.0 - _ssim_fn(blurred_corr, imgs, data_range=1.0)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

