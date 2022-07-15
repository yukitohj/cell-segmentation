import albumentations as albu
import albumentations.pytorch as albu_torch


def get_preprocess(img_size) -> albu.BasicTransform:
    """モデル入力に必要な前処理を返します．

    Args:
        img_size (_type_): 正方形にリサイズする際の一辺の長さ

    Returns:
        albu.BasicTransform: _description_
    """
    return albu.Compose([
        albu.Resize(512, 512),
        albu.Normalize(),
        albu.Lambda(mask=lambda x, **kwargs: x//255),
        albu_torch.ToTensorV2()
    ])


def get_augmentation() -> albu.BasicTransform:
    """データ拡張処理を返します．
    """
    flip = albu.Flip()
    rotate = albu.RandomRotate90()
    blur = albu.OneOf([
        albu.Blur(always_apply=True),
        albu.MotionBlur(always_apply=True),
        albu.GaussianBlur(always_apply=True),
        albu.GlassBlur(always_apply=True)
    ])
    noise = albu.OneOf([
        albu.GaussNoise(always_apply=True),
        albu.JpegCompression(always_apply=True),
        albu.ISONoise(always_apply=True),
        albu.MultiplicativeNoise(always_apply=True),
        albu.Downscale(always_apply=True)
    ])
    distortion = albu.OneOf([
        albu.OpticalDistortion(always_apply=True),
        albu.GridDistortion(always_apply=True),
        albu.ElasticTransform(always_apply=True),
    ])
    to_gray = albu.ToGray()
    gamma = albu.RandomGamma()
    brightness = albu.RandomBrightness()
    contrast = albu.RandomContrast()
    coarse_dropout = albu.CoarseDropout(max_height=32, max_width=32, fill_value=0)
    pixel_dropout = albu.PixelDropout()

    return albu.Compose([
        flip, rotate, blur, noise, distortion, to_gray, gamma, brightness, contrast, coarse_dropout, pixel_dropout
    ])
