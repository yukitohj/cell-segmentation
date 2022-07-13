from typing import Tuple

import albumentations
import numpy as np
import torch.utils.data
from hydra.utils import to_absolute_path
from PIL import Image


class RawDataset(torch.utils.data.Dataset):
    def __init__(self, input_paths: list, label_paths: list) -> None:
        """入力画像とラベル画像をそれぞれ読み込んでそのまま返却するDataset.

        Args:
            input_paths (list): 入力画像へのパスリスト.
            label_paths (list): ラベル画像へのパスリスト.nanが含まれていてもよい.

        Raises:
            Exception: リストのサイズが異なる
        """

        self._input_paths = input_paths
        self._label_paths = label_paths

        if len(self._input_paths) != len(self._label_paths):
            raise Exception("2つのリストのサイズが異なります")

    def __getitem__(self, index: int) -> Tuple:
        """indexに対応する入力画像とラベル画像をよみこみ、タプルで返します.
        対応するラベル画像が存在しない場合はNoneが返ります.
        入力画像が存在しない場合は例外を投げます．

        Args:
            index (int): 番地

        Returns:
            Tuple: (PIL入力画像, PILラベル画像)
        """

        # 入力画像がないという例外は許さない
        try:
            input = Image.open(to_absolute_path(self._input_paths[index])).convert("RGB")
            input = np.array(input, dtype=np.uint8)
        except:
            raise

        # ラベル画像がないという例外はＯＫ.　predict専用のときなどは、ラベルがない．
        try:
            label = Image.open(to_absolute_path(self._label_paths[index])).convert('L')
            label = np.array(label, dtype=np.uint8)
        except:
            label = None

        return input, label

    def __len__(self) -> int:
        return len(self._input_paths)


class SegmentationDataset(RawDataset):
    """セグメンテーションモデル用のデータセット
    """

    def __init__(self, input_paths: list, label_paths: list, transform: albumentations.BasicTransform = None) -> None:
        super().__init__(input_paths, label_paths)
        self._transform = transform

    def __getitem__(self, index: int) -> Tuple:
        image, mask = super().__getitem__(index)
        if self._transform is not None:
            transformed = self._transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']
        return image, mask

    def __len__(self) -> int:
        return super().__len__()


class ImageWithPathDataset(RawDataset):
    """画像とラベルとしてその画像のパスを返すようなデータセット
    （画像、パス）

    """

    def __init__(self, input_paths: list, label_paths: list, transform: albumentations.BasicTransform = None) -> None:
        super().__init__(input_paths, label_paths)
        self._transform = transform

    def __getitem__(self, index: int) -> Tuple:
        image, _ = super().__getitem__(index)
        path = to_absolute_path(self._input_paths[index])
        if self._transform is not None:
            image = self._transform(image=image)['image']
        # print(image, path)
        return image, path

    def __len__(self) -> int:
        return super().__len__()
