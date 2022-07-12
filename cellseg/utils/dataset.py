from typing import Tuple
import torch.utils.data
from PIL import Image
from hydra.utils import to_absolute_path
import albumentations
import pandas as pd
import numpy as np


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
        except:
            raise

        # ラベル画像がないという例外はＯＫ.　predict専用のときなどは、ラベルがない．
        try:
            label = Image.open(to_absolute_path(self._label_paths[index])).convert('L')
        except:
            label = None

        return np.array(input, dtype=np.uint8), np.array(label, dtype=np.uint8)

    def __len__(self) -> int:
        return len(self._input_paths)


class SegmentationDataset(torch.utils.data.Dataset):
    """セグメンテーションモデル用のデータセット
    """

    def __init__(self, dataset: torch.utils.data.Dataset, augmentation: albumentations.BasicTransform = None) -> None:
        """セグメンテーション用処理を行ったデータセットを返します．

        Args:
            dataset (torch.utils.data.Dataset): 単純にPIL画像を返すようなDataset. RawDatasetを推奨.
            augmentation (_type_): albumentationsで作成したtransformer
        """
        self._dataset = dataset
        self._argument = augmentation

    def __getitem__(self, index) -> Tuple:
        image, mask = self._dataset[index]
        if self._argument is not None:
            argumented = self._argument(image=image, mask=mask)
            image, mask = argumented['image'], argumented['mask']

        return image, mask

    def __len__(self) -> int:
        return len(self._dataset)


def create_dataset(input_paths: list, label_paths: list, augmentation: albumentations.BasicTransform) -> torch.utils.data.Dataset:
    """セグメンテーション用のデータセットを返します．

    Args:
        input_paths (list): 入力画像へのパスリスト．
        label_paths (list): ラベル画像へのパスリスト.nanが含まれていてもよい.
        augmentation (albumentations.BasicTransform): albumentationsで作成したtransformer

    Returns:
        torch.utils.data.Dataset: データセット
    """
    dataset = RawDataset(input_paths, label_paths)
    dataset = SegmentationDataset(dataset, augmentation)
    return dataset
