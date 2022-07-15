from typing import Tuple
from .dataset import SegmentationDataset, ImageWithPathDataset


def get_filelists_from_csvs(csv_paths: list, split=False, shuffle=False, seed=0) -> Tuple:
    """CSVファイルを読み込み、inputファイルパス、labelファイルパスを返します.
    CSVファイルには'input'列及び、'label'列が含まれている必要があります．

    Args:
        csv_paths (list): CSVファイルへのパスリスト
        split (bool, optional): trainとtestに分割するか. Defaults to False.
        shuffle (bool, optional): 要素をシャッフルするか. Defaults to False.
        seed (int, optional): シャッフル時のシード. Defaults to 0.

    Returns:
        Tuple: splitがTrueの場合(train_inputs, train_labels), (test_inputs, test_labels)
        splitがFalse の場合(inputs, labels)
    """
    import pandas as pd
    from hydra.utils import to_absolute_path

    csvs = [pd.read_csv(to_absolute_path(csv_path)) for csv_path in csv_paths]
    df = pd.concat(csvs, axis=0).reset_index(drop=True)

    if shuffle:
        df = df.sample(frac=1, ignore_index=True, random_state=seed)

    if split:
        t = int(len(df)*0.7)
        train = (df[:t]['input'].reset_index(drop=True), df[:t]['label'].reset_index(drop=True))
        test = (df[t:]['input'].reset_index(drop=True), df[t:]['label'].reset_index(drop=True))
        return train, test

    return df['input'].reset_index(drop=True), df['label'].reset_index(drop=True)
