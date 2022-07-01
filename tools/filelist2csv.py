from cProfile import label
import pandas as pd
import sys
import os
from glob import glob

if __name__ == "__main__":
    """
    input/ と labels/ があるディレクトリへのパスを第一引数で指定する．
    input/以下のパスとlabels/以下のパスが同じpng,tif,tiffファイルをペアとして認識し、csvを同ディレクトリに生成する.
    """

    path = sys.argv[1]

    # png, tif, tiffをすべて探索
    inputs = []
    labels = []
    for ext in ["png", "tif", "tiff"]:
        inputs.extend(glob(os.path.join(path, f"input/**/*.{ext}"), recursive=True))
        labels.extend(glob(os.path.join(path, f"label/**/*.{ext}"), recursive=True))

    # {input/以下のパス : 全体のパス} なるmapを生成する
    input_common_path_len = len(os.path.join(path, "input"))
    label_common_path_len = len(os.path.join(path, "label"))

    inputs = {x[input_common_path_len:]: x for x in inputs}
    labels = {x[label_common_path_len:]: x for x in labels}

    #  mapをDataframeで管理する．"input/以下のパス"をインデックスとして結合することで、
    #  ペアを見つけるとともに、ペアを作れなかった対応画像にはNanが入る．
    inputs = pd.DataFrame.from_dict(inputs, orient='index', columns=["input"])
    labels = pd.DataFrame.from_dict(labels, orient='index', columns=["label"])

    table = pd.concat([inputs, labels], axis=1)
    table = table.reset_index(drop=True)

    table.to_csv(os.path.join(path, "pair.csv"))
    print(table)
