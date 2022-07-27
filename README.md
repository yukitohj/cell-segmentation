# Cell-Segmentation 細胞画像セグメンテーション深層学習モデル
## 機能
- セマンティックセグメンテーションに関するモデルの訓練やテスト
- コンフィグを指定することで柔軟な動作
- さまざまな画像データ水増し
- EarlyStopping
- Optunaによるハイパーパラメータ探索
- mlflowによるブラウザGUIでのパラメータ・評価指標管理

# スタートガイド
代表的な使い方を述べます．
## 動作環境
- Docker
- CUDA 10.2 以上
- cuDNN 7 以上
- port:5000



## とりあえず動かすなら
```
host$ make docker-build
host$ make docker-run
docker$ python main.py
docker$ mlflow ui -h 0.0.0.0
>ブラウザでマシンのport5000にアクセスするとUIを確認できる
```

## 初期設定の確認
ベースとなるコンフィグは[./conf/config.yaml](./conf/config.yaml)です．

## 設定のカスタマイズ
コンフィグに変更を加える場合は、「変数名=値」を実行時に指定してください．
詳しくは[Hydra](https://hydra.cc/docs/intro/)を御覧ください．
```
docker$ python main.py lr=0.5 model.img_size=256
```

## Optunaの使用
Optunaを使用する場合は、実行時に「-m」オプションを指定してください．
詳しくは、[Optuna Sweeper](https://hydra.cc/docs/plugins/optuna_sweeper/)を御覧ください．
```
docker$ python main.py -m 'max_epochs=choice(10, 20, 30)'
```

## 入力データのカスタマイズ
入力データを管理するためには、以下のようなinput列とlabel列が含まれたCSVファイルが必要です．
これらCSVファイルへのパスをdata.train_testやdata.predictに指定することで、指定された画像がプログラム内で使用されます．
なおlabelファイルがない場合でも空地が入ったlabel列が必要です．
input, label列以外の列が含まれていても問題ありません．
必要に応じて[./tools/filelist2csv.py](./tools/filelist2csv.py)をご利用ください．

| input                           | label                           | ... | 
| ------------------------------- | ------------------------------- | --- | 
| ./data/raw/part1/input/6--1.png | ./data/raw/part1/label/6--1.png | ... | 
| ./data/raw/part1/input/6--2.png | ./data/raw/part1/label/6--2.png | ... | 
| ...                             | ...                             | ... | 


# 仕様

## Docker
プログラムはpytorchにより記述されています．
DockerHubより*pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime*をpullし、それをベースとして利用しています．
開発-デバッグ用の設定が組み込まれています．[Makefile](./Makefile)に記述されたとおりに実行すると、以下のように設定されます．
- ファイルの権限問題のため、Docker内外でユーザーIDとグループIDの統一
- docker内でGithubやGitlabを利用するため、./.sshフォルダ（~/.sshではない）のコピー
- カレントディレクトリのマウント
- gpuすべての有効化
- ポート5000のフォワーディング
- 必要モジュールのインストール

## main.py

### 機能
- データ拡張をふんだんに用いたセマンティックセグメンテーションモデルの訓練
- コンフィグでモデルのアーキテクチャなど指定
- 水増しのサンプル画像の保存
- 各エポックごとに評価指標の記録
- 訓練終了後にtest画像及びpred_only画像の推論結果の保存
- 訓練終了後にモデルの保存
- EarlyStopping
- Optunaによるハイパーパラメータ探索
- 最適化アルゴリズムはAdam固定
- 損失関数はDiceloss固定

### 必要なもの
- configファイル（初期値は[./conf/config.yaml](./conf/config.yaml)）
- configをカスタムする実行時引数
- 入力データに関するCSVファイル.(詳細は[入力データのカスタマイズ](#入力データのカスタマイズ))


### 出力されるもの
- [./outputs/](./outputs/): configのコピーや出力ログ
- [./mlruns](./mlruns/)または[./multirun](./multirun/): 評価指標、モデル、推論結果など. mlflowで閲覧可能.「-m」オプションにより実行した場合はmultirunに保存される．


### データの扱い
コンフィグでは「train_test」データ及び「predict」データの2種類を指定する必要があります．「train_test」データはランダムに7:3に分割され、訓練用、テスト用で利用されます．predictデータは訓練全体終了後に推論だけ行われます．predictデータに対する評価指標は計算されません．predictデータのみ、labelファイルがなくても動作します．

データの拡張は訓練データを訓練に使うときのみ適用されます．評価指標を計算する際には訓練データであってもデータ拡張は行われません．

データ拡張に関しては[./cellseg/preprocess.py](./cellseg/preprocess.py)を御覧ください．



### 評価指標
- DiceCoefficient (test_dice, train_dice)
- JaccardIndex (test_jaccard, train_jaccard)
- meanIoU (test_miou, train_miou)
- DiceLoss (test_loss, train_loss)