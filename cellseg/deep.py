import copy
import os
import random

import ignite
import ignite.engine as engine
import mlflow
import segmentation_models_pytorch as smp
import torch
import torch.optim
from albumentations import Compose
from ignite.contrib.engines.common import save_best_model_by_val_score
from ignite.handlers.early_stopping import EarlyStopping
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .metrics import get_metrics, to_loggable_metrics
from .preprocess import get_augmentation, get_preprocess
from .utils import (ImageWithPathDataset, SegmentationDataset,
                    get_filelists_from_csvs)


def train(cfg: DictConfig) -> dict:

    model = smp.create_model(**cfg.model).cuda()

    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # 訓練にだけ必要な水増し
    augmentation = get_augmentation()
    # すべてのデータに必要な前処理
    preprocess = get_preprocess(cfg.data.img_size)

    train, test = get_filelists_from_csvs(cfg.data.train_test, split=True, shuffle=True)
    only_pred = get_filelists_from_csvs(cfg.data.predict, split=False, shuffle=False)

    train_dataset = SegmentationDataset(train[0], train[1], Compose([augmentation, preprocess]))
    # 訓練データでも評価指標をだすときは水増しは適用しない
    train_eval_dataset = SegmentationDataset(train[0], train[1], preprocess)
    test_eval_dataset = SegmentationDataset(test[0], test[1], preprocess)
    only_pred_dataset = ImageWithPathDataset(only_pred[0], preprocess)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=cfg.batch_size)
    test_eval_loader = DataLoader(test_eval_dataset, batch_size=cfg.batch_size)
    only_pred_loader = DataLoader(only_pred_dataset, batch_size=cfg.batch_size)

    metrics = get_metrics(loss_fn)

    trainer = engine.create_supervised_trainer(model, optimizer, loss_fn, "cuda")
    train_evaluator = engine.create_supervised_evaluator(model, metrics, "cuda")
    test_evaluator = engine.create_supervised_evaluator(model, metrics, "cuda")
    predictor = engine.create_supervised_evaluator(model, {}, "cuda")

    earlystopping = EarlyStopping(cfg.patience, lambda x: x.state.metrics['miou'], trainer, cfg.min_delta)
    test_evaluator.add_event_handler(engine.Events.COMPLETED, earlystopping)

    # 最初に一回だけ実行
    trainer.add_event_handler(engine.Events.STARTED, log_sample_augmented_images, train_dataset, 5)
    # エポック完了ごとに実行
    trainer.add_event_handler(engine.Events.EPOCH_COMPLETED, log_train_metrics, train_eval_loader, train_evaluator)
    trainer.add_event_handler(engine.Events.EPOCH_COMPLETED, log_test_metrics, test_eval_loader, test_evaluator)
    # 最後に一回だけ実行
    trainer.add_event_handler(engine.Events.COMPLETED, log_test_images, test_eval_loader, predictor, cfg.batch_size)
    trainer.add_event_handler(engine.Events.COMPLETED, log_pred_only_images, only_pred_loader, predictor)
    trainer.add_event_handler(engine.Events.COMPLETED, save_model, model, cfg.model.arch)

    trainer.run(train_loader, max_epochs=cfg.max_epochs)

    test_evaluator.run(test_eval_loader)
    final_test_metrics = test_evaluator.state.metrics
    final_test_metrics = to_loggable_metrics(final_test_metrics, 'test')
    return final_test_metrics


def log_sample_augmented_images(trainer, train_dataset, n_samples):
    """augment後の画像数枚をサンプルとして保存します.

    Args:
        n_samples (_type_): サンプルの枚数
    """
    for i in range(n_samples):
        img, mask = random.choice(train_dataset)
        img = img.numpy().transpose(1, 2, 0)
        mask = mask.numpy()*255
        mlflow.log_image(img, f"sample/img{i}.png")
        mlflow.log_image(mask, f"sample/mask{i}.png")


def log_train_metrics(trainer, train_eval_loader, evaluator):
    """train画像に対するmetricsを保存します
    """
    evaluator.run(train_eval_loader)
    metrics = to_loggable_metrics(evaluator.state.metrics, "train")
    mlflow.log_metrics(metrics, trainer.state.epoch)


def log_test_metrics(trainer, test_eval_loader, evaluator):
    """test画像に対するmetricsを保存します
    """
    evaluator.run(test_eval_loader)
    metrics = to_loggable_metrics(evaluator.state.metrics, "test")
    mlflow.log_metrics(metrics, trainer.state.epoch)


def log_test_images(trainer, test_eval_loader, evaluator, batch_size):
    """test画像を推論した結果を保存します.
    """
    def save_images(evaluator):
        preds = evaluator.state.output[0]
        masks = evaluator.state.output[1]
        preds = preds.round().cpu().long().numpy()
        masks = masks.cpu().long().numpy()
        for i, pred in enumerate(preds):
            pred = pred.transpose(1, 2, 0)*255
            mask = masks[i]*255
            index = (evaluator.state.iteration-1)*batch_size+i
            mlflow.log_image(pred, f"test/predicted_{index}.png")
            mlflow.log_image(mask, f"test/labelmask_{index}.png")
    with evaluator.add_event_handler(engine.Events.ITERATION_COMPLETED, save_images):
        evaluator.run(test_eval_loader)


def log_pred_only_images(trainer, only_pred_loader, predictor):
    """predict画像を推論した結果を保存します.
    """
    def save_image(predictor):
        preds = predictor.state.output[0]
        paths = predictor.state.output[1]
        preds = preds.round().cpu().long().numpy()
        for pred, path in zip(preds, paths):
            name = os.path.basename(path)
            pred = pred.transpose(1, 2, 0)*255
            mlflow.log_image(pred, f"predict/{name}.png")
    with predictor.add_event_handler(engine.Events.ITERATION_COMPLETED, save_image):
        predictor.run(only_pred_loader)


def save_model(trainer, model, arch):
    model_path = os.path.join(mlflow.get_artifact_uri(), arch+".pth")
    if model_path.startswith("file://"):
        model_path = model_path[len("file://"):]
    model = copy.deepcopy(model)
    print(model_path)
    torch.save(model.to('cpu').state_dict(), model_path)
