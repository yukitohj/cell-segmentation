import os
import random

import ignite.engine as engine
import mlflow
import segmentation_models_pytorch as smp
import torch
import torch.optim
from albumentations import Compose
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .metrics import get_metrics, to_loggable_metrics
from .preprocess import get_argumentation, get_preprocess
from .utils import get_filelists_from_csvs, ImageWithPathDataset, SegmentationDataset


def train(cfg: DictConfig) -> float:
    model = smp.create_model(**cfg.model).cuda()

    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    augumentation = get_argumentation()
    preprocess = get_preprocess(cfg.data.img_size)

    train, test = get_filelists_from_csvs(cfg.data.train_test, split=True, shuffle=True, seed=cfg.seed)
    only_pred = get_filelists_from_csvs(cfg.data.predict, split=False, shuffle=False)

    # eval用にはaugmentationsは適用しない
    # pred用はラベルがないこと前提
    train_dataset = SegmentationDataset(train[0], train[1], Compose([augumentation, preprocess]))
    train_eval_dataset = SegmentationDataset(train[0], train[1], preprocess)
    test_eval_dataset = SegmentationDataset(test[0], test[1], preprocess)
    only_pred_dataset = ImageWithPathDataset(only_pred[0], only_pred[1], preprocess)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=cfg.batch_size)
    test_eval_loader = DataLoader(test_eval_dataset, batch_size=cfg.batch_size)
    only_pred_loader = DataLoader(only_pred_dataset, batch_size=cfg.batch_size)

    metrics = get_metrics(loss_fn)

    trainer = engine.create_supervised_trainer(model, optimizer, loss_fn, "cuda")
    evaluator = engine.create_supervised_evaluator(model, metrics, "cuda")
    predictor = engine.create_supervised_evaluator(model, {}, "cuda")

    @trainer.on(engine.Events.STARTED)
    def log_sample_argumented_images(trainer):
        for i in range(5):
            img, mask = random.choice(train_dataset)
            img = img.numpy().transpose(1, 2, 0)
            mask = mask.numpy()*255
            mlflow.log_image(img, f"sample/img{i}.png")
            mlflow.log_image(mask, f"sample/mask{i}.png")

    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_eval_loader)
        metrics = to_loggable_metrics(evaluator.state.metrics, "train")
        mlflow.log_metrics(metrics, trainer.state.epoch)

    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(test_eval_loader)
        metrics = to_loggable_metrics(evaluator.state.metrics, "test")
        mlflow.log_metrics(metrics, trainer.state.epoch)

    @trainer.on(engine.Events.COMPLETED)
    def log_entire_results(trainer):
        def save_images(evaluator):
            preds = evaluator.state.output[0]
            masks = evaluator.state.output[1]
            preds = preds.round().cpu().long().numpy()
            masks = masks.cpu().long().numpy()
            for i, pred in enumerate(preds):
                pred = pred.transpose(1, 2, 0)*255
                mask = masks[i]*255
                index = (evaluator.state.iteration-1)*cfg.batch_size+i
                mlflow.log_image(pred, f"test/predicted_{index}.png")
                mlflow.log_image(mask, f"test/labelmask_{index}.png")

        def save_image(predictor):
            preds = predictor.state.output[0]
            paths = predictor.state.output[1]
            preds = preds.round().cpu().long().numpy()
            for pred, path in zip(preds, paths):
                name = os.path.basename(path)
                pred = pred.transpose(1, 2, 0)*255
                mlflow.log_image(pred, f"predict/{name}.png")

        with evaluator.add_event_handler(engine.Events.ITERATION_COMPLETED, save_images):
            evaluator.run(test_eval_loader)
        with predictor.add_event_handler(engine.Events.ITERATION_COMPLETED, save_image):
            predictor.run(only_pred_loader)

    trainer.run(train_loader, max_epochs=cfg.max_epochs)
    evaluator.run(test_eval_loader)
    return evaluator.state.metrics['miou']
