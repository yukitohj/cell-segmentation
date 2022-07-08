import albumentations as albu
import albumentations.pytorch as albu_torch
import ignite
import ignite.engine as engine
import ignite.metrics
import segmentation_models_pytorch as smp
import torch
import torch.optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from cellseg.metrics import get_metrics

from cellseg.utils import read_csvs

from .utils.dataset import create_dataset


def train(cfg: DictConfig):
    model_cfg = cfg.cellseg.model
    model = smp.create_model(**model_cfg)

    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.cellseg.lr)

    augmentation = albu.Compose([
        albu.Resize(512, 512),
        albu.Normalize(),
        albu.Lambda(mask=lambda x, **kwargs: x//255),
        albu_torch.ToTensorV2(transpose_mask=True)
    ])

    train, test = read_csvs(cfg.cellseg.data.train_test, split=True, shuffle=True, seed=cfg.cellseg.seed)
    train_dataset = create_dataset(train[0], train[1], augmentation)
    train_eval_dataset = create_dataset(train[0], train[1], augmentation)
    test_eval_dataset = create_dataset(test[0], test[1], augmentation)

    train_loader = DataLoader(train_dataset, batch_size=4)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=4)
    test_eval_loader = DataLoader(test_eval_dataset, batch_size=4)

    metrics = get_metrics(loss_fn)

    trainer = engine.create_supervised_trainer(model, optimizer, loss_fn)
    evaluator = engine.create_supervised_evaluator(model, metrics=metrics)

    @trainer.on(engine.Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")

    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_eval_loader)
        metrics = evaluator.state.metrics
        print('train', metrics)

    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(test_eval_loader)
        metrics = evaluator.state.metrics
        print('test', metrics)

    trainer.run(train_loader, max_epochs=cfg.cellseg.max_epochs)
