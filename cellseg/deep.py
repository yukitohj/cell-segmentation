import random
import ignite.engine as engine
import segmentation_models_pytorch as smp
import torch
import torch.optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from cellseg.metrics import get_metrics, to_loggable_metrics
from cellseg.preprocess import get_argumentation, get_preprocess
from albumentations import Compose
from cellseg.utils import get_filelists_from_csvs
import mlflow
from .utils.dataset import create_dataset
from PIL import Image


def train(cfg: DictConfig) -> float:
    model = smp.create_model(**cfg.cellseg.model).cuda()

    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.cellseg.lr)

    augumentation = get_argumentation()
    preprocess = get_preprocess(cfg.cellseg.data.img_size)

    train, test = get_filelists_from_csvs(cfg.cellseg.data.train_test, split=True, shuffle=True, seed=cfg.cellseg.seed)

    # eval用にはaugmentationsは適用しない
    train_dataset = create_dataset(train[0], train[1], Compose([augumentation, preprocess]))
    train_eval_dataset = create_dataset(train[0], train[1], preprocess)
    test_eval_dataset = create_dataset(test[0], test[1], preprocess)

    train_loader = DataLoader(train_dataset, batch_size=cfg.cellseg.batch_size)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=cfg.cellseg.batch_size)
    test_eval_loader = DataLoader(test_eval_dataset, batch_size=cfg.cellseg.batch_size)

    metrics = get_metrics(loss_fn)

    trainer = engine.create_supervised_trainer(model, optimizer, loss_fn, "cuda")
    evaluator = engine.create_supervised_evaluator(model, metrics, "cuda")

    @trainer.on(engine.Events.STARTED)
    def log_sample_argumented_images(trainer):
        for i in range(5):
            img, mask = random.choice(train_dataset)
            img = img.numpy().transpose(1, 2, 0)
            mask = mask.numpy()*255
            mlflow.log_image(img, f"sample_img{i}.png")
            mlflow.log_image(mask, f"sample_mask{i}.png")

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
        def save_image(evaluator):
            preds = evaluator.state.output[0]
            masks = evaluator.state.output[1]
            preds = preds.round().cpu().long().numpy()
            masks = masks.cpu().long().numpy()
            for i, pred in enumerate(preds):
                pred = pred.transpose(1, 2, 0)*255
                mask = masks[i]*255
                index = (evaluator.state.iteration-1)*cfg.cellseg.batch_size+i
                mlflow.log_image(pred, f"predicted_{index}.png")
                mlflow.log_image(mask, f"labelmask_{index}.png")

        with evaluator.add_event_handler(engine.Events.ITERATION_COMPLETED, save_image):
            evaluator.run(test_eval_loader)

    trainer.run(train_loader, max_epochs=cfg.cellseg.max_epochs)
    evaluator.run(test_eval_loader)
    return evaluator.state.metrics['miou']
