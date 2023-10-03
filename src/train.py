# Import necessary libraries
import torchvision
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import timm
import segmentation_models_pytorch as smp


# Define the model architecture
class ImageClassifier(pl.LightningModule):
    def __init__(self, model_name="resnet18", num_classes=10):
        super(ImageClassifier, self).__init__()
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    # validataion step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

    # test step
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer


# Define the model architecture
class ImageSegmentation(pl.LightningModule):
    def __init__(self, model_name="resnet18", num_classes=2):
        super(ImageSegmentation, self).__init__()
        self.model = smp.Unet(
            model_name, encoder_weights="imagenet", classes=num_classes
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.squeeze(1).long()

        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    # validataion step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.squeeze(1).long()
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

    # test step
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.squeeze(1).long()
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer


# Define the model architecture
class ObjectDetection(pl.LightningModule):
    def __init__(self, num_classes=2):
        super(ObjectDetection, self).__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = (
            y_hat["loss_classifier"]
            + y_hat["loss_box_reg"]
            + y_hat["loss_objectness"]
            + y_hat["loss_rpn_box_reg"]
        )
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer


# import datamodule
from datamodule import get_data_module

import argparse
import omegaconf

# test classification
if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser(description="Configure trainer, model, and data.")

    # add argument
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config YAML file."
    )

    # parse argument
    args = parser.parse_args()

    # load configuration
    config = omegaconf.OmegaConf.load(args.config)

    # sperate config
    # model config
    model_config = config.model
    # data config
    data_config = config.data

    task = data_config.task

    # train config
    trainer = pl.Trainer(**config.trainer)
    # if you want to use config
    # trainer = pl.Trainer(**config.trainer)

    if data_config.task == "classification":
        # data config
        datamodule = get_data_module(**data_config)
        datamodule.setup()
        # model config
        model = ImageClassifier(**model_config)
        trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)
    elif data_config.task == "segmentation":
        # data config
        datamodule = get_data_module(**data_config)
        datamodule.setup()
        # model config
        model = ImageSegmentation()
        trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)
    elif data_config.task == "detection":
        # data config
        datamodule = get_data_module(**data_config)
        datamodule.setup()
        # model config
        model = ObjectDetection(**model_config)
        trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)
