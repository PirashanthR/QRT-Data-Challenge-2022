"""
pytorch lightening encapsulation of models for easier optimization
"""

import pytorch_lightning as pl
from model import Model
from torch.nn import functional as F
import torch

class QRTChallengeRegressor(pl.LightningModule):
    def __init__(self, config):
        super(QRTChallengeRegressor, self).__init__()
        self.lr = config["lr"]

        self.model = Model(config["dropout"])

        self.cel_loss = torch.nn.CosineEmbeddingLoss()
        self.mse_loss = torch.nn.MSELoss()
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred = self.forward(x)

        target_cosine = torch.ones(y.shape[0])
        loss = self.cel_loss(input1=pred.squeeze(2), input2=y.squeeze(2), target = target_cosine)
        mse_loss = self.mse_loss(pred.view(-1), y.view(-1))
        
        self.log("ptl/cel_loss", loss)
        self.log("ptl/mse_loss", mse_loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred = self.forward(x)
        target_cosine = torch.ones(y.shape[0])
        loss = self.cel_loss(input1=pred.squeeze(2), input2=y.squeeze(2), target = target_cosine)        
        mse_loss = self.mse_loss(pred.view(-1), y.view(-1))

        return {"cel_loss": loss, "mse_loss": mse_loss}

    def validation_epoch_end(self, outputs):
        avg_mse_loss = torch.stack(
            [x["mse_loss"] for x in outputs]).mean()
        avg_cel_loss = torch.stack(
            [x["cel_loss"] for x in outputs]).mean()
        self.log("ptl/cel_loss", avg_cel_loss)
        self.log("ptl/mse_loss", avg_mse_loss)