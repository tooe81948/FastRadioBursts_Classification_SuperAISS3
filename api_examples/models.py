import timm
from timm.optim import Lookahead

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchmetrics.classification import MultilabelF1Score
from metrics import PulseRecall

import pytorch_lightning as pl


class MLPLayer(nn.Module):
    def __init__(self, input_size, output_size=3, hidden_sizes=None):
        super(MLPLayer, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 256]

        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # Add ReLU activation only between hidden layers
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class TimmLightningModel(pl.LightningModule):
    def __init__(self, 
                 model_name='resnet18',
                 pretrained=False, 
                 num_classes=3, 
                 learning_rate=1e-3, 
                 hidden_dim=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(model_name, 
                                       pretrained=pretrained, 
                                       num_classes=0)
        self.mlp = MLPLayer(input_size=self.model.num_features, 
                            output_size=num_classes,
                            hidden_sizes=hidden_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()

        # accuracy
        self.train_f1 = MultilabelF1Score(
            num_labels=num_classes, 
            threshold=0.5, 
            average='weighted', 
            multidim_average='global')
        self.val_f1 = MultilabelF1Score(
            num_labels=num_classes, 
            threshold=0.5, 
            average='weighted', 
            multidim_average='global')
        
        # pulse recall
        self.train_pulse_recall = PulseRecall()
        self.val_pulse_recall = PulseRecall()


    def forward(self, x):
        x = self.model(x)
        x = self.mlp(x)
        return x

    def predict(self, x):
        logits = self.forward(x)
        prob = torch.sigmoid(logits)
        return prob
    
    def configure_optimizers(self):
        base_optimizer = torch.optim.Adam(self.parameters(), 
                                          lr=self.hparams.learning_rate)
        optimizer = Lookahead(base_optimizer)
        
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='max', 
                                      factor=0.1, 
                                      patience=10, 
                                      verbose=True)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_frb_accuracy",
            }
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat_sig = torch.sigmoid(y_hat)
        
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', 
                 loss, 
                 on_step=False, 
                 on_epoch=True, 
                 logger=True, 
                 prog_bar=True)
        
        # accuracy
        self.train_f1(y_hat_sig, y)
        accuracy = self.train_f1.compute()
        self.log('train_f1', 
                 accuracy, 
                 on_step=False, 
                 on_epoch=True, 
                 logger=True, 
                 prog_bar=True)
        
        # pulse recall
        self.train_pulse_recall(y_hat, y)
        pulse_recall = self.train_pulse_recall.compute()
        self.log("train_pulse_recall",
                 pulse_recall,
                 on_step=False, 
                 on_epoch=True, 
                 logger=True,
                 prog_bar=True)
        
        # frb accuracy | pulse_recall * accuracy
        frb_accuracy = pulse_recall * accuracy
        self.log('train_frb_accuracy', 
                 frb_accuracy, 
                 on_step=False, 
                 on_epoch=True, 
                 logger=True,
                 prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat_sig = torch.sigmoid(y_hat)
        
        # loss
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', 
                 loss, 
                 on_step=False, 
                 on_epoch=True, 
                 logger=True, 
                 prog_bar=True)
        
        # accuracy
        self.val_f1(y_hat_sig, y)
        accuracy = self.val_f1.compute()
        self.log('val_f1', 
                 accuracy, 
                 on_step=False, 
                 on_epoch=True, 
                 logger=True, 
                 prog_bar=True)

        # pulse recall
        self.val_pulse_recall(y_hat, y)
        pulse_recall = self.val_pulse_recall.compute()
        self.log("val_pulse_recall",
                 pulse_recall,
                 on_step=False, 
                 on_epoch=True, 
                 logger=True,
                 prog_bar=True)
        
        
        # frb accuracy
        frb_accuracy = pulse_recall * accuracy
        self.log('val_frb_accuracy', 
                 frb_accuracy, 
                 on_step=False, 
                 on_epoch=True, 
                 logger=True,
                 prog_bar=True)


    def test_step(self, batch, batch_idx):
        # Implement the test step if needed
        pass
    
