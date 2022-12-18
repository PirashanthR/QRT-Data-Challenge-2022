"""
pytorch lightening encapsulation of data for easier optimization
"""

import pandas as pd
import os
from sklearn.model_selection import KFold
from data import MyDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class QRTDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            k: int = 1,  # fold number #to be done optimization using cross validation
            split_seed: int = 12345,  # split needs to be always the same for correct cross validation
            num_splits: int = 10,
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False
        ):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.data_dir = data_dir
        # num_splits = 10 means our dataset will be split to 10 parts
        # so we train on 90% of the data and validate on 10%
        assert 1 <= self.hparams.k <= self.hparams.num_splits, "incorrect fold number"

    def setup(self, stage=None):
        """Split train and validation data 
        """
        X = pd.read_csv(os.path.join(self.data_dir, 'X_train_YG7NZSq.csv'), index_col=0, sep=',')
        X.columns.name = 'date'

        Y = pd.read_csv(os.path.join(self.data_dir, 'Y_train_wz11VM6.csv'), index_col=0, sep=',')
        Y.columns.name = 'date'
        if self.hparams.num_splits == 1:
          X_train = X
          Y_train = Y
        else:
          ## We use kfold based on stock (as test set is on the same time step but for other stocks)
          kf = KFold(n_splits=self.hparams.num_splits, shuffle=True, random_state=self.hparams.split_seed)
          all_splits = [k for k in kf.split(X)] 
          train_indexes, val_indexes = all_splits[self.hparams.k]
          train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()
          X_train, Y_train, X_val, Y_val = X.iloc[train_indexes], Y.iloc[train_indexes],  X.iloc[val_indexes], Y.iloc[val_indexes]

        X_train_reshape = pd.concat([ X_train.T.shift(i+1).stack(dropna=False) for i in range(250) ], 1).dropna() 
        X_train_reshape.columns = pd.Index(range(1,251), name='timeLag') ##size X_train_reshape (25200, 250)
        targets_train = Y_train.T.stack()
        self.data_train = MyDataset(X_train_reshape, targets_train)
        self.data_val = None
        if self.hparams.num_splits != 1:
          X_val_reshape = pd.concat([ X_val.T.shift(i+1).stack(dropna=False) for i in range(250) ], 1).dropna() 
          X_val_reshape.columns = pd.Index(range(1,251), name='timeLag') ##size X_train_reshape (25200, 250)
          targets_val = Y_val.T.stack()

          self.data_val = MyDataset(X_val_reshape, targets_val)

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory, shuffle=True)

    def val_dataloader(self):
        if self.data_val:
            return DataLoader(dataset=self.data_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                              pin_memory=self.hparams.pin_memory)