import numpy as np
import pandas as pd
from patsy import dmatrix

import torch
import torch.nn as nn



def load_latency_data(path='LUT.csv'):
    lut_df = pd.read_csv(path)

    device = pd.DataFrame(lut_df["Device"])
    operation = pd.DataFrame(lut_df["Operation"])
    
    device = dmatrix("Device + 0", lut_df)
    operation = dmatrix("Operation + 0", lut_df)
    
    device = pd.DataFrame(device, columns=["a31", "s9", "s10"])
    operation = pd.DataFrame(operation, columns=["identity", "conv", "diconv", "logits"])
    
    lut_df_dum = pd.concat([device, operation, lut_df.iloc[:,2:]], axis=1)
    return lut_df_dum


def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'best_model.pth')
    
class LatencyPredictor(nn.Module):
    def __init__(self, in_features, h_units):
        super(LatencyPredictor, self).__init__()
        self.in_featuers = in_features
        self.h_units = h_units

        self.model = nn.Sequential(
            nn.Linear(in_features, h_units),
            nn.ReLU(inplace=True),
            nn.Linear(h_units, h_units//2),
            nn.ReLU(inplace=True),
            nn.Linear(h_units//2, 1),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class AverageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt