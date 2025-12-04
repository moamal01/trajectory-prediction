import pandas as pd
import torch
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    """
    Dataset for trajectory prediction from CSV coordinates.

    Each training sample consists of:
        - input_len consecutive frames as input
        - pred_len consecutive frames as target
    """

    def __init__(self, csv_path, input_len=10, pred_len=5):
        df = pd.read_csv(csv_path)
        
        self.coords = torch.tensor(df[["X", "Y"]].values, dtype=torch.float32)
        self.input_len = input_len
        self.pred_len = pred_len
        
        if len(self.coords) < input_len + pred_len:
            raise ValueError("Sequence too short for given input_len + pred_len")

    
    def __len__(self):
        """Returns the number of available training samples."""
        return len(self.coords) - (self.input_len + self.pred_len) + 1
    
    def __getitem__(self, index):
        x = self.coords[index : index + self.input_len]
        y = self.coords[index + self.input_len : index + self.input_len + self.pred_len]
        
        return x.clone(), y.clone()