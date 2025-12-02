import pandas as pd
import torch
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, csv_path, input_len=10, pred_len=5):
        df = pd.read_csv(csv_path)
        
        self.coords = torch.tensor(df[["X", "Y"]].values, dtype=torch.float32)
        self.input_len = input_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.coords) - self.input_len + self.pred_len
    
    def __getitem__(self, index):
        x = self.coords[index : index + self.input_len]
        y = self.coords[index + self.input_len : index + self.input_len + self.pred_len]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
  

ts = TrajectoryDataset("../../data/normalized/Professional/match1/csv/1_01_00_ball.csv")
x, y = ts[0]

print(x.shape)
print(y.shape)