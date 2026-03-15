import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os

class SegDataset(Dataset):
    def __init__(self, data_list_file: str, template_path: str, target_size=(128, 128, 128)):
        # Read subject paths from txt file
        with open(data_list_file, 'r') as file:
            self.subject_paths = file.read().splitlines()

        
        self.moving_template = np.load(template_path)  
        self.moving_template = torch.tensor(self.moving_template, dtype=torch.float32)
        # self.moving_template = self.moving_template.permute(3, 0, 1, 2)  # (5, D, H, W)


        self.moving_template = F.interpolate(
            self.moving_template.unsqueeze(0), 
            size=target_size,
            mode='nearest'
        ).squeeze(0)  # Back to (5, 128, 128, 128)

        self.target_size = target_size

    def __len__(self):
        return len(self.subject_paths)

    def __getitem__(self, idx):

        fixed_map = np.load(self.subject_paths[idx])  # (D, H, W, 5)
        fixed_map = torch.tensor(fixed_map, dtype=torch.float32)
        # fixed_map = fixed_map.permute(3, 0, 1, 2)  # (5, D, H, W)

        # Resize fixed map
        fixed_map = F.interpolate(
            fixed_map.unsqueeze(0),  # (1, 5, D, H, W)
            size=self.target_size,
            mode='nearest'
        ).squeeze(0)  # (5, 128, 128, 128)
    
        assert torch.all(torch.sum(fixed_map, dim=0) == 1),  "Invalid onehot"
        assert torch.all(torch.sum(self.moving_template, dim=0) == 1), "Template corrupted"
        return self.moving_template, fixed_map
