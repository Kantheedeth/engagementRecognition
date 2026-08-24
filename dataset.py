import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class EngagementDataset(Dataset):
    def __init__(self, split_dir):
        """
        Args:
            split_dir (str): Path to the split directory, e.g., 'feature_matrices/train'
        """
        self.file_paths = glob.glob(os.path.join(split_dir, "*.npy"))
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        
        # Parse label from filename: <video_name>_label<label>.npy
        filename = os.path.basename(path)
        label = int(filename.split('_label')[-1].split('.npy')[0])
        
        # Load the feature matrix: shape (8, 616), float32
        matrix = np.load(path).astype(np.float32)
        
        # Convert to tensors
        x = torch.from_numpy(matrix)
        y = torch.tensor(label, dtype=torch.long)
        
        return x, y

if __name__ == "__main__":
    train_dir = os.path.join("feature_matrices", "train")
    if os.path.exists(train_dir):
        dataset = EngagementDataset(train_dir)
        print(f"Dataset loaded: {len(dataset)} items.")
        if len(dataset) > 0:
            x, y = dataset[0]
            print(f"Sample tensor shape: {x.shape}, label: {y}")
    else:
        print(f"Directory {train_dir} does not exist yet.")
