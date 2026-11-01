"""
Dataset class for BraTS PNG slices.
"""
import os
from PIL import Image
from torch.utils.data import Dataset

class BraTSPNGDataset(Dataset):
    def __init__(self, root_dir, df, transform):
        self.root_dir = root_dir
        self.df = df
        self.transform = transform
        self.files = [os.path.join(root_dir, f) for f in df['source_file']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        im = Image.open(self.files[idx]).convert('RGB')
        return self.transform(im), idx

