from __future__ import print_function, division
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from skimage import io

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class CheXpertDataset(Dataset):
    """ CheXpert dataset. """

    def __init__(self, csv_file, root_dir, transform=None):
        """
            Args:
                csv_file (string): Path to the csv file with annotation.
                root_dir (string): Directory with all the images.
                transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.chexpert_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.chexpert_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.chexpert_frame.iloc[idx, 0])
        image = io.imread(img_name)
        pathologies = self.chexpert_frame.iloc[idx, 5:]\
            .fillna(0.0)\
            .replace(-1.0, 1.0)\
            .as_matrix()\
            .astype('float')
        sample = {'image': image, 'pathologies': pathologies}

        if self.transform:
            sample['image'] = self.transform(image)

        return sample