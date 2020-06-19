import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset



class SampleProperty(object):
    def __init__(self, row):
        self._sample = row.strip().split(' ')
    @property
    def path(self):
        return self._sample[0]

    @property
    def label(self):
        return int(self._sample[1])

class FaceImageDataset(Dataset):
    def __init__(self, data_root, list_file_root, modality, transform=None):
        self.data_root = data_root
        self.list_file_root = list_file_root
        self.modality = modality
        self.transform = transform
        self.Sample_List = [SampleProperty(x) for x in open(self.list_file_root)]

    def __getitem__(self, idx):
        img_path = self.Sample_List[idx].path
        label = self.Sample_List[idx].label

        if self.modality == 'RGB':
            image = Image.open(os.path.join(self.data_root, img_path)).convert('RGB')
        if self.modality == 'Gray':
            image = Image.open(os.path.join(self.data_root, img_path)).convert('L')
        if self.transform is not None:
            image = self.transform(image)###C H W  
        label=torch.tensor(label)

        return image,label

    def __len__(self):
        return len(self.Sample_List)




class FaceImagePiarDataset(Dataset):
    def __init__(self, data_root, list_file_root, modality, transform=None):
        self.data_root = data_root
        self.list_file_root = list_file_root
        self.modality = modality
        self.transform = transform
        self.Sample_List = [line.strip().split(' ') for line in open(self.list_file_root)]

    def __getitem__(self, idx):
        img_path1 = self.Sample_List[idx][0]
        img_path2 = self.Sample_List[idx][1]
        label = int(self.Sample_List[idx][2])

        if self.modality == 'RGB':
            image1 = Image.open(os.path.join(self.data_root, img_path1)).convert('RGB')
            image2 = Image.open(os.path.join(self.data_root, img_path2)).convert('RGB')
        if self.modality == 'Gray':
            image1 = Image.open(os.path.join(self.data_root, img_path1)).convert('L')
            image2 = Image.open(os.path.join(self.data_root, img_path2)).convert('L')
        if self.transform is not None:
            image1 = self.transform(image1)###C H W  
            image2 = self.transform(image2)###C H W  
        label=torch.tensor(label)

        return image1,image2,label

    def __len__(self):
        return len(self.Sample_List)