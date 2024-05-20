from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
import torch

class MinMaxNormalizeImage:
    def __call__(self, img: torch.Tensor):
        min_val = img.min()
        max_val = img.max()
        normalized_tensor = (img - min_val) / (max_val - min_val)
        return normalized_tensor


class DiffusionLensDataset(Dataset):
    def __init__(self, data_folder: str):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(144)
            ])

        # k = 100

        data = []

        for (i, file) in enumerate(os.listdir(data_folder)):
            # if i == k:
            #     break
            im = np.load(os.path.join(data_folder, file), allow_pickle=True)
            data.append(im)

        
        self.dataset = np.array(data, dtype=np.float32).transpose(0,2,3,1)
        

    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, idx):
        image = self.dataset[idx]

        image = self.transform(image)

        return image