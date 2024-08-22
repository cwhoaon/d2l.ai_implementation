from torch.utils.data import Dataset
import torchvision
import os
import pandas as pd
import torch

class BananaDetection():
    def __init__(self, path, is_train):
        csv_fname = os.path.join(path, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
        
        csv_data = pd.read_csv(csv_fname)
        csv_data = csv_data.set_index('img_name')
        images, targets = [], []
        for img_name, target in csv_data.iterrows():
            images.append(torchvision.io.read_image(
                os.path.join(path, 'bananas_train' if is_train else
                            'bananas_val', 'images', f'{img_name}')))
            targets.append(list(target))

        self.images = images
        self.label = torch.tensor(targets).unsqueeze(1) / 256

    def __getitem__(self, i):
        return self.images[i].float(), self.label[i]
    
    def __len__(self):
        return len(self.images)