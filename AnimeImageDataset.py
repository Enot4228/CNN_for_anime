import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as t
import torchvision
import pandas as pd


class AnimeImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        super(Dataset, self).__init__()
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = self.annotations.iloc[index, 1]
        image = torchvision.io.read_image(img_path, mode=torchvision.io.image.ImageReadMode.RGB).float()
        if self.transform:
            image = self.transform(image)
        y_label = []
        for i in range(0, 10):
            if i == self.annotations.iloc[index, 2]:
                y_label.append(1)
            else:
                y_label.append(0)
        y_label = t.ToTensor()(np.array([y_label]))[0, ...]
        return image, y_label
