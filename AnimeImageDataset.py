from torch.utils.data import Dataset
import torchvision.transforms as T
import os
from skimage import io
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
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image)
            image = T.Resize(size=200)(image)
        y_label = self.annotations.iloc[index, 1]
        return image, y_label
