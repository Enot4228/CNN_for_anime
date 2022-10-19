import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision
from torch.utils.data import DataLoader
from AnimeImageDataset import AnimeImageDataset

in_channel = 3
num_classes = 20
learning_rate = 1e-3
batch_size = 64
num_epochs = 30

dataset = AnimeImageDataset(csv_file='./dataset_paths.csv', root_dir='./',
                            transform=T.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)



