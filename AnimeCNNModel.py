import torch
import torch.nn as nn


class AnimeCNNModel(nn.Module):

    def __init__(self, in_channel):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channel, 32, (5, 5), padding='same', device='cuda')
        self.conv_2 = nn.Conv2d(32, 64, (5, 5), padding='same', device='cuda')
        self.conv_3 = nn.Conv2d(64, 128, (5, 5), padding='same', device='cuda')
        self.conv_4 = nn.Conv2d(128, 128, (5, 5), padding='same', device='cuda')
        self.max_pool = nn.MaxPool2d((3, 3), (2, 2))
        self.ada_aver_pool = nn.AdaptiveAvgPool2d((1, 128))
        self.dense_1 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        pass
