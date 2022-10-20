import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as t
from torch.utils.data import DataLoader
from AnimeImageDataset import AnimeImageDataset
from AnimeCNNModel import AnimeCNNModel

torch.device('cpu')

in_channel = 3
num_classes = 20
learning_rate = 1e-4
batch_size = 64
num_epochs = 30

dataset = AnimeImageDataset(csv_file='./path_dataset.csv', root_dir='./',
                            transform=t.Resize((700, 700)))
#print(dataset[2])
train_set, test_set = torch.utils.data.random_split(dataset, [7, 3], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)

model = AnimeCNNModel(in_channel=in_channel)

# image_dir = './data/Berserk/berserk_1.jpg'
# test_tensor = torchvision.io.read_image(image_dir, mode=torchvision.io.image.ImageReadMode.RGB).float()
# test_tensor = torchvision.transforms.Resize((700, 700))(test_tensor)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

for epoch in range(1, num_epochs+1):
    print(f'Epoch: {epoch}')
    losses = []
    for x_train, y_train in train_loader:
        # x_train = x_train.to(device=device)
        # y_train = y_train.to(device=device)
        y_pred = model(x_train)
        # print(y_pred.shape, y_train.shape)
        loss = loss_func(y_pred.float(), y_train.float()[0, ...])
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # with torch.no_grad():
    #     for x_val, y_val in val_loader:
    #         y_pred = model(x_val)
    #         loss = loss_func(y_pred.float(), y_val)
