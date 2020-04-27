from torch.optim import Adam
import torch.nn as nn
from FlowerImageClassifierNN import FlowerClassifierCNNModel
from train_test_loader import train_dataset_loader
from image_display import testvar

cnn_model = FlowerClassifierCNNModel()
optimizer = Adam(cnn_model.parameters())
loss_fn = nn.CrossEntropyLoss()


def train_and_build(n_epoches):
    for epoch in range(n_epoches):
        cnn_model.train()
        for i, (images, labels) in enumerate(train_dataset_loader):
            optimizer.zero_grad()
            outputs = cnn_model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()


# train_and_build(200) 30-45 mins
# train_and_build(400) <1h10mins
#testvar = 1
