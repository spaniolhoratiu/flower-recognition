import torch
from pytorch_training_cnn_model import cnn_model
from train_test_loader import test_dataset_loader
from train_test_loader import test_dataset


def test_accuracy():
    cnn_model.eval()
    test_acc_count = 0
    for k, (test_images, test_labels) in enumerate(test_dataset_loader):
        test_outputs = cnn_model(test_images)
        _, prediction = torch.max(test_outputs.data, 1)
        test_acc_count += torch.sum(prediction == test_labels.data).item()
    test_accuracy = test_acc_count / len(test_dataset)
    return test_accuracy

# test_accuracy = test_acc_count / len(test_dataset)
# print(test_accuracy)
