import cv2
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage
from data_football import FootballDataset
from torchvision.models import resnet50, ResNet50_Weights
from my_resnet_model import  MyResNet
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from argparse import ArgumentParser
from tqdm import tqdm

def get_args():
    parser = ArgumentParser(description="CNN training")
    parser.add_argument("--root", "-r", type=str, default="D:\\python\\pythonProject\\football\\data\\football_train", help="Root of the dataset")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs")
    parser.add_argument("--image-size", "-i", type=int, default=224, help="Image size")
    parser.add_argument("--batch-size", "-b", type=int, default=2, help="Batch size")
    parser.add_argument("--logging", "-l", type=str, default="D:\\python\\pythonProject\\football\\tensorboard")
    parser.add_argument("--trained_models", "-t", type=str, default="D:\\python\\pythonProject\\football\\model")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    args = parser.parse_args()
    return args


def my_collate_fc(batch):
    images, labels = list(zip(*batch))
    images = torch.cat(images)
    final_labels = []
    for label in labels:
        final_labels.extend(label)
    return images, final_labels
if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    num_epochs = args.epochs
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)
    writer = SummaryWriter(args.logging)

    transform = Compose([
        ToPILImage(),
        Resize((args.image_size, args.image_size)),
        ToTensor(),
    ])

    dataset = FootballDataset(args.root, transform)
    training_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=False,
        shuffle=True,
        collate_fn=my_collate_fc
    )
    model = MyResNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model"])
        optimizer = model.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch =0
        best_acc = 0
    for epoch in range(start_epoch, num_epochs):
        model.train()
        progress_bar = tqdm(training_loader, colour="green")
        for iter, (image, label) in enumerate(progress_bar):
            image = image.to(device)
            label = label.to(device)

            outputs = model(image)
            loss_value = criterion(outputs)
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}: Loss {:.3f}".format(epoch+1, args.epochs, iter+1, len(training_loader), loss_value))
            writer.add_scalar("Train/Loss", loss_value, epoch * len(training_loader) + iter)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()


