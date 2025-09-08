import cv2
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage

class FootballDataset(Dataset):
    def __init__(self, root, train = True, transform = None):
        self.train = train
        if train:
            self.root = os.path.join(root, "football_train")
        else:
            self.root = os.path.join(root, "football_test")
        print(self.root)
        self.matches = os.listdir(self.root)
        self.match_files = [os.path.join(self.root, match_file) for match_file in self.matches]
        print(self.match_files)
        self.from_id = 0
        self.to_id = 0
        self.video_select = {}
        for path in self.match_files:
            json_dir, video_dir = os.listdir(path)
            json_dir, video_dir = os.path.join(path, json_dir), os.path.join(path, video_dir)
            with open(json_dir, "r") as jf:
                json_data = json.load(jf)
            self.to_id +=len(json_data["images"])
            self.video_select[path] = [self.from_id+1, self.to_id]
            self.from_id= self.to_id
        self.transform = transform
    def __len__(self):
        return self.to_id
    def __getitem__(self, idx):
        for key, value in self.video_select.items():
            if value[0]<=idx +1 <= value[1]:
                idx = idx - value[0]
                select_path = key
        json_dir, video_dir = os.listdir(select_path)
        json_dir, video_dir = os.path.join(select_path, json_dir), os.path.join(select_path, video_dir)
        json_file = open(json_dir, "r")
        annotations = json.load(json_file)["annotations"]

        cap = cv2.VideoCapture(video_dir)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        flag, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotations = [anno for anno in annotations if anno["image_id"]== idx +1 and anno["category_id"]==4]
        box = [annotation["bbox"] for annotation in annotations]
        cropped_images = [frame[int(y):int(y+h), int(x):int(x+w)] for [x, y , w , h] in box]

        jerseys = [int(annotation["attributes"]["jersey_number"]) for annotation in annotations]
        if self.transform:
            cropped_images = [self.transform(image) for image in cropped_images]
            cropped_images = torch.stack(cropped_images)
        return  cropped_images, jerseys

if __name__ == '__main__':
    transform = Compose([
        ToPILImage(),
        Resize((224, 112)),
        ToTensor(),
    ])
    dataset = FootballDataset("D:\\python\\pythonProject\\football\\data", train= False, transform=transform)
    cropped_images, jerseys = dataset.__getitem__(200)
    print(cropped_images.shape)