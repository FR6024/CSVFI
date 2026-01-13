import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def get_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges


class UCF(Dataset):
    def __init__(self, data_root , ext="png"):

        super().__init__()

        self.data_root = data_root
        self.file_list = sorted(os.listdir(self.data_root))

        self.transforms = transforms.Compose([
                transforms.CenterCrop((224,224)),
                transforms.ToTensor(),
            ])

    def __getitem__(self, idx):

        imgpath = os.path.join(self.data_root , self.file_list[idx])
        imgpaths = [os.path.join(imgpath , "frame0.png") , os.path.join(imgpath , "frame1.png") ,os.path.join(imgpath , "frame2.png") ,os.path.join(imgpath , "frame3.png") ,os.path.join(imgpath , "framet.png")]

        images = [Image.open(img) for img in imgpaths]

        images_cv2 = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]
        edges_images = [get_edges(img) for img in images_cv2]

        edges_images = [transforms.ToTensor()(img) for img in edges_images]

        images = [self.transforms(img) for img in images]

        return images[:-1], edges_images[:-1], images[-1]

    def __len__(self):

        return len(self.file_list)

def get_loader(data_root, batch_size, shuffle, num_workers, test_mode=True):

    dataset = UCF(data_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

if __name__ == "__main__":

    dataset = UCF_triplet("./ucf_test/")

    print(len(dataset))

    dataloader = DataLoader(dataset , batch_size=1, shuffle=True, num_workers=0)
