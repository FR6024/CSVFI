import os

import cv2
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import numpy as np
import torch


def get_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    return edges

class VimeoSepTuplet(Dataset):
    def __init__(self, data_root, is_training , input_frames="1357", mode='mini'):
        """
        Creates a Vimeo Septuplet object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
            input_frames: Which frames to input for frame interpolation network.
        """
        self.data_root = data_root
        self.image_root = os.path.join(self.data_root, 'sequences')
        self.training = is_training
        self.inputs = input_frames

        train_fn = os.path.join(self.data_root, 'sep_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'sep_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        if mode != 'full':
            tmp = []
            for i, value in enumerate(self.testlist):
                if i % 38 == 0:
                    tmp.append(value)
            self.testlist = tmp

        if self.training:

            self.transforms = transforms.Compose([
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                # transforms.ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        if self.training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index])

        imgpaths = [imgpath + f'/im{i}.png' for i in range(1,8)]

        pth_ = imgpaths

        # Load images
        images = [Image.open(pth) for pth in imgpaths]

        ## Select only relevant inputs
        inputs = [int(e)-1 for e in list(self.inputs)]
        inputs = inputs[:len(inputs)//2] + [3] + inputs[len(inputs)//2:]
        images = [images[i] for i in inputs]
        imgpaths = [imgpaths[i] for i in inputs]

        if self.training:
            seed = random.randint(0, 2**32)
            images_ = []
            for img_ in images:
                torch.manual_seed(seed)
                images_.append(self.transforms(img_))
            images = images_

              # Random Temporal Flip
            if random.random() >= 0.5:
                images = images[::-1]
                imgpaths = imgpaths[::-1]

            # edge
            images_cv2 = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]
            edges_images = [get_edges(img) for img in images_cv2]

            edges_images = [transforms.ToTensor()(img) for img in edges_images]
            images = [transforms.ToTensor()(img) for img in images]

            gt = images[len(images) // 2]
            images = images[:len(images) // 2] + images[len(images) // 2 + 1:]

            gt_edge = edges_images[len(edges_images) // 2]
            images_edge = edges_images[:len(edges_images) // 2] + edges_images[len(edges_images) // 2 + 1:]

            return images, gt, images_edge, gt_edge
        else:
            T = self.transforms

            images_cv2 = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]
            edges_images = [get_edges(img) for img in images_cv2]

            edges_images = [transforms.ToTensor()(img) for img in edges_images]
            images = [T(img_) for img_ in images]

            gt = images[len(images)//2]
            images = images[:len(images)//2] + images[len(images)//2+1:]

            gt_edge = edges_images[len(edges_images) // 2]
            images_edge = edges_images[:len(edges_images) // 2] + edges_images[len(edges_images) // 2 + 1:]

            imgpath = '_'.join(imgpath.split('/')[-2:])

            return images, gt, imgpath, images_edge, gt_edge

    def __len__(self):
        if self.training:
            return len(self.trainlist)
        else:
            return len(self.testlist)


def get_loader(mode, data_root, batch_size, shuffle, num_workers, test_mode=None):

    if mode == 'train':
        is_training = True

    else:
        is_training = False
    dataset = VimeoSepTuplet(data_root, is_training=is_training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)



if __name__ == "__main__":
    dataset = VimeoSepTuplet("E:/DataSet/vimeo_septuplet/vimeo_septuplet/", is_training=False)
    print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    print(len(dataloader))
