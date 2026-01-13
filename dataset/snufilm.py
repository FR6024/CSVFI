from os.path import join

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
import cv2
import numpy as np
import torch


class NonePlaceholder:
    pass

def get_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

class Snufilm_quintuplet(Dataset):
    def __init__(self, db_dir, mode='extreme'):
        self.db_dir = db_dir
        self.transform = transforms.Compose([transforms.ToTensor()])
        # self.transform = transforms.Compose([transforms.CenterCrop((480,848)),transforms.ToTensor()])
        self.mode = mode
        self.input1_list = []
        self.input3_list = []
        self.input5_list = []
        self.input7_list = []
        self.gt_list = []
        with open(join(self.db_dir, 'test-{}.txt'.format(mode)), 'r') as f:
            self.triplet_list = f.read().splitlines()

    def __getitem__(self, index):
        images = []
        edges_image = []
        lst = self.triplet_list[index].split(' ')
        lst = self.get_quintuplet(lst)
        gt_path = lst[2].split('/')[-2:]
        # boundary cases
        sk = False
        p = False
        W = H = 0
        try:
            for l in lst:
                pad_H = pad_W = 0
                image = Image.open(join(self.db_dir, l))
                W, H = image.size
                # print(image.size)
                if H/2 % 2==1:
                    pad_H = 2
                    p = True
                    # print('HHHHHHH-----2',l)
                if W/2 % 2==1:
                    pad_W = 2
                    p = True
                    # print('WWWWWWW-----2',l)
                if H/4 % 2==1:
                    pad_H += 4
                    p = True
                    # print('HHHHHHH',l)
                if W/4 % 2==1:
                    pad_W += 4
                    p = True
                    # print('WWWWWWW',l)
                image = ImageOps.expand(image, border=(pad_W//2, pad_H//2, pad_W//2, pad_H//2), fill=0)

                # print('=================',image.size)
                images.append(self.transform(image))
                edges_image.append(image)
            images_cv2 = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in edges_image]
            edges_images = [get_edges(img) for img in images_cv2]
            edges_images = [transforms.ToTensor()(img) for img in edges_images]

            return images[:2]+images[3:], edges_images[:2]+edges_images[3:], images[2], sk, p, (W, H), gt_path
        except:
            place = torch.ones((1, 3, 3, 3))
            sk = True
            return [place], [place], place, sk, p, (1, 1), gt_path

    def get_quintuplet(self, lst):
        """
        lst -- list of paths of a triplet
        """
        if self.mode == 'extreme':
            offset = 16
        elif self.mode == 'hard':
            offset = 8
        elif self.mode == 'medium':
            offset = 4
        else:
            offset = 2
        im3_idx_str = lst[0].split('/')[-1].split('.')[0]
        im1_idx_str = str(int(im3_idx_str) - offset).zfill(len(im3_idx_str))
        im7_idx_str = str(int(im3_idx_str) + offset * 2).zfill(len(im3_idx_str))
        im1_pth = '/'.join([item if not item.endswith('.png') else im1_idx_str + '.png' for item in lst[0].split('/')])
        im7_pth = '/'.join([item if not item.endswith('.png') else im7_idx_str + '.png' for item in lst[0].split('/')])
        return [im1_pth, *lst, im7_pth]

    def __len__(self):

        return len(self.triplet_list)

def custom_collate(batch):
    if not batch:
        return None

    batch = [item for item in batch if not isinstance(item, NonePlaceholder)]
    # check
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def get_loader(data_root, batch_size, shuffle, num_workers, mode='easy'):

    dataset = Snufilm_quintuplet(data_root, mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

if __name__ == "__main__":
    dataset = Snufilm_quintuplet('E:/DataSet/SNUFilm/', mode='easy')
    print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    print(len(dataloader))