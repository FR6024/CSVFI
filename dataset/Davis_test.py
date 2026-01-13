import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

def get_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

class Davis(Dataset):
    def __init__(self, data_root , ext="png"):

        super().__init__()

        self.data_root = data_root
        self.images_sets = []

        for label_id in os.listdir(self.data_root):

            ctg_imgs_ = sorted(os.listdir(os.path.join(self.data_root , label_id)))
            ctg_imgs_ = [os.path.join(self.data_root , label_id , img_id) for img_id in ctg_imgs_]
            # print(ctg_imgs_)
            for start_idx in range(0,len(ctg_imgs_)-6,2):
                add_files = ctg_imgs_[start_idx : start_idx+7 : 2]
                add_files = add_files[:2] + [ctg_imgs_[start_idx+3]] + add_files[2:]
                self.images_sets.append(add_files)


        self.transforms = transforms.Compose([
                transforms.CenterCrop((480,848)),
                transforms.ToTensor()
            ])

        # print(len(self.images_sets))

    def __getitem__(self, idx):

        imgpath_r = []

        imgpaths = self.images_sets[idx]
        images = [Image.open(img) for img in imgpaths]
        images4 = [transforms.CenterCrop((480,848))(img) for img in images]
        images_cv2 = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images4]
        edges_images = [get_edges(img) for img in images_cv2]

        edges_images = [transforms.ToTensor()(img) for img in edges_images]

        images = [self.transforms(img) for img in images]

        for i in range(0, 5):
            imgpath = imgpaths[i].split('/')[-1][:-4]
            imgpath = imgpath.split('\\')[-2:]
            imgpath_r.append(imgpath)

        return images[:2] + images[3:], edges_images[:2] + edges_images[3:], images[2], imgpath_r[2]

    def __len__(self):

        return len(self.images_sets)

def get_loader(data_root, batch_size, shuffle, num_workers, test_mode=True):

    dataset = Davis(data_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

if __name__ == "__main__":

    dataset = Davis("E:/DataSet/davis-90/480p/")

    print(len(dataset))

    dataloader = DataLoader(dataset , batch_size=1, shuffle=True, num_workers=0)