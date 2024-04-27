from torch.utils.data import Dataset
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

# Dataset Class
NUM_ATTR = 6
class FashionNet_Dataset(Dataset):

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = [[] for _ in range(NUM_ATTR)]
        self.transform = transform

        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))

        with open(txt.replace('.txt', '_attr.txt')) as f:
            for line in f:
                attrs = line.split()
                for i in range(NUM_ATTR):
                    self.labels[i].append(int(attrs[i]))

    def __len__(self):
        return len(self.labels[0])

    def __getitem__(self, index):

        path = self.img_path[index]
        label = np.array([self.labels[i][index] for i in range(NUM_ATTR)])
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
    

# class FashionNet_Dataset(Dataset):

#     def __init__(self, root, txt, transform=None):
#         self.img_path = []
#         self.labels = [[] for _ in range(NUM_ATTR)]
#         self.transform = transform

#         with open(txt) as f:
#             for line in f:
#                 self.img_path.append(os.path.join(root, line.split()[0]))
#                 # make dummy label for test set
#                 if 'test' in txt:
#                     for i in range(NUM_ATTR):
#                         self.labels[i].append(0)
#         if 'test' not in txt:
#             with open(txt.replace('.txt', '_attr.txt')) as f:
#                 for line in f:
#                     attrs = line.split()
#                     for i in range(NUM_ATTR):
#                         self.labels[i].append(int(attrs[i]))

#     def __len__(self):
#         return len(self.labels[0])

#     def __getitem__(self, index):

#         path = self.img_path[index]
#         label = np.array([self.labels[i][index] for i in range(NUM_ATTR)])
#         image = Image.open(path).convert('RGB')

#         if self.transform:
#             image = self.transform(image)

#         return image, label