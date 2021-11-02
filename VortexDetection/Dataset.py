import cv2 as cv
import numpy as np
import torch
import albumentations as A
from torch.utils.data import Dataset
from PIL import Image
from albumentations.pytorch import ToTensorV2
from VortexDetection import Configration


class dataset(Dataset):
    def __init__(self, img_path, anchors,
            R=[13, 26, 52], image_size=416, C=2,):      # Where R for Three Resulations
                                        # And C Number of classes (vortex, non vortex)
        self.img_path = img_path
        self.image_size = image_size
        self.R = R
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for stride 32, 16, and 8
        self.num_anchors = self.anchors.shape[0]
        self.C = C

    def __len__(self):
        return len(' ')

    def __getitem__(self, index):
        img_path = self.img_path
        image = np.array(Image.open(img_path).convert("RGB"))
        test_transforms = A.Compose(
            [   A.LongestMaxSize(max_size=Configration.Image_Size),
                A.PadIfNeeded(min_height=Configration.Image_Size, min_width=Configration.Image_Size,
                    border_mode=cv.BORDER_CONSTANT
                ),
                A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
                ToTensorV2(),
            ],)
        augmentations =test_transforms(image=image)
        image = augmentations["image"]
        targets = [torch.zeros((self.num_anchors // 3, R, R, 6)) for R in self.R]
        return image, tuple(targets)