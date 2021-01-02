import os
import cv2
from utils import get_file_type
import numpy as np
import pandas as pd
import albumentations
import torch
# from torch.utils.data import Dataset

from typing import Optional
from tqdm import tqdm



class Melanoma(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        config: type,
        transforms: Optional[albumentations.core.composition.Compose] = None,
        test: bool = False,
        albu_norm: bool = False
    ):
        # 1. Is it good practice to name self.df = dataframe, or self.df = df
        self.df = dataframe
        self.config = config
        self.transforms = transforms
        self.test = test
        self.albu_norm = albu_norm
        
        '''
        This is necessary as when there is no augmentations passed in, there will not be a case whereby albu_norm is True since albu_norm
        only co-exists with transforms=True
        '''
        
        if self.transforms is None:
            assert self.albu_norm is False
            print('Transforms is None and Albumentation Normalization is not initialized!')
            
        self.image_extension = get_file_type(image_folder_path=config.train_path, allowed_extensions=None)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):


        label = self.df.target.values[idx]
        label = torch.as_tensor(data=label, dtype=torch.int64, device=None)
        image_id = self.df.image_name.values[idx]

        if self.test:
            image_path = os.path.join(self.config.test_path, "{}{}".format(image_id, self.image_extension))
        else:
            image_path = os.path.join(self.config.train_path, "{}{}".format(image_id, self.image_extension))
        

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        

        if self.albu_norm is False:
            image = image.astype(np.float32) / 255.0

        if self.transforms is not None:
            albu_dict = {"image": image}
            transform = self.transforms(**albu_dict)
            image = transform["image"]
        else:
            image = torch.as_tensor(data=image, dtype=torch.float32, device=None)

        return image_id, image, label