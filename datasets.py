import os, json
import cv2
import torch.utils.data
from PIL import Image
import numpy as np
import re
from torchvision import transforms, utils


class CHAIRS2020(torch.utils.data.Dataset):
    def __init__(self, folder_path, split = 'train', transform = None):
        super(CHAIRS2020, self).__init__()
        print('Loading data...')

        self.transform = transform
        self.folder_path = folder_path
        self.folder_path += split + '/'
        images = os.listdir(self.folder_path  + 'image/' )
        images.sort(key=lambda f: int(re.sub('\D', '', f)))
        #print(images)
        self.img_files = [self.folder_path + 'image/' + x  for x in images]
        self.mask_files = [self.folder_path + 'mask/' + x  for x in images]
        #for consistency across runs.
        paired_id_names = [(int(x.split('.')[0]), self.folder_path + 'image/' + x) for x in images]
        self.paired_id_names = paired_id_names
        paired_id_mask = [(int(x.split('.')[0]), self.folder_path + 'mask/' + x) for x in images]
        self.paired_id_mask = paired_id_mask

        self.image_names = [img_name for (img_id, img_name) in paired_id_names]
        self.image_ids = [img_id for (img_id, img_name) in paired_id_names]
        self.mask_names = [mask_name for (img_id, mask_name) in paired_id_mask]
        print('...loaded.')
        
      
    def __getitem__(self, index):

        img_  = Image.open(open(self.image_names[index], 'rb'))
        #img_ = img_.convert("RGB")
        mask_ = Image.open(open(self.mask_names[index], 'rb'))

        if self.transform:
            img_ = self.transform(img_)
            mask_ = self.transform(mask_)
        
        #mask_ =  transforms.ToTensor()(mask_)
        #change the mask between some numbers
        #temp = torch.zeros(1, 128, 128, dtype=torch.float)
        temp = torch.zeros(1, mask_.size()[1], mask_.size()[2], dtype=torch.float)
        #temp[:,:,:] = 0
        for i in range(0, len(mask_)):
            temp[i][ mask_[0]> 0.4  ] = 1
            temp[i][ mask_[0]<= 0.4 ] = 0
        label = temp
        
        return img_, label


    def __len__(self):
        return len(self.img_files)

