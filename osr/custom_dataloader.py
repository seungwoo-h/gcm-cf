import os
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class Custom_Dataset(Dataset):
    
    def __init__(self, subset, root='./data/'):
  
        """
        subset: 'train' / 'val' / 'test'
        train -> seen in train
        val -> seen in test
        test -> unseen in test
        
        *folder structure*
            root/dog/xxx.png
            root/dog/xxy.png
            root/dog/[...]/xxz.png

            root/cat/123.png
            root/cat/nsdf3.png
            root/cat/[...]/asd932_.png
        """
        seen_labels = {'horse': 0, 'person': 1, 'house': 2, 'dog': 3} # IMPORTANT
        unseen_labels = {'guitar': 0, 'giraffe': 1, 'elephant': 2} # IMPORTANT
        if subset == 'train':
            train_img_dir = os.path.join(root, 'train')
            self.dataset = []
            for label in seen_labels:
                imgs_per_label = glob(os.path.join(train_img_dir, label, '*.jpg'))
                self.dataset.extend([(Image.open(i), seen_labels[label]) for i in imgs_per_label])
        elif subset == 'val':
            val_img_dir = os.path.join(root, 'test')
            self.dataset = []
            for label in seen_labels:
                imgs_per_label = glob(os.path.join(val_img_dir, label, '*.jpg'))
                self.dataset.extend([(Image.open(i), seen_labels[label]) for i in imgs_per_label])
        elif subset == 'test':
            test_img_dir = os.path.join(root, 'test')
            self.dataset = []
            for label in unseen_labels:
                imgs_per_label = glob(os.path.join(test_img_dir, label, '*.jpg'))
                self.dataset.extend([(Image.open(i), unseen_labels[label]) for i in imgs_per_label])
        
        if subset == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
            
        elif subset == 'val' or subset == 'test':
            self.transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return (img, label)
    
    def __len__(self):
        return len(self.dataset)