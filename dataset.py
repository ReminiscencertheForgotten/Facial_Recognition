import os
from load import*
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms

class My_Dataset(Dataset):
    
    def __init__(self, root_path, src_filename, transformer=None):
        super(My_Dataset, self).__init__()
        
        self.root_path = root_path
        self.transformer = transformer
        self.labels = pd.read_csv(src_filename)
        
    def __len__(self):   
        return len(self.labels)
    
    def __getitem__(self, index):
        picName = self.labels.iloc[index, 0]
        ownerName = picName[:-9]
        
        img_path = os.path.join(self.root_path, ownerName, picName)
        image = Image.open(img_path).crop((60, 55, 190, 190))
        
        label = self.labels.iloc[index, 1]
        if self.transformer:
            image = self.transformer(image)
            
        return image, label
    
    
def getDataloader(bacth_size, params):
    
    print('Preparing dataloader...')
    
    train_filename = params['train_filename']
    valid_filename = params['valid_filename']
    test_filename = params['test_filename']
    root_path = params['root_path']

    transformer_normal = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(125),
        transforms.RandomCrop(125, padding=15),
        transforms.Normalize(
            mean=[0.471, 0.448, 0.408], 
            std=[0.234, 0.239, 0.242]
        ),        
    ])
    
    train_set = My_Dataset(
        root_path=root_path, 
        src_filename=train_filename, 
        transformer=transformer_normal
    )
    
    valid_set = My_Dataset(
        root_path=root_path,
        src_filename=valid_filename,
        transformer=transformer_normal
    )
    
    test_set = My_Dataset(
        root_path=root_path,
        src_filename=test_filename,
        transformer=transformer_normal
    )
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=bacth_size,
        num_workers=1,
    )
    
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=bacth_size,
        num_workers=1,
    )
    
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=bacth_size,
        num_workers=1,
    )
    
    print('Dataloader ready.\n')
    return train_loader, valid_loader, test_loader