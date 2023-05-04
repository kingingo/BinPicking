import torch
from torch_geometric.data import InMemoryDataset, Data
from typing import Callable, List, Optional, Union
import os
import os.path as osp
from torch_geometric.io import read_txt_array

class DataNet(InMemoryDataset):
        
    category_ids = {
        'data' : [0,1,2]
    }
        
    seg_classes = {
        'data': [0,1,2] 
        #0 nothing
        #1 stacking box
        #2 box
    }
    
    categories = {}
        
    def __init__(
        self, 
        root, 
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.categories = list(self.category_ids.keys())
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.y_mask = torch.zeros((len(self.seg_classes.keys()), 3),
                                  dtype=torch.bool)
        for i, labels in enumerate(self.seg_classes.values()):
            self.y_mask[i, labels] = 1

    @property
    def num_classes(self) -> int:
        return self.y_mask.size(-1)

    @property
    def raw_file_names(self):
        return self.categories

    @property
    def processed_file_names(self):
        return [
            osp.join(f'{self.categories}_{split}.pt')
            for split in ['train', 'test']
        ]

    def download(self):
        print("No download available...")

    def process(self):
        data_list = []
        #categories_ids = [self.category_ids[cat] for cat in self.categories]
        #print("categories_ids: {}".format(categories_ids))
        #cat_idx = {categories_ids[i]: i for i in range(len(categories_ids))}
        
        for i in range(len(self.raw_file_names)):
            folder = self.raw_file_names[i]
            path = osp.join(self.raw_dir, folder)
            list_files = os.listdir(path)
            
            for file in list_files:
                data = read_txt_array(osp.join(path, file))
                pos = data[:, :3]
                x = data[:, 3:6]
                y = data[:, -1].type(torch.long)
                
                for i in range(len(y)):
                    if y[i] > 1:
                        y[i] = 2
                
                data = Data(pos=pos, x=x, y=y, category=0)
                data_list.append(data)
                        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])