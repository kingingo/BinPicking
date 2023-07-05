import torch
from torch_geometric.data import InMemoryDataset, Data
from typing import Callable, List, Optional, Union
import os
import os.path as osp
from torch_geometric.io import read_txt_array
#import open3d as o3d
import numpy as np
class DataNet(InMemoryDataset):
    test = False
    modelname = ''
        
    category_ids = {
        'unbekannt': [0],
        'data' : [0,1,2,3,4,5,6],
        'stackingbox' : [1],
        'banana' : [2],
        'apple' : [3],
        'orange': [4],
        'pear' : [5],
        'plum' : [6]
    }
        
    seg_classes = {
        'unbekannt': [0],
        'data' : [0,1,2,3,4,5,6],
        'stackingbox' : [1],
        'banana' : [2],
        'apple' : [3],
        'orange': [4],
        'pear': [5],
        'plum': [6],
        #0 nothing / unknown
        #1 stacking box
        #2 banana
        #3 apple
        #4 orange
        #5 pear
        #6 plum
    }
    
    categories = {}
        
    def __init__(
        self, 
        root, 
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        test: bool = False,
        modelname = ''
    ):
        self.test = test
        self.modelname = modelname
        if test:
            self.categories = [list(self.category_ids.keys())[0]]
        else:
            self.categories = list(self.category_ids.keys())
            
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.y_mask = torch.zeros((len(self.seg_classes.keys()), len(self.seg_classes.keys())),
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
        plist = []
        
        for split in ['train', 'test']:
            for cat in self.categories:
                plist.append(f'{cat}_{split}_{self.modelname}.pt')
        
        return plist

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
                pos = data[:, :3] #XYZ
                x = data[:, 3:6] #rgb
                y = data[:, -1].type(torch.long) #LABEL
                data = Data(pos=pos, x=x, y=y, category=0)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                
                data_list.append(data)
                     
        data, slices = self.collate(data_list)
        print("processed_paths:{}".format(self.processed_paths[0]))
        torch.save((data, slices), self.processed_paths[0])