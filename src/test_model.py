from model import Net
import torch
from torch_geometric.data import Data
import sys
from torch_geometric.loader import DataLoader
import os.path as osp
from torch_geometric.io import read_txt_array

if __name__ == '__main__':
    
    if len(sys.argv) >= 2:
        model_name = sys.argv[1]
        file = sys.argv[2]
        
        model_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'models')  
        filepath = model_path + "/" + model_name + ".pth"
        
        if osp.isfile(filepath) and osp.isfile(file):
        
            data = read_txt_array(file)
            pos = data[:, :3]
            x = data[:, 3:6]
            y = data[:, -1].type(torch.long)
            
            for i in range(len(y)):
                if y[i] > 1:
                    y[i] = 2
            
            data = Data(pos=pos, x=x, y=y, category=0)
            model = torch.load(filepath)
            model.eval()
            predict = model(data)
            
            print("Predict: {}".format(predict))
        
        
        