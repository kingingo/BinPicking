from model import Net
import torch
import sys
import os.path as osp

if len(sys.argv) >= 2:
    model_name = sys.argv[1]
    model_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'models')  
    filepath = model_path + "/" + model_name + ".pth"
    
    if osp.isfile(filepath):
        model = torch.load(filepath)
        