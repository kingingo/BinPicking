from pointnet2.model import Net
import torch
from torch_geometric.data import Data
import sys
from torch_geometric.loader import DataLoader
import os.path as osp
from torch_geometric.io import read_txt_array
from datanet import DataNet
import torch_geometric.transforms as T

def calc_perc(part, vol):
    if part == vol:
        return 100
    
    if part == 0:
        return 0
    
    return (part/vol) * 100

if __name__ == '__main__':
    
    if len(sys.argv) >= 3:
        model_name = sys.argv[1]
        file = sys.argv[2]
        gpu = sys.argv[3]
        
        model_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'models', model_name)  
        modelname = osp.basename(model_path).split("_")[0]
        if not osp.isfile(model_path):
            print("Couldn't find {}".format(model_path))
            exit()
        else:
            print("Model:{}".format(model_path))
            
        filepath = osp.join(osp.dirname(osp.realpath(__file__)), file)
        if not osp.isfile(model_path):
            print("Couldn't find {}".format(filepath))
            exit()
        else:
            print("File:{}".format(filepath))
            
        transform = T.Compose([
            T.RandomJitter(0.01),
            T.RandomRotate(15, axis=0),
            T.RandomRotate(15, axis=1),
            T.RandomRotate(15, axis=2)
        ])
        pre_transform = T.NormalizeScale()
        
        print("Modelname: {}".format(modelname))
        device = torch.device('cpu')
        data = read_txt_array(filepath)
        pos = data[:, :3]
        x = data[:, 3:6]
        y = data[:, -1].type(torch.long) 
                
        data = Data(pos=pos, x=x, category=0)
        data = pre_transform(data)
        data = transform(data)
        #file_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'predict') 
        #test_dataset = DataNet(root = file_path, transform=transform,pre_transform=pre_transform)
        dataloader = DataLoader([data], batch_size=1, shuffle=False,num_workers=1)
        
        model = torch.load(model_path,map_location=device)
        model.eval()
        for idx, data in enumerate(dataloader):
            if modelname == 'pointnet':
                t = []
                for i in range(len(data.pos)):
                    t.append([[data.pos[i][0],data.pos[i][1],data.pos[i][2],data.x[i][0],data.x[i][1],data.x[i][2]]])
                
                points = torch.tensor(t)
                points = points.transpose(2, 1)
                points = points.to(device)
                outs, _, _ = model(points)
                outs = outs.view(-1, 7)
            else:
                data = data.to(device)
                outs = model(data.x, data.pos, data.batch)
        
        pred = outs.max(1)[1]
        f = open('predict_{}.txt'.format(modelname), 'wb')
        npos = pos.numpy()
        nrgb = x.numpy()
        nlabel = y.numpy()
        npred = pred.numpy()
        
        vol = len(npos)
        #[0,1,2] 
        #0 - nothing
        #1 - stackingbox
        #2 - banana
        #3 - apple
        #4 - orange
        category_name = ["nothing", "stackingbox", "banana", "apple", "orange", "pear", "plum", "hammer"]
        category_correctness = []
        for i in range(len(category_name)):
            category_correctness.append({'wrong':0, 'correct': 0, 'volume': 0, 'name': category_name[i]})
            
        for i in range(vol):
            x = npos[i][0]
            y = npos[i][1]
            z = npos[i][2]
            
            r = nrgb[i][0]
            g = nrgb[i][1]
            b = nrgb[i][2]
            
            pred_label = pred[i]
            label = int(nlabel[i])
            
            index = 'correct' if int(pred_label) == label else 'wrong'
            category_correctness[label][index] += 1
            category_correctness[label]['volume'] += 1
            
            line = "{} {} {} {} {} {} {} {}\n".format(x,y,z,int(r),int(g),int(b),label,pred_label)
            f.write(line.encode())
        f.close()
        
        for i in range(len(category_name)):
            print("{}: {}/{}, {}%".format(category_name[i],category_correctness[i]['correct'], category_correctness[i]['volume'],calc_perc(category_correctness[i]['correct'], category_correctness[i]['volume'])))
        
        
        
        