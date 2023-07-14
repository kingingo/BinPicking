from pointnet2.model import Net
import torch
#import tensorflow as tf
from torch_geometric.data import Data
import sys
from torch_geometric.loader import DataLoader
import os.path as osp
from torch_geometric.io import read_txt_array
from datanet import DataNet
import torch_geometric.transforms as T
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics import Dice
from torchmetrics.classification import MulticlassPrecisionRecallCurve
from sklearn.metrics import precision_recall_curve
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn import metrics
from operator import itemgetter
import json
from json import JSONEncoder

df = None
modelname = None

def init_df():
    global df, modelname
    
    if df is None:
        df = open('details_{}.txt'.format(modelname), 'wb')
        
def dwrite(_str):
    global df
    init_df()
    df.write(_str.encode())
    df.write("\n".encode())
    print(_str)
    

def key_to_json(data):
    if data is None or isinstance(data, (bool, int, str)):
        return data
    if isinstance(data, (tuple, frozenset)):
        return str(data)
    raise TypeError

def to_json(data):
    if data is None or isinstance(data, (bool, int, tuple, range, str, list)):
        return data
    if isinstance(data, (set, frozenset)):
        return sorted(data)
    if isinstance(data, dict):
        return {key_to_json(key): to_json(data[key]) for key in data}
    raise TypeError

def extract(lst, index = 0):
    return list( map(itemgetter(index), lst ))

def calc_perc(part, vol):
    if part == vol:
        return 100
    
    if part == 0 or vol == 0:
        return 0
    
    return (part/vol) * 100

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def max_indicies(list, target, min_confidence_value = 0.5):
    dwrite("min confidence value: {}".format(min_confidence_value))
    name_maping = ['Nothing', 'Stackingbox', 'Banana', 'Apple', 'Orange', 'Pear', 'Plum','Hammer']
    avg = [0] * 7
    vol = [0] * 7
    avg_pred = [0] * 7
    vol_pred = [0] * 7
    
    avg_n = 0
    vol_n = 0
    
    nlabels = []
    npreds = []
    preds = list.max(1)[1]
    
    for i in range(preds.shape[0]):
        index = preds[i]
        label = int(target[i])
        
        max_value = list[i].exp().max()
        mean = list[i].mean()
        median = list[i].median()
        
        avg_pred[index] += max_value
        vol_pred[index] +=1
        
        if label == index:
            vol[label] += 1
            avg[label] += max_value
        
        if label == 0:
            vol_n += 1
            avg_n += max_value
            
        
        if max_value < min_confidence_value:
            
            '''
            print("label: {}, pred_label: {}, max_value: {}, mean: {}, median: {}, pred: {}, exp: {}"
                  .format(
                      label, 
                      index, 
                      max_value ,
                      mean, 
                      median,
                      list[i],
                      list[i].exp()
                      )
                  )
                  '''
            index = 0
        else:
            '''
            if index > 1:
                dwrite("label: {}, pred_label: {}, max_value: {}, mean: {}, median: {}, pred: {}, exp: {}"
                  .format(
                      label, 
                      index, 
                      max_value ,
                      mean, 
                      median,
                      list[i],
                      list[i].exp()
                      )
                  )
            '''
        nlabels.append(torch.tensor(index, dtype=torch.int64))
        npreds.append(max_value)
        
    for i in range(len(avg)):
        
        if vol[i] != 0:
            avg[i] /= vol[i]
            
        if vol_pred[i] != 0:
            avg_pred[i] /= vol_pred[i]
        
        dwrite("label: {}, name: {}, avg_value:{}, avg_value_pred: {}".format(i, name_maping[i], avg[i], avg_pred[i]))    

    if vol_n != 0:
        avg_n /= vol_n
    dwrite("Nothing vol_n: {} ,avg_max_value: {}".format(vol_n, avg_n))
    
    return nlabels, npreds

def unique(_list):
    ulist = []
    
    for entry in _list:
        if entry not in ulist:
            ulist.append(entry)
            
    ulist.sort()
    return ulist

def fix_labels(labels):
    nlabels = []
    
    for label in labels:
        if label > 6:
            label = 0
        nlabels.append(label)
    
    return nlabels

def conv_labels(labels):
    
    if not torch.is_tensor(labels):
        classes = unique(labels)
    else:
        classes = torch.unique(labels)
        
    print("CLASSES: {}".format(classes))
    new_labels = []
    
    for i in range(len(labels)):
        label = labels[i]
        new_label = -1
        
        for j in range(len(classes)):
            if classes[j] == label:
                new_label = j
                break
        
        if new_label == -1:
            print("No suitable label found {} {}".format(label, classes))
            break
        
        new_labels.append(new_label)
    
    return new_labels
    
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
            dwrite("Model:{}".format(model_path))
            
        filepath = osp.join(osp.dirname(osp.realpath(__file__)), file)
        if not osp.isfile(model_path):
            print("Couldn't find {}".format(filepath))
            exit()
        else:
            dwrite("File:{}".format(filepath))
            
        transform = T.Compose([
            T.RandomJitter(0.01),
            T.RandomRotate(15, axis=0),
            T.RandomRotate(15, axis=1),
            T.RandomRotate(15, axis=2)
        ])
        pre_transform = T.NormalizeScale()
        
        dwrite("Modelname: {}".format(modelname))
        device = torch.device('cpu')
        data = read_txt_array(filepath)
        pos = data[:, :3]
        x = data[:, 3:6]
        labels = data[:, -1].type(torch.long) 
        labels = fix_labels(labels)
                
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
        #pred = outs.max(1)[1]
        pred_labels, preds = max_indicies(outs, labels)
       
        '''
        npred = pred.numpy()
        prr = np.array(npred)
        
        pt = {}
        for u in range(npred.shape[0]):
            p = npred[u].item()
            if p != 1:
                pt[p] = outs[u].tolist()
                
        f = open("pointnet333.txt", "a")
        f.write(json.dumps(pt))
        f.close()
        '''
        
        f2 = open('predict_{}_2.txt'.format(modelname), 'wb')
        f = open('predict_{}.txt'.format(modelname), 'wb')
        npos = pos.numpy()
        nrgb = x.numpy()
        nlabel = labels
        
        vol = len(npos)
        #[0,1,2] 
        #0 - nothing
        #1 - stackingbox
        #2 - banana
        #3 - apple
        #4 - orange
        #5 pear
        #6 plum
        category_name = ["nothing", "stackingbox", "banana", "apple", "orange", "pear", "plum", "hammer"]
        category_correctness = []
        pred_catogory_counter = []
        for i in range(len(category_name)):
            category_correctness.append({'wrong':0, 'correct': 0, 'label': 0, "pred_label": 0, 'name': category_name[i]})

                    
        for i in range(vol):
            x = npos[i][0]
            y = npos[i][1]
            z = npos[i][2]
            
            r = nrgb[i][0]
            g = nrgb[i][1]
            b = nrgb[i][2]
            
            pred_label = pred_labels[i]
            label = int(nlabel[i])
            
            index = 'correct' if int(pred_label) == label else 'wrong'
            category_correctness[label][index] += 1
            category_correctness[pred_label]['pred_label'] += 1
            category_correctness[label]['label'] += 1
            
            line = "{} {} {} {} {} {} {} {}\n".format(x,y,z,int(r),int(g),int(b),pred_label, preds[i])
            line2 = "{} {} {} {} {} {} {} {}\n".format(x,y,z,int(r),int(g),int(b),label,pred_label)
            f2.write(line2.encode())
            f.write(line.encode())
        f2.close()
        f.close()
        
        cc = 0
        for i in range(len(category_name)):
            vol_label = category_correctness[i]['label']
            vol_pred = category_correctness[i]['pred_label']
            
            c = calc_perc(category_correctness[i]['correct'], vol_label) / 100
            w = calc_perc(category_correctness[i]['wrong'], vol_label) / 100
            cc += c
            
            dwrite("{} - correct/volume:{}/{} [{}%], wrong/volume:{}/{} [{}%], vol_pred:{}/{}, ".format(
                category_name[i],
                category_correctness[i]['correct'], 
                vol_label, 
                c,
                category_correctness[i]['wrong'], 
                vol_label,
                w,
                vol_pred,
                vol_label
            ))

        s = cc / len(category_name)
        dwrite("All {}".format(s))
        
        metric = MulticlassJaccardIndex(num_classes=7)
        clabels = torch.tensor(labels)
        cpreds = torch.tensor(pred_labels)
        
        iou = metric(cpreds, clabels)
        dwrite("IOU: {}".format(iou))
        dice_score = Dice(average='micro')
        dice = dice_score(preds = cpreds, target = clabels)
        dwrite("DICE: {}".format(dice))

        
        #m = tf.keras.metrics.MeanIoU(num_classes=2)
        #m.update_state( preds = cpreds, target = clabels )
        #miou = m.result().numpy()
        #print("MIOU: {}".format(miou))
        
        dwrite(metrics.classification_report(labels, pred_labels, digits=7))
        dwrite(np.array2string(metrics.confusion_matrix(labels, pred_labels)))
        df.close()