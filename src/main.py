
import torch_geometric.transforms as T
import torch
import models.pointnet.pointnet as pointnet
from models.pointnet_transfer.classifier import PointNetClassifier
import torch.nn.functional as F
from torch_geometric.datasets import ModelNet
import models.pointnet_transfer.modelnet as dataset
from torch_geometric.loader import DataLoader
import os.path as osp
import argparse
import util
import time
import json
from models.pointnet_transfer.stn import orthogonality_constraint

def create_transformation_list():
    transformations = [];
    num = 1024;
    
    transformations.append(['SamplePoints',T.SamplePoints(num)]);
    transformations.append(['FixedPoints',T.FixedPoints(num)]);
    return transformations;

def create_pre_transformation_list():
    pre_transformations = [];
    
    pre_transformations.append(['NormalizeScale',T.NormalizeScale()]);
    
    return pre_transformations;
    
def get_optimizer(name, model):
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=0.001);
    elif name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=0.001)
    elif name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    

def create_optimization_list():
    optimizer_list = [];
    
    optimizer_list.append('Adam');
    optimizer_list.append('AdamW');
    optimizer_list.append('SGD');
    
    return optimizer_list;

def train(model, model_name, dataloader, optimizer, epoch, device, print_freq=10):
    model.train()
    
    avg_loss = util.AverageMeter()
    avg_time = util.AverageMeter()

    for i, (inputs, labels) in enumerate(dataloader, 0):
        start = time.time()
        
        optimizer.zero_grad()
        
        if model_name == 'pointnet_transfer':
            inputs, labels = inputs.to(device), labels.to(device)
            output, trans_inp, trans_feat = model(inputs)
            loss = F.cross_entropy(output, labels)
            if trans_inp is not None:
                loss += 0.001 * orthogonality_constraint(trans_inp)
            if trans_feat is not None:
                loss += 0.001 * orthogonality_constraint(trans_feat)
        else:
            inputs = inputs.to(device)
            loss = F.nll_loss(model(inputs), inputs.y)
            
        loss.backward()
        optimizer.step()
        
        #measuring training time
        end = time.time()
        avg_loss.update(loss.item())
        avg_time.update(end - start)
        
        if i > 0 and i % print_freq == 0:
            print('Train Epoch {:3} [{:3.0f}% of {}]: Loss: {:6.3f}'
                  .format(epoch, (i + 1) / len(dataloader) * 100.0,
                          len(dataloader.dataset), loss.item()))
    return avg_loss.val, avg_time.val


def test(model, model_name, dataloader, epoch, device):
    model.eval()
    
    avg_loss = util.AverageMeter()
    avg_time = util.AverageMeter()

    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader, 0):
            start = time.time()
            
            inputs = inputs.to(device)
            output = model(inputs)
            
            if model_name == 'pointnet_transfer':
                labels = labels.to(device)
                
                loss = F.cross_entropy(output[0], labels)
                avg_loss.update(loss.item())
                pred = torch.max(output.data, dim=1)[1]
            else:
                labels = inputs.y
                pred = output.max(1)[1]
                
            end = time.time()
            avg_time.update(end - start)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    acc = float(correct) / float(total)
    print('Test Epoch {:3}: Avg. loss: {:6.3f}, Accuracy: {:.2%}, Avg. Time/batch: {:5.3f}s'
          .format(epoch, avg_loss.val, acc, avg_time.val))
    return avg_loss.val, avg_time.val, acc

if __name__ == '__main__':
    
    #Argument Parser
    parser = argparse.ArgumentParser(description="Script for training a \ Pointnet classifier");
    parser.add_argument("--model", type=str, choices=("pointnet","pointnet_transfer"), help="Choose a model");
    parser.add_argument("--dataset", type=str, choices=("ModelNet40",), help="Name of dataset");
    args = parser.parse_args();
    
    num_classes = 0
    if args.dataset == "ModelNet40":
        num_classes = 40
    
    #Choose model
    modelname = args.model;
    
    
    print("Training PointNet Classifier")
    print("Dataset: data/{}".format(args.dataset))
    print("Model: {}".format(modelname))
    
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..',
                    'data/' + args.dataset)
    optimizer_list = create_optimization_list();
    pre_transformation_list = create_pre_transformation_list();
    transformation_list = create_transformation_list();
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for opt_name in optimizer_list:
        for pre_trans_set in pre_transformation_list:
            for trans_set in transformation_list:
                

                if modelname == 'pointnet_transfer':
                    path = osp.join(path, 'raw')
                    train_dataset = dataset.ModelNetTrans(path, 2048,True)
                    test_dataset = dataset.ModelNetTrans(path, 2048,False);
                    
                    model = PointNetClassifier(num_classes=num_classes)
                else:
                    train_dataset = pointnet.ModelNetPoint(path, '10', True, trans_set[1], pre_trans_set[1])
                    test_dataset = pointnet.ModelNetPoint(path, '10', False, trans_set[1], pre_trans_set[1])
                    
                    model = pointnet.Net().to(device)
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                        num_workers=6)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                                        num_workers=6)
                
                for i, (a,o) in enumerate(train_loader, 0):
                    print(a);
                    print(o);
                        
                optimizer = get_optimizer(opt_name, model);
                        
                for epoch in range(1, 201):
                    train(model, modelname, train_loader, optimizer,epoch, device)
                    avg_loss, avg_time, test_acc = test(model, modelname, test_loader, epoch, device)
                    print(f'Opt: {opt_name}, Trans: {trans_set[0]}, Pretrans: {pre_trans_set[0]}, Epoch: {epoch:03d}, Test: {test_acc:.4f}')
                
                save_path = osp.join(osp.dirname(osp.realpath(__file__)), 'models', opt_name, pre_trans_set[0], trans_set[0]);
                osp.mkdir(save_path)
                torch.save(model, save_path);
                print(f'save model with Opt: {opt_name}, Trans: {trans_set[0]}, Pretrans: {pre_trans_set[0]}');
    
    