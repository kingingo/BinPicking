
import torch_geometric.transforms as T
import torch
import models.pointnet.pointnet as pointnet
from models.pointnet_transfer.classifier import PointNetClassifier
import torch.nn.functional as F
from torch_geometric.datasets import ModelNet
import models.pointnet_transfer.modelnet as dataset
from torch_geometric.loader import DataLoader
import os
import os.path as osp
import argparse
import util
import time
import json
from models.pointnet_transfer.stn import orthogonality_constraint
import logging

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

def plog(msg):
    logging.info(msg)
    print(msg)

def train(model, model_name, dataloader, optimizer, epoch, device, print_freq=10):
    model.train()
    
    avg_loss = util.AverageMeter()
    avg_time = util.AverageMeter()

    for i, (inputs, labels) in enumerate(dataloader, 0):
        start = time.time()
        
        optimizer.zero_grad()
        inputs = inputs.to(device)
        
        if model_name == 'pointnet_transfer':
            labels = labels.to(device)
            output, trans_inp, trans_feat = model(inputs)
            loss = F.cross_entropy(output, labels)
            if trans_inp is not None:
                loss += 0.001 * orthogonality_constraint(trans_inp)
            if trans_feat is not None:
                loss += 0.001 * orthogonality_constraint(trans_feat)
        else:
            loss = F.nll_loss(model(inputs), inputs.y)
            
        loss.backward()
        optimizer.step()
        
        #measuring training time
        end = time.time()
        avg_loss.update(loss.item())
        avg_time.update(end - start)
        
        if i > 0 and i % print_freq == 0:
            plog('Train Epoch {:3} [{:3.0f}% of {}]: Loss: {:6.3f}'
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
                output = output[0]
                loss = F.cross_entropy(output, labels)
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
    plog('Test Epoch {:3}: Avg. loss: {:6.3f}, Accuracy: {:.2%}, Avg. Time/batch: {:5.3f}s'
          .format(epoch, avg_loss.val, acc, avg_time.val))
    return avg_loss.val, avg_time.val, acc
    

def start_model(model, train_dataset, test_dataset, opt_name, device, file_name):
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                            num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                            num_workers=6)
            
    optimizer = get_optimizer(opt_name, model);
    log_name = file_name + ".log";
    logging.basicConfig(filename=log_name, encoding='utf-8', level=logging.DEBUG)        
    for epoch in range(1, 201):
        train(model, modelname, train_loader, optimizer,epoch, device)
        avg_loss, avg_time, test_acc = test(model, modelname, test_loader, epoch, device)
        plog(f'Opt: {opt_name}, Trans: {trans_set[0]}, Pretrans: {pre_trans_set[0]}, Epoch: {epoch:03d}, Test: {test_acc:.4f}')
    
    save_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'models');
    
    if not osp.exists(save_path):
        os.mkdir(save_path)
    
    save_path += file_name + '.model';
    torch.save(model, save_path);
    plog(f'save model with Opt: {opt_name}, Trans: {trans_set[0]}, Pretrans: {pre_trans_set[0]}');

if __name__ == '__main__':
    
    #Argument Parser
    parser = argparse.ArgumentParser(description="Script for training a \ Pointnet classifier");
    parser.add_argument("--model", type=str, choices=("pointnet","pointnet_transfer", "all"), help="Choose a model");
    parser.add_argument("--dataset", type=str, choices=("ModelNet40",), help="Name of dataset");
    parser.add_argument("--gpu_id", type=str, choices=("0","1","2","3"), help="Choose the gpu device id");
    args = parser.parse_args();
    
    num_classes = 0
    if args.dataset == "ModelNet40":
        num_classes = 40
    
    if args.model == 'all':
        models = ["pointnet","pointnet_transfer"];
    else:
        models = [args.model];
    
    for modelname in models:
        print("Training PointNet Classifier")
        print("Dataset: data/{}".format(args.dataset))
        print("Model: {}".format(modelname))
        
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..',
                        'data/' + args.dataset)
        optimizer_list = create_optimization_list();
        pre_transformation_list = create_pre_transformation_list();
        transformation_list = create_transformation_list();
        
        if torch.cuda.is_available():
            device = torch.device('cuda:'+args.gpu_id)
        else:
            device = torch.device('cpu')
            
        
        for opt_name in optimizer_list:
            if modelname == 'pointnet_transfer':
                log_name = modelname + "_" + opt_name;
                path = osp.join(path, 'raw')
                train_dataset = dataset.ModelNetTrans(path, 1024,True)
                test_dataset = dataset.ModelNetTrans(path, 1024,False);
                
                model = PointNetClassifier(num_classes=num_classes)
                start_model(model, train_dataset, test_dataset, opt_name, device, log_name);
            elif modelname == 'pointnet':
                for pre_trans_set in pre_transformation_list:
                    for trans_set in transformation_list:
                        log_name = modelname + "_" + opt_name + "_" + pre_trans_set[0] + "_" + trans_set[0];
                        train_dataset = pointnet.ModelNetPoint(path, '10', True, trans_set[1], pre_trans_set[1])
                        test_dataset = pointnet.ModelNetPoint(path, '10', False, trans_set[1], pre_trans_set[1])
                        
                        model = pointnet.Net()
                        start_model(model, train_dataset, test_dataset, opt_name, device, log_name);
                    
                    
    
    