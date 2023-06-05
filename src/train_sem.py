
import os.path as osp
import torch
import torch.nn.functional as F
from datanet import DataNet
from torchmetrics.functional import jaccard_index
from torch_geometric.utils import scatter
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from point_transformer.model import Net as NetTrans
from pointnet2.model import Net as Net2
from pointnet2.model import get_optimizer as get_optimizer2
from pointnet2.model import get_transformations as get_transformations2
from pointnet2.model import get_pre_transformations as get_pre_transformations2
from datetime import datetime
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--use_model', type=str, default='all', help="Choose specific Model: Pointnet, Pointnet2 or PointTransformer")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
    parser.add_argument('--epoch', default=31, type=int, help='Epoch to run [default: 31]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--use_cpu', type=bool, default=False, help='Use the cpu [default: False]')
    parser.add_argument('--num_workers', type=int, default=2, help='Sets Dataloader Workers [default: 6]')

    return parser.parse_args()

def train(model, device, train_loader, optimizer):
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    length = len(train_loader)
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.pos, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes

        if (i + 1) % length == 0:
            print(f'[{i+1}/{length}] Loss: {total_loss / 10:.4f} '
                  f'Train Acc: {correct_nodes / total_nodes:.4f}')
            total_loss = correct_nodes = total_nodes = 0

@torch.no_grad()
def test(model, device, test_loader):
    model.eval()

    ious, categories = [], []
    y_map = torch.empty(test_loader.dataset.num_classes, device=device).long()
    for data in test_loader:
        data = data.to(device)
        outs = model(data.x, data.pos, data.batch)

        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for out, y, category in zip(outs.split(sizes), data.y.split(sizes),
                                    data.category.tolist()):
            category = list(DataNet.seg_classes.keys())[category]
            part = DataNet.seg_classes[category]
            part = torch.tensor(part, device=device)

            y_map[part] = torch.arange(part.size(0), device=device)

            #, absent_score=1.0
            iou = jaccard_index(out[:, part].argmax(dim=-1), y_map[y],
                                num_classes=part.size(0), task = 'multiclass')
            ious.append(iou)

        categories.append(data.category)

    iou = torch.tensor(ious, device=device)
    category = torch.cat(categories, dim=0)

    mean_iou = scatter(iou, category, reduce='mean')  # Per-category IoU.
    return float(mean_iou.mean())  # Global IoU.

def get_model(modelname, train_dataset, device):
    if(modelname == "pointnet"):
        print("choose pointnet")
        return None, None, None
    elif(modelname == "pointnet2"):
        model = Net2(train_dataset.num_classes).to(device)
        optimizer = get_optimizer2(model, args.learning_rate)
        return model, optimizer, None
    elif(modelname == "pointtransformer"):
        model = NetTrans(3, train_dataset.num_classes, dim_model=[32, 64, 128, 256, 512], k=16).to(device)
        optimizer = get_optimizer2(model, args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return model, optimizer, scheduler
    
        print("choose NONE!!")
    return None, None, None
        
def start_model(modelname, transform, pre_transform):
    print("load data...")
    train_dataset = DataNet(root = train_path, transform=transform,pre_transform=pre_transform,modelname=modelname)
    test_dataset = DataNet(root = test_path, transform=transform,pre_transform=pre_transform, test=True,modelname=modelname)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    
    model, optimizer, schedule = get_model(modelname, train_dataset, device)
    print("start training {}".format(modelname))
    for epoch in range(1, args.epoch):
        train(model=model, device=device, train_loader=train_loader, optimizer=optimizer)
        iou = test(model=model, device = device, test_loader=test_loader)
        print(f'{modelname} Epoch: {epoch:02d}, Test IoU: {iou:.4f}')
        
        if schedule is not None:
            schedule.step()
    torch.save(model, model_path + '/'+modelname+'_model_'+datetime.today().strftime('%Y-%m-%d_%H:%M:%S')+".pth")
    print("done training {}".format(modelname))
   
def get_transformations(modelname):
    print("load transformation functions...")
    if(modelname == "pointnet"):
        print("")
    elif(modelname == "pointnet2") or (modelname == "pointtransformer"):
        transform = get_transformations2()
        pre_transform = get_pre_transformations2()
        return transform, pre_transform
    return None, None

if __name__ == '__main__':
    print("load arguments...")
    args = parse_arguments()
    
    print("load train,test and models path...")
    model_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'models')
    train_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'binpicking', 'train')
    test_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'binpicking', 'test')
    
    print("choose device...")
    if torch.cuda.is_available() and not args.use_cpu:
        device = torch.device('cuda:'+str(args.gpu))
        torch.cuda.empty_cache()
        print("{} chosen".format('cuda:'+str(args.gpu)))
    else:
        device = torch.device('cpu')
        print("cpu chosen")

    modelname = args.use_model.lower()
    print("chose {}".format(modelname))
    if(modelname == "all"):
        models = ['pointtransformer', 'pointnet2']
        
        for modelname in models:
            transform, pre_transform = get_transformations(modelname)
            start_model(modelname, transform, pre_transform)
    else:
        transform, pre_transform = get_transformations(modelname)
        start_model(modelname, transform, pre_transform)
        
            
    