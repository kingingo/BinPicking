import os
import os.path as osp
import torch
import torch.nn.functional as F
from model import Net
from torchmetrics.functional import jaccard_index
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from datanet import DataNet
import torch
from datetime import datetime
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--use_cpu', type=bool, default=False, help='Use the cpu [default: False]')

    return parser.parse_args()

def train():
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes

        if (i + 1) % 10 == 0:
            print(f'[{i+1}/{len(train_loader)}] Loss: {total_loss / 10:.4f} '
                  f'Train Acc: {correct_nodes / total_nodes:.4f}')
            total_loss = correct_nodes = total_nodes = 0


@torch.no_grad()
def test(loader):
    model.eval()

    ious, categories = [], []
    y_map = torch.empty(loader.dataset.num_classes, device=device).long()
    for data in loader:
        data = data.to(device)
        outs = model(data)

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


if __name__ == '__main__':
    args = parse_arguments()
    
    if torch.cuda.is_available() and not args.use_cpu:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print('Available devices ', torch.cuda.device_count())
        print('Active CUDA Device: GPU', torch.cuda.current_device())
        
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    
    model_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'models')
    train_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'binpicking', 'train')
    test_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'binpicking', 'test')
    transform = T.Compose([
        T.RandomJitter(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2)
    ])
    pre_transform = T.NormalizeScale()

    train_dataset = DataNet(root = train_path, transform=transform,pre_transform=pre_transform)
    test_dataset = DataNet(root = test_path, transform=transform,pre_transform=pre_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=6)

    model = Net(train_dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(1, args.epoch):
        train()
        iou = test(test_loader)
        print(f'Epoch: {epoch:02d}, Test IoU: {iou:.4f}')
        
    torch.save(model, model_path + '/model_'+datetime.today().strftime('%Y-%m-%d_%H:%M:%S')+".pth")
