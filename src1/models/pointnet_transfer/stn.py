import torch
from torch import nn
import torch.nn.functional as F


def _init_identity(module, dim):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 0.0)
        with torch.no_grad():
            module.bias.data = torch.eye(dim).view(-1)


class PointSpatialTransformer(nn.Module):
    """
    Spatial Transformer Network for point clouds
    
    Reference:
    ----------
    PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
    Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas
    arXiv:1612.00593v2 [cs.CV]
    (Supplementary Section C)
    """
    def __init__(self, dim=3):
        super(PointSpatialTransformer, self).__init__()
        self.dim = dim
        self.features = nn.Sequential(nn.Conv1d(dim, 64, kernel_size=1),
                                      nn.BatchNorm1d(64),
                                      nn.ReLU(True),
                                      nn.Conv1d(64, 128, kernel_size=1),
                                      nn.BatchNorm1d(128),
                                      nn.ReLU(True),
                                      nn.Conv1d(128, 1024, kernel_size=1),
                                      nn.BatchNorm1d(1024),
                                      nn.ReLU(True))

        self.regressor = nn.Sequential(nn.Linear(1024, 512),
                                       nn.BatchNorm1d(512),
                                       nn.ReLU(True),
                                       nn.Linear(512, 256))

        self.transform = nn.Linear(256, dim * dim)

        # Initialize initial transformation to be identity
        _init_identity(self.transform, dim)


    def forward(self, x):
        x = self.features(x)
        x = torch.max(x, dim=2, keepdim=True)[0]
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        x = self.transform(x)
        # resize x into 2D square matrix
        x = x.view(x.size(0), self.dim, self.dim)
        return x


def orthogonality_constraint(A):
    """
    Regularization function to constrain A as an orthogonal matrix

    Parameters:
    -----------
    A : torch.Tensor
        A batch square matrix of dimensions batch_size x dim x dim
    """
    batch = A.size(0)
    dim = A.size(1)
    I = torch.eye(dim)
    I = I.reshape((1, dim, dim))
    I = I.repeat(batch, 1, 1)
    if A.is_cuda:
        I = I.cuda()
    return F.mse_loss(I, torch.bmm(A, A.transpose(1, 2)))
