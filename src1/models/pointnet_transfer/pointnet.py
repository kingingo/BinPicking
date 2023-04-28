from . import stn
import torch
from torch import nn
import torch.nn.functional as F


def _mlp(input_dim, *sizes):
    """
    Creates an MLP using the sizes as features dimensions
    E.g. 2-layer MLP with 3D input, 64D hidden features,
    and 64D output features is created using _mlp(3, 64, 64)

    Parameters:
    -----------
    input_dim: int
        Dimension of input
    sizes: int of variable length
        Dimension of each feature in MLP
    """
    assert(len(sizes) > 0)
    modules = []
    prev_dim = input_dim
    for sz in sizes:
        modules.append(nn.Conv1d(prev_dim, sz, kernel_size=1))
        modules.append(nn.BatchNorm1d(sz))
        modules.append(nn.ReLU(inplace=True))
        prev_dim = sz
    return nn.Sequential(*modules)


class PointNetFeatures(nn.Module):
    """
    Global feature extractor for point clouds

    Reference:
    ----------
    PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
    Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas
    arXiv:1612.00593v2 [cs.CV]
    (Fig. 2)
    """
    def __init__(self, input_dim=3, perform_input_transform=True,
                 perform_feature_transform=True):
        """
        Parameters:
        -----------
        input_dim: int
            Dimension of points in the input (default: 3)
        perform_input_transform: bool
            Whether to apply a Spatial Transformer Network on the input
            features
        perform_feature_transform: bool
            Whether to apply a Spatial Transformer Network on the intermediate
            features
        """
        super(PointNetFeatures, self).__init__()
        self._input_transform = perform_input_transform
        self._feature_transform = perform_feature_transform
        if self._input_transform:
            self.stn1 = stn.PointSpatialTransformer(dim=input_dim)
        self.mlp1 = _mlp(input_dim, 64, 64)
        if self._feature_transform:
            self.stn2 = stn.PointSpatialTransformer(dim=64)
        self.mlp2 = _mlp(64, 64, 128, 1024)

    def forward(self, x):
        trans_inp = None
        trans_feat = None
        if self._input_transform:
            trans_inp = self.stn1(x)
            x = torch.bmm(trans_inp, x)
        x = self.mlp1(x)
        if self._feature_transform:
            trans_feat = self.stn2(x)
            x = torch.bmm(trans_feat, x)
        x = self.mlp2(x)
        x = torch.max(x, dim=2, keepdim=True)[0]
        return x, trans_inp, trans_feat
