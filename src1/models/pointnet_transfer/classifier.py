from torch import nn
import torch.nn.functional as F
from .pointnet import PointNetFeatures, _mlp


class PointNetClassifier(nn.Module):
    """
    Point cloud classifier network

    Reference:
    ----------
    PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
    Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas
    arXiv:1612.00593v2 [cs.CV]
    (Supplementary Section C)
    """
    def __init__(self, num_classes, input_dim=3):
        super(PointNetClassifier, self).__init__()
        self.feature_extractor = PointNetFeatures(input_dim)
        self.classifier = _mlp(1024, 512, 256)
        self.output = nn.Linear(256, num_classes)

    def forward(self, x):
        global_feat, trans_inp, trans_feat = self.feature_extractor(x)
        out = self.classifier(global_feat)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.3, training=self.training)
        out = self.output(out)
        return out, trans_inp, trans_feat
