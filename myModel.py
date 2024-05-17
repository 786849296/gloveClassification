import torch
from torch import nn
from torchvision import models
from torch.nn import functional as F

from resnet_3x3 import resnet18 as touchRes



class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class SpatialAttention(nn.Module):
    def __init__(self, k = 3):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=k, stride=1, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_att = torch.cat((x_max, x_avg), dim=1)
        x_att = self.conv(x_att)
        w = self.sigmoid(x_att)
        return x * w

class CBAMBlock(nn.Module):
    def __init__(self, in_channel, radio, k=3):
        super(CBAMBlock, self).__init__()
        self.CAMaxPool = nn.AdaptiveMaxPool2d(1)
        self.CAAvgPool = nn.AdaptiveAvgPool2d(1)
        self.CAConv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // radio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // radio, in_channel, 1, bias=False),
        )
        self.SAConv = nn.Conv2d(2, 1, kernel_size=k, stride=1, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_max = self.CAMaxPool(self.CAConv(x))
        x_avg = self.CAAvgPool(self.CAConv(x))
        w = self.sigmoid(x_max + x_avg)
        x = w * x
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_att = torch.cat((x_max, x_avg), dim=1)
        x_att = self.SAConv(x_att)
        w = self.sigmoid(x_att)
        return x * w
    
class SpatialAttention3d(nn.Module):
    def __init__(self, k = 3):
        super(SpatialAttention3d, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=k, stride=1, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_att = torch.cat((x_max, x_avg), dim=1)
        x_att = self.conv(x_att)
        w = self.sigmoid(x_att)
        return x * w



class R3d18(nn.Module):
    def __init__(self, num_classes):
        super(R3d18, self).__init__()
        self.res = models.video.r3d_18()
        self.res.stem[0] = nn.Conv3d(1, 64, 3)
        self.res.fc = nn.Linear(512, num_classes)
        # self.res.layer3 = nn.Identity()
        # self.res.layer4 = nn.Identity()

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.res(x)
        return x



# create by https://github.com/erkil1452/touch.git
class TouchNet(nn.Module):
    '''
    This model represents our classification network for 1..N input frames.
    '''

    def __init__(self, num_classes=27, nFrames=5):
        super(TouchNet, self).__init__()
        self.net = touchRes()
        self.combination = nn.Conv2d(128*nFrames, 128, kernel_size=1, padding=0)
        self.classifier = nn.Linear(128, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # CNN of each input frame
        xs=[]
        for i in range(x.size(1)):
            xi = x[:,i:i+1,...]
            xi = self.net(xi)
            xs += [xi]
        x = torch.cat(xs, dim=1)

        # combine
        x = self.combination(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        return x
    


class GloveNet(nn.Module):
    def __init__(self, num_classes, frames):
        super(GloveNet, self).__init__()
        # self.posEncode = SpatialAttention()
        self.finger = nn.Sequential(
            nn.GRU(45, 16, batch_first=True, dropout=0.3, bidirectional=True),
            SelectItem(0),
            nn.ReLU(),
            nn.Linear(32, 5),
            nn.Flatten())
        self.ffb = nn.Sequential(
            nn.GRU(10, 16, batch_first=True, dropout=0.3, bidirectional=True),
            SelectItem(0),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Flatten())
        self.palm = models.video.r3d_18()
        self.palm.stem[0] = nn.Conv3d(1, 64, 3)
        # self.palm.layer2 = nn.Identity()
        self.palm.layer3 = nn.Identity()
        self.palm.layer4 = nn.Identity()
        self.palm.fc = nn.Linear(128, 100)
        self.fc = nn.LazyLinear(num_classes)
        
    def forward(self, x):
        # x = self.posEncode(x)
        f1 = x[..., 0:3, 0:3]
        f2 = x[..., 0:3, 3:6]
        f3 = x[..., 3:6, 0:3]
        f4 = x[..., 3:6, 3:6]
        f5 = x[..., 6:9, 0:3]
        finger = torch.stack([f1, f2, f3, f4, f5], dim=1).reshape(x.shape[0], x.shape[1], -1);
        ffb = x[..., 6:9, 3:6]
        ffb = ffb.reshape(x.shape[0], x.shape[1], -1)
        ffb = torch.cat([ffb, x[..., 9, 0].unsqueeze(2)], dim=2)
        palm = x[..., 0:10, 6:16].unsqueeze(1)
        finger = self.finger(finger)
        ffb = self.ffb(ffb)
        palm = self.palm(palm)
        x = torch.cat([palm, finger, ffb], dim=1)
        x = self.fc(x)
        return x
    


# create by https://github.com/YunzhuLi/senstextile/blob/master/classification/object_classification/models.py
class CNNet(nn.Module):
    def __init__(self, num_classes):
        super(CNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, 5, padding=2)
        self.gru = nn.GRU(64, 120, num_layers=2, bidirectional=True)
        self.fc1 = nn.Linear(120 * 2, 84)
        self.fc2 = nn.Linear(84, num_classes)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # x: [B, T, H, W]
        B, T, H, W = x.size()
        x = self.pool(F.relu(self.conv1(x.view(B * T, 1, H, W))))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(B, T, -1).transpose(0, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = torch.cat([x[-1, :, :120], x[0, :, 120:]], 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x