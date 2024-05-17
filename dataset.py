import os
from pyexpat import features
from torch.utils.data import Dataset
from torchvision.transforms import v2
import pandas as pd
import numpy as np
import torch
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

dataDir = "dataGlove"

class VideoDataset(Dataset):
    def __init__(self, kFlodIndex=None, isTrain = False):
        self.isTrain = isTrain
        self.features = []
        self.labels = []
        self.videoNum = 50
        self.classNum = len(os.listdir(dataDir))
        for j, (classDir) in enumerate(os.listdir(dataDir)):
            classDir = os.path.join(dataDir, classDir)
            for i in os.listdir(classDir):
                frames = []
                i = os.path.join(classDir, i)
                cluster = np.loadtxt(os.path.join(i, "keyFrame.txt"))
                _, ids = np.unique(cluster, True)
                ids = np.sort(ids)
                for frame in np.array(os.listdir(i))[ids]:
                    frame = os.path.join(i, frame)
                    frames.append(pd.read_csv(frame, header=None, index_col=False, usecols=range(16)))
                frames = np.stack(frames, 0)
                self.features.append(frames)
                self.labels.append(j)
        self.features = np.stack(self.features, 0)
        self.labels = np.array(self.labels)
        
        # tsne = TSNE(perplexity=50).fit(self.features.reshape(250, -1))
        # scatter = plt.scatter(tsne.embedding_[:, 0], tsne.embedding_[:, 1], c=self.labels, cmap="Set1")
        # plt.xlabel("t-SNE dimension 1")
        # plt.ylabel("t-SNE dimension 2")
        # plt.legend(handles=scatter.legend_elements()[0],labels=os.listdir(dataDir))
        # plt.show()

        self.mean = self.features.mean()
        self.std = self.features.std()
        self.max = self.features.max()
        self.features = self.features[kFlodIndex]
        self.labels = self.labels[kFlodIndex]
        self.features = (self.features - self.mean) / self.std
                
    def __getitem__(self, index):
        feature = torch.tensor(self.features[index], dtype=torch.float32)
        if self.isTrain:
            noise = (torch.randn(feature.shape) - 0.5) * 0.5
            feature = feature + noise
        # feature /= self.max
        # feature = v2.Normalize([self.mean], [self.std])(feature)
        label = torch.tensor(self.labels[index], dtype=torch.int64)
        return feature, label
    
    def __len__(self):
        return len(self.features)
    