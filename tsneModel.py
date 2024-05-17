import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from dataset import VideoDataset



data = VideoDataset(range(0, 250))
dataLoader = DataLoader(data, 9, False)
model = torch.load("GloveNet_k4.pt").cpu()
model.fc = torch.nn.Identity()
model.eval()
label = []
features = []
for x, y in dataLoader:
    label.append(y)
    x = model(x)
    features.append(x.reshape(x.shape[0], -1))
features = torch.cat(features, 0).detach().numpy()
label = torch.cat(label, 0).detach().numpy()

tsne = TSNE(perplexity=50).fit(features)
scatter = plt.scatter(tsne.embedding_[:, 0], tsne.embedding_[:, 1], c=label, cmap="Set1")
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.legend(handles=scatter.legend_elements()[0],labels=os.listdir("dataGlove"))
plt.show()