import os
import time
from collections import OrderedDict
import numpy as np
from sklearn.model_selection import KFold
import torch
from torch import nn, t
from torch.utils.data import DataLoader
from torch_geometric.nn import summary

from dataset import VideoDataset
import myModel
from confusionMat import draw_confusion_matrix



class Args():
    def __init__(self):
        self.frames = 9
        self.batch = 16
        self.epoch = 20
        self.lr = 0.001
        self.kFlod = 5
        self.featuresNum = 350
        self.classNum = 7
args = Args()        
end = time.time()



# create by https://github.com/erkil1452/touch.git
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        


def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.data.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.data.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res[0], res[1]



class Trainer():
    def __init__(self):
        self.model =  myModel.CNNet(args.classNum).cuda()
        print(self.model)
        print(summary(self.model, torch.randn(args.batch, args.frames, 10, 16, device='cuda')))
        self.optimizer = torch.optim.Adam(self.model.parameters(), args.lr, weight_decay=args.lr / 100)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epoch * 2)
        self.criterion = nn.CrossEntropyLoss()
        self.best = { "top1" : 0, "epoch" : 0 }
        
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.lossese = AverageMeter()
        self.top1 = AverageMeter()
        self.top3 = AverageMeter()
        self.label_true = []
        self.label_pred = []

    def step(self, x, y, lenDataLoader, isTrain, isDrawMat):
        if isTrain:
            self.model.train() 
        else:
            self.model.eval()
        global end
        self.data_time.update(time.time() - end)
        if isTrain:
            self.optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()
        yPred = self.model(x)
        loss = self.criterion(yPred, y)
        if isTrain:
            loss.backward()
            self.optimizer.step()
        (prec1, prec3) = accuracy(yPred, y, topk=(1, min(3, args.classNum)))
        losses = OrderedDict([
            ('Loss', loss.data.item()),
            ('Top1', prec1),
            ('Top3', prec3),
        ])
        self.lossese.update(losses['Loss'], x.shape[0])
        self.top1.update(losses['Top1'], x.shape[0])
        self.top3.update(losses['Top3'], x.shape[0])
        # measure elapsed time
        self.batch_time.update(time.time() - end)
        end = time.time()
        if not isTrain and i == lenDataLoader - 1 and (self.best["top1"] < self.top1.avg):
            self.best["top1"] = self.top1.avg
            self.best["epoch"] = e
            torch.save(self.model, f"{self.model.__class__.__name__}_k{k}.pt")
        if isDrawMat:
            self.label_true.append(y)
            self.label_pred.append(np.argmax(yPred.cpu().detach(), axis=-1).flatten())
        print('[{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'
                .format(e, i, lenDataLoader, batch_time=self.batch_time, data_time=self.data_time, loss=self.lossese, top1=self.top1, top3=self.top3,))
        
    def drawMat(self):
        label_t = torch.cat(self.label_true, dim=0).cpu()
        label_p = torch.cat(self.label_pred, dim=0).cpu()
        draw_confusion_matrix(np.array(label_t.flatten()), np.array(label_p.flatten()), os.listdir("dataGlove"))



if __name__ == "__main__":
    kf = KFold(shuffle=True)
    for k, (trainIndex, validIndex) in enumerate(kf.split(np.zeros(args.featuresNum))):
        print(f"Fold {k}:")
        print(f"  Train: index={trainIndex}")
        print(f"  Test:  index={validIndex}")
        trainDataset = VideoDataset(trainIndex, True)
        validDataset = VideoDataset(validIndex, False)
        trainDataLoader = DataLoader(trainDataset, args.batch, True, num_workers=2)
        validDataLoader = DataLoader(validDataset, args.batch, False, num_workers=2)
        
        trainer = Trainer()
        
        for e in range(args.epoch):
            print(f"epoch: {e}")
            print("train")
            end = time.time()
            for i, (x, y) in enumerate(trainDataLoader):
                trainer.step(x, y, len(trainDataLoader), True, False)
            trainer.scheduler.step()
            print(f"lr: {trainer.optimizer.state_dict()['param_groups'][0]['lr']:.12f}")
            
            print("valid")
            end = time.time()
            trainer.batch_time.reset()
            trainer.data_time.reset()
            trainer.lossese.reset()
            trainer.top1.reset()
            trainer.top3.reset()
            for i, (x, y) in enumerate(validDataLoader):
                trainer.step(x, y, len(validDataLoader), False, False)
                
        print(f"\nbest epoch: {trainer.best['epoch']}")
        print(f"best score: {trainer.best['top1']}")
        end = time.time()
        trainer.batch_time.reset()
        trainer.data_time.reset()
        trainer.lossese.reset()
        trainer.top1.reset()
        trainer.top3.reset()
        trainer.model = torch.load(f"{trainer.model.__class__.__name__}_k{k}.pt")
        for i, (x, y) in enumerate(validDataLoader): 
            trainer.step(x, y, len(validDataLoader), False, True)  
        trainer.drawMat()            
