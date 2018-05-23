# import packages
import argparse
import numpy as np
import random
import os
import sys
import shutil
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import Logger

# import torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms
import torchvision.datasets as dset
import torchvision.transforms as transforms
from models.projsgd import ProjSGD


# Define Parameters
ARCH = "alexnet"
DATASET = "cifar10"
DATAPATH = "~/dataset/"
RANDOMSEED = None
USE_PRETRAINED = False  # only worked for imagenet
LEARNING_RATE = 1e-1
MOMENTUM_RATE = 0.9
WEIGHT_DECAY = 5e-4
EPOCHES = 164
LRDECAY_EPOCHES = 30
LRDECAY_SCHEDULE = []
LRDECAYRATE = 0.1
BATCH_SIZE = 128

PRINT_FREQ = 100
SHOW_PROGRESS = False
SHOW_SV_INFO = False # show Singular Value Info
SHOW_SV_EPOCH = []

RESUME_PATH = ''
START_EPOCH = 0
LOAD_WORKER = 4
EVALUATE = False

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    print('GPU IS AVAILABLE TO USE')
    cudnn.benchmark = True
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
Tensor = FloatTensor

if DATASET.startswith("cifar"):
    import models.cifar as models
else:
    import models.imagenet as models

stdout_backup = sys.stdout

def main():
    # make dictionary to store model
    ROOTPATH = os.path.dirname(os.path.abspath(__file__)) + "/results/"
    ROOTPATH += ARCH+"_"+DATASET
    print('ROOTPATH is %s' %ROOTPATH)
    if not os.path.exists(ROOTPATH):
        os.mkdir(ROOTPATH)

    sys.stdout = Logger(ROOTPATH+"/log.txt","w", stdout_backup)

    # if gpu is to be used

    best_prec1 = 0

    # Random seed
    global RANDOMSEED
    if RANDOMSEED == None:
        RANDOMSEED = random.randint(1, 10000)
    random.seed(RANDOMSEED)
    torch.manual_seed(RANDOMSEED)
    if USE_CUDA:
        torch.cuda.manual_seed_all(RANDOMSEED)
    print("use random seed: "+str(RANDOMSEED))

    # setup data loader
    if DATASET.startswith('cifar'):
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        train_transform = torchvision.transforms.Compose(
            [torchvision.transforms.RandomHorizontalFlip(),
             torchvision.transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(),
             torchvision.transforms.Normalize(mean, std)])
        test_transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
        if DATASET == 'cifar10':
            train_data = dset.CIFAR10(DATAPATH, train=True, transform=train_transform, download=True)
            test_data = dset.CIFAR10(DATAPATH, train=False, transform=test_transform, download=True)
            num_classes = 10
        elif DATASET == 'cifar100':
            train_data = dset.CIFAR100(DATAPATH, train=True, transform=train_transform, download=True)
            test_data = dset.CIFAR100(DATAPATH, train=False, transform=test_transform, download=True)
            num_classes = 100
        print('Number of training samples: ', len(train_data))
        print('Number of testing samples: ', len(test_data))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                                                   num_workers=LOAD_WORKER, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,
                                                  num_workers=LOAD_WORKER, pin_memory=True)
    else:
        raise NotImplementedError

    # set up model
    if USE_PRETRAINED:
        print("=> using pre-trained model '{}'".format(ARCH))
        model = models.__dict__[ARCH](num_classes=num_classes,pretrained=True).to(DEVICE)
    else:
        print("=> creating model '{}'".format(ARCH))
        model = models.__dict__[ARCH](num_classes=num_classes).to(DEVICE)
    if USE_CUDA:
        model = torch.nn.DataParallel(model)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print(model)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM_RATE, weight_decay=WEIGHT_DECAY)

    # check if we can load checkpoint
    global START_EPOCH
    START_EPOCH = 0
    if RESUME_PATH:
        if os.path.isfile(RESUME_PATH):
            print("=> loading checkpoint '{}'".format(RESUME_PATH))
            checkpoint = torch.load(RESUME_PATH)
            START_EPOCH = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_d.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(RESUME_PATH, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(RESUME_PATH))

    if EVALUATE:
        validate(train_loader, model, criterion)
        validate(test_loader, model, criterion)
        return

    # Dense Training
    print("=================================================================")
    print("====================Start Dense Training=========================")
    print("=================================================================")
    train_acu_top1 = []
    train_acu_top5 = []
    test_acu_top1 = []
    test_acu_top5 = []
    for epoch in range(START_EPOCH, EPOCHES):
        adjust_learning_rate(optimizer, epoch,LEARNING_RATE,LRDECAY_SCHEDULE)

        # train for one epoch
        acu_top1,acu_top5 = train(train_loader, model, criterion, optimizer, epoch, verbose = SHOW_PROGRESS)
        train_acu_top1.append(acu_top1)
        train_acu_top5.append(acu_top5)

        # evaluate on validation set
        acu_top1, acu_top5 = validate(test_loader, model, criterion, verbose = False)
        test_acu_top1.append(acu_top1)
        test_acu_top5.append(acu_top5)

        # remember best prec@1 and save checkpoint
        is_best = acu_top1 > best_prec1
        best_prec1 = max(acu_top1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': ARCH,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, epoch, savepath = ROOTPATH)

        if SHOW_SV_INFO:
            if SHOW_SV_EPOCH == [] or epoch in SHOW_SV_EPOCH:
                model.showOrthInfo()

    np.save(ROOTPATH+"/acu.npy",{"train_acu_top1":train_acu_top1
                                ,"train_acu_top5":train_acu_top5,
                                "test_acu_top1":test_acu_top1
                                ,"test_acu_top5":test_acu_top5})

    # Plot the performance
    epoches = range(0, EPOCHES)

    plt.plot(epoches, test_acu_top1, 'r-', label = 'test_acu1')
    plt.plot(epoches, test_acu_top5, 'r--', label = 'test_acu5')
    plt.legend()
    plt.savefig(ROOTPATH+'/test_performance.pdf', bbox_inches='tight',format="pdf", dpi = 300)
    plt.close()
    plt.plot(epoches, train_acu_top1, 'r-', label = 'train_acu1')
    plt.plot(epoches, train_acu_top5, 'r--', label = 'train_acu5')
    plt.legend()
    plt.savefig(ROOTPATH+'/train_performance.pdf', bbox_inches='tight',format="pdf", dpi = 300)
    plt.close()


def train(train_loader, model, criterion, optimizer, epoch, verbose = True, verbose_sum = True, prune = False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(DEVICE)
        target = target.to(DEVICE)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if prune:
            model.prune(sparsity = SPARSITY, create_mask = False, prune_bias = PRUNEBIAS)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and i % PRINT_FREQ == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    if verbose_sum:
        print('Training Epoch [{0}]:\t'
              '  Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(epoch, top1=top1, top5=top5))
    return top1.avg,top5.avg


def validate(test_loader, model, criterion, verbose = True, verbose_sum = True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):

        input = input.to(DEVICE)
        target = target.to(DEVICE)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and i % PRINT_FREQ == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(test_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    if verbose_sum:
        print('Testing:  Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg,top5.avg


def save_checkpoint(state, is_best, epoch, savepath='./'):
    torch.save(state, savepath+'/checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(savepath+'/checkpoint.pth.tar', savepath+'/model_best.pth.tar')

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

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, epoch, inilr, lr_schedule = []):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_schedule == []:
        lr = inilr * (LRDECAYRATE ** (epoch // LRDECAY_EPOCHES))
    else:
        lr = inilr
        for schedule_epo in lr_schedule:
            if epoch >= schedule_epo:
                lr = lr * LRDECAYRATE
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser(description='Write your test code in a saperate py file.')
parser.add_argument('--filepath', '-f', metavar='filepath', required=True,
                    help='the path of the test file, which should be a py file')
args = parser.parse_args()

if __name__ == '__main__':
    exec(open(args.filepath).read())
    # The old tests, may be not working anymore
    # exec(open("./oldtest.py").read())
