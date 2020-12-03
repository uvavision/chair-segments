import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import os
import argparse
from tqdm import tqdm as tqdm
import time, os

import datasets
from models import *
#from models import *
from train import *
from test import *
from visualization import *


model_list = {'unet': unet(), 'fcnvgg16': fcnvgg16(), 'fcnresnet50' : fcnresnet50(), 'fcnresnet101': fcnresnet101()}

parser = argparse.ArgumentParser(description='PyTorch Chair Segments Training')
parser.add_argument('--data', metavar='DIR', default='data',
                    help='path to dataset')
parser.add_argument('--dataset', metavar='DIR', default='Chair',
                    help='"Chair" for ChairSegments')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--arch', '-a', metavar='ARCH', default='unet',
                    choices=model_list,
                    help='model architecture: ' +
                        ' | '.join(model_list) +
                        ' (default: unet) other options: fcnvgg16, fcnresnet50, fcnresnet101')
parser.add_argument('--batchSize', '-b', default=10, type=int,
                    metavar='N',
                    help='mini-batch size (default: 10), this is the total ')
parser.add_argument('--criterion', '-crit', default='BCE',
                    help='criterion')
parser.add_argument('--optimizer', '-opt', default='Adam',
                    help='optimizer Adam or SGD')
parser.add_argument('--momentum', '-mome',  default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight_decay', '-wdecay',  default=1e-5, type=float,
                    help='weight_decay')
parser.add_argument('--lines', '-lines',  default=3,
                    help='lines, 3 by default')
parser.add_argument('--size', '-s',  default=10,
                    help='images per line, 10 by default')
parser.add_argument('--epochs', '-e',  default=1, type=int,
                    help='epochs , 10 by default')


args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# Transform
rsizet      =  transforms.Compose([
                    transforms.Resize([64,64]),
                    transforms.ToTensor()
                ])

# Data
dspath      = args.data + '/' + args.dataset 

trainset    = datasets.CHAIRS2020(dspath + '064/', split = 'train', transform = rsizet)
valset      = datasets.CHAIRS2020(dspath + '064/', split = 'val',   transform = rsizet)


trainLoader = torch.utils.data.DataLoader(dataset = trainset,
                                            batch_size = args.batchSize,
                                            shuffle = True)

valLoader   = torch.utils.data.DataLoader(dataset = valset,
                                            batch_size = args.batchSize,
                                            shuffle = False) # No need.

print('==> Generating Images and Ground Truth..')
visualizeImages(args.dataset, 'results/' + args.dataset, args.size, args.lines, valset, rsizet)

# Model
print('==> Building ' + args.arch +' model..')
model = model_list[args.arch]
model = model.to(device)

#criterion
criterion = nn.BCEWithLogitsLoss() if args.criterion=='BCE' else nn.CrossEntropyLoss()
#optimizer
print('optimizer: ' + args.optimizer)
if (args.optimizer == 'Adam'):
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
elif(args.optimizer == 'SGD'):
    optimizer = optim.SGD(model.parameters(), args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
else:
    optimizer = optim.RMSprop(model.parameters(), args.lr, momentum = args.momentum, weight_decay = args.weight_decay)

type_train  = 'scratch'
start_epoch = 0
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('logs'), 'Error: no checkpoint directory found!'
    modelname = args.dataset + '064_' + args.arch + '_' + type_train + '.pth'
    checkpoint = torch.load(os.path.join('logs', modelname))
    model.load_state_dict(checkpoint['state_dict'])
    
    msg  =  '\n' + modelname
    msg  += ', best_loss: ' + str(checkpoint['best_loss'])
    msg  += ', best_iou: ' + str(checkpoint['iou'])
    msg  += ', best_dice: ' + str(checkpoint['dice'])
    msg  += ', best_prec: ' + str(checkpoint['prec'])
    msg  += ', best_epoch: ' + str(checkpoint['epoch'])
    print(msg)
    file1 = open('resumefile.txt',"a")
    file1.write(msg)
    file1.close()
    start_epoch = checkpoint['epoch']


model.cuda()
criterion.cuda()

train_losses, train_iou, train_dice, train_prec = [], [], [], []
val_losses,   val_iou,   val_dice,   val_prec   = [], [], [], []

best_loss, best_epoch       = 1, 0
max_iou, max_dice, max_prec = 0, 0, 0

epochs      = args.epochs
best_acc    = 0  # best test accuracy
dataname    = args.dataset + '064_' + args.arch + '_' + type_train
print(dataname)
tt          = 0
st          = time.time()
for epoch in range(start_epoch, epochs + 1):  # Number of epochs.

    cumloss, totaliou, totaldice, totalprec = 0, 0, 0, 0
    # Perform a round of training.  ##########################################################
    
    cumloss, totaliou, totaldice, totalprec = trainRound(args.arch, model, args.dataset, 
                                                                trainLoader, optimizer, criterion, epoch)
    print('cumloss: ' + str(cumloss) + ', iou: ' +  str(totaliou) + ', dice:  ' + str(totaldice) + ', prec:  ' + str(totalprec) )
    
    train_losses.append(cumloss)
    train_iou.append(totaliou)
    train_dice.append(totaldice)
    train_prec.append(totalprec)

    # Perform a round of validation.  ##########################################################
    
    cumloss, totaliou, totaldice, totalprec = valRound(args.arch, model, args.dataset, 
                                                                valLoader, criterion, epoch)
    print('cumloss: ' + str(cumloss) + ', iou: ' +  str(totaliou) + ', dice:  ' + str(totaldice) + ', prec:  ' + str(totalprec) )

    iou  = totaliou
    dice = totaldice
    prec = totalprec

    val_losses.append(cumloss)
    val_iou.append(totaliou)
    val_dice.append(totaldice)
    val_prec.append(totalprec)

    # remember best acc@1 and save checkpoint
    is_best = best_loss > cumloss

    if(is_best):
        tt = time.time() - st
        best_epoch = epoch
        max_iou, max_dice, max_prec, best_loss  = max(iou,  max_iou), max(dice, max_dice), max(prec, max_prec), min(cumloss, best_loss)
        plots('results/'+ dataname, val_losses, train_losses, val_iou, train_iou, val_dice, train_dice, val_prec, train_prec)
        
        state = {
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_loss': best_loss,
        'optimizer' : optimizer.state_dict(),
        'iou': max_iou,
        'dice': max_dice,
        'prec': max_prec
        }
        if not os.path.isdir('logs'):
            os.mkdir('logs')
        torch.save(state, os.path.join('logs', dataname ) + ".pth")
        visualizeResults(args.dataset, model, args.arch , args.size, args.lines, valset)

###### Saving time and numbers in respective file:
save_files(dataname, tt, best_epoch, max_iou, max_dice, max_prec, val_losses, val_iou, val_dice, val_prec)
plots('results/'+ dataname, val_losses, train_losses, val_iou, train_iou, val_dice, train_dice, val_prec, train_prec)