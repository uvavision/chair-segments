import torch
from torchvision import transforms

from tqdm import tqdm as tqdm
import time, os


def save_files(nmodel, tt, best_epoch, max_iou, max_dice, max_prec, val_losses, val_iou, val_dice, val_prec):

    # Visualize results at this epoch.    
    h = int(tt/3600)
    m = int((tt - h*3600 )/60)
    s = int( tt - h*3600 -m*60)

    tmessage  = '\n' + nmodel + ' ' + str(best_epoch) + ' epochs took ' + str(h) + 'h ' + str(m) + 'min ' + str(s) + ' seg\n'
    mmessage = '\n' + nmodel + "\tmax_iou: " + str(max_iou) + "\tmax_dice: " + str(max_dice) + "\tmax_prec: " + str(max_prec) 
    
    #### save in the same Results/size032/Airplane032
    size_filename = 'results/' +  nmodel +'.txt'
    print('size_filename: ' + size_filename)

    ###### Saving numbers:

    file1 = open(size_filename,"a")
    file1.write(mmessage)
    file1.write(tmessage)
    file1.write('\nval_losses: ')
    [file1.write(str(x) + '\t') for x in val_losses]
    file1.write('\nval_iou: ')
    [file1.write(str(x) + '\t') for x in val_iou]
    file1.write('\nval_dice: ')
    [file1.write(str(x) + '\t') for x in val_dice]
    file1.write('\nval_prec: ')
    [file1.write(str(x) + '\t') for x in val_prec]
    file1.close()



def metrics(predicted, groundtruth):
    
    nPositive   = 0
    iou         = 0
    dice        = 0

    predicted[predicted >= 0.5] = 1.0
    predicted[predicted < 0.5] = 0
    groundtruth[groundtruth >= 0.5] = 1.0
    groundtruth[groundtruth < 0.5] = 0

    iflat = predicted.view(-1)
    tflat = groundtruth.view(-1)

    ###precision
    total   = (tflat == tflat).sum()
    correct = (iflat == tflat).sum()
    precision =  correct.item() / total.item()
    
    ###jaccard only the ones with mask
    for index in range(0, len(groundtruth)):

        if(groundtruth[index].sum()> 5):
            iflat = predicted[index].view(-1)
            tflat = groundtruth[index].view(-1)
    
            intersection = (iflat * tflat).sum()
            sumpred = iflat.sum()
            sumgt = tflat.sum()

            # iou = target n pred / target u pred 
            iou += (intersection/ (sumpred - intersection + sumgt)).item()
            
            #Dice
            dice += (2*intersection / (sumpred + sumgt )).item()
            nPositive +=1
    iou  = iou / nPositive
    dice = dice/ nPositive

    return iou, dice, precision, nPositive


def trainpass(namemodel, model, inputs, labels, optimizer, criterion):

    inputs = inputs.cuda()  # [N, 1, H, W]
    labels = labels.cuda()  # [N, H, W] with class indices (0, 1)
    optimizer.zero_grad()
    
    if (namemodel == 'unet' or namemodel == 'fcnvgg16'):
        scores = model(inputs) # [N, 2, H, W]   
    if (namemodel == 'fcnresnet50' or namemodel == 'fcnresnet101'):
        scores = model(inputs)['out'] # [N, 2, H, W]

    loss = criterion(torch.sigmoid(scores.cpu()), labels.cpu())
    cumloss = loss.item()
    
    loss.backward()
    optimizer.step()
    iou, dice, prec, nPositive = metrics(torch.sigmoid(scores.cpu()), labels.cpu())

    return cumloss, iou, dice, prec, nPositive


def trainRound(namemodel, model, dataset, trainLoader, optimizer, criterion, epoch):
    count, total, cumloss = 0, 0, 0
    totaliou, totaldice, totalprec, totalpos = 0, 0, 0, 0
    model.train()
    t = tqdm(trainLoader, desc = 'Train %d' % epoch)
    
    for (i, (inputs, labels)) in enumerate(t):
        loss_, iou, dice, prec, nPositive = trainpass(namemodel, model, inputs, labels, optimizer, criterion)
        total += inputs.data.size(0)
        totaliou  += iou
        totaldice += dice
        totalprec += prec
        totalpos  += nPositive
        count     += 1
        cumloss   += loss_
        t.set_postfix(loss = cumloss/count, iou = iou, dice = dice, prec = prec)

    return cumloss/count, totaliou/count, totaldice/count, totalprec/count

def valpass(namemodel, model, inputs, labels, criterion):
    
    inputs = inputs.cuda()
    labels = labels.cuda()

    # Compute predictions.
    if (namemodel == 'unet' or namemodel == 'fcnvgg16'):
        scores = model(inputs) # [N, 2, H, W]   
    if (namemodel == 'fcnresnet50' or namemodel == 'fcnresnet101'):
        scores = model(inputs)['out'] # [N, 2, H, W]   

    loss = criterion(torch.sigmoid(scores.cpu()), labels.cpu())
    cumloss = loss.item()

    #metrics
    iou, dice, prec, nPositive = metrics(torch.sigmoid(scores.cpu()), labels.cpu())
    return cumloss, iou, dice, prec, nPositive

def valRound(namemodel, model, dataset, valLoader, criterion, epoch):

    model.eval()
    
    t = tqdm(valLoader, desc = 'Val %d' % epoch)
    count, total, cumloss = 0, 0, 0
    totaliou, totaldice, totalprec, totalpos = 0, 0, 0, 0
    with torch.no_grad():

        for (i, (inputs, labels)) in enumerate(t):
            loss_, iou, dice, prec, nPositive = valpass(namemodel, model, inputs, labels, criterion)
            totaliou  += iou
            totaldice += dice
            totalprec += prec
            totalpos  += nPositive
            
            total   += inputs.data.size(0)    # Show variables in progress bar.
            count   += 1
            cumloss += loss_
            t.set_postfix(loss = cumloss/count, iou = iou, dice = dice, prec = prec)
    
    return cumloss/count, totaliou/count, totaldice/count, totalprec/count
