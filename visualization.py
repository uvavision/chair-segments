from PIL import Image
import numpy as np
import IPython.display as fig
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import torch
import IPython.display as fig
from io import BytesIO

plt.rcParams.update({'figure.max_open_warning': 0})
colors = ['bo-', 'ro-', 'go-', 'co-', 'mo-']

def show_grid(masks, nrows, passed):
    grid = make_grid(masks, nrow = nrows)
    pil_image = Image.fromarray(np.transpose(grid.mul(255).byte().numpy(), (1, 2, 0)))
    bio = BytesIO(); pil_image.save(bio, format = 'png')
    fig.display(fig.Image(bio.getvalue()))
    pil_image.save(passed + '.png')
    #fig.savefig(fig.Image(bio.getvalue()))

def plots(dataname, val_losses, train_losses, val_iou, train_iou, val_dice, train_dice, val_prec, train_prec):

    plt.figure(figsize = (20, 4))
    plt.subplot(1, 4, 1)
    plt.plot(val_losses, 'bo-', label = 'val-loss')
    plt.plot(train_losses, 'ro-', label = 'train-loss')
    plt.grid('on')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['validation', 'training'], loc='upper left')

    plt.subplot(1, 4, 2)
    plt.plot(val_iou, 'bo-', label = 'val-iou')
    plt.plot(train_iou, 'ro-', label = 'train-iou')
    plt.ylabel('Intersection Over Union')
    plt.grid('on')
    plt.xlabel('epoch')
    plt.legend(['validation', 'training'], loc='upper left')

    plt.subplot(1, 4, 3)
    plt.plot(val_dice, 'bo-', label = 'val-dice')
    plt.plot(train_dice, 'ro-', label = 'train-dice')
    plt.ylabel('Dice')
    plt.grid('on')
    plt.xlabel('epoch')
    plt.legend(['validation', 'training'], loc='upper right')
    plt.savefig( dataname + '_plots.png')
    plt.show()

    plt.subplot(1, 4, 4)
    plt.plot(val_prec, 'bo-', label = 'val-prec')
    plt.plot(train_prec, 'ro-', label = 'train-prec')
    plt.ylabel('Precision')
    plt.grid('on')
    plt.xlabel('epoch')
    plt.legend(['validation', 'training'], loc='upper right')
    plt.savefig( dataname + '_plots.png')
    plt.show()



def visualizeImages(dataset, dataname, size, lines, valset, trans):
    
    lst_val  =  [ x.rstrip() for x in open('ids/idlst_val.txt', "r")]
    lst_test =  [ x.rstrip() for x in open('ids/idlst_test.txt', "r")]

    for i in range(0, lines):

        lstval_images, lsttest_images   = [], []                    ##### Images            
        lstval_gt, lsttest_gt           = [], []                    ##### Ground Truth

        for j in range(0, size):

            index = i*size + j
            imageval,  labelval  = valset[int(lst_val[index])]      ##### val images
            lstval_images.append(imageval)
            lstval_gt.append(torch.cat((labelval, labelval, labelval), 0))

        ##### validation set
        show_grid(lstval_images,size, dataname + '_valimage_'   + str(i))
        show_grid(lstval_gt,    size, dataname + '_valgt_'      + str(i))



def visualizeResults(dataset, model, dataname, size, lines, valset):

    model.cuda()
    lst_val  =  [ x.rstrip() for x in open('ids/idlst_val.txt', "r")]
    lst_test =  [ x.rstrip() for x in open('ids/idlst_test.txt', "r")]

    for i in range(0, lines):

        lstval_pred = []           ##### Prediction by the model
        lsttest_pred = []

        for j in range(0, size):

            index = i*size + j
            ##### val images
            imageval,  labelval  = valset[int(lst_val[index])]
            imageval = imageval.cuda()
            #nvidiscores = model(imageval.unsqueeze(0))
            #preimage = torch.sigmoid(scores['out'].cpu()[0])
            
            if (dataname == 'unet' or dataname == 'fcnvgg16'):
                scores = model(imageval.unsqueeze(0)) # [N, 2, H, W]   
            elif (dataname == 'fcnresnet50' or dataname == 'fcnresnet101'):
                scores = model(imageval.unsqueeze(0))['out'] # [N, 2, H, W]
            else:
                scores = 0
            preimage = torch.sigmoid(scores.cpu()[0])
            lstval_pred.append(torch.cat((preimage, preimage, preimage), 0))
            
        ##### validation set
        show_grid(lstval_pred,  size,'results/'+ dataset + '064_' +  dataname + '_scratch_valpred_'    + str(i))
