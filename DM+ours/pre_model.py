import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5,1,4"

import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import datasets, transforms
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import torchvision
import wandb
from utils import get_premodel
def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ResNet18', help='model')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--num_premodels', type=int, default=1000, help='path to save results')
    parser.add_argument('--pre_epochs', type=int, default=40, help='path to save results')
    # To decrease train time, we can set smaller "pre_epochs" and num_premodels""
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    #args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    args.dsa = True
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    num_premodels=args.num_premodels
    pre_epochs=args.pre_epochs
    for i in range(0,num_premodels):
        print(i)
        net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model

        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr_net, momentum=0.9, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss().to(args.device)
        trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=6)
        for ep in range(pre_epochs):
            loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug = False)
            loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug = False)
            print("simple train net, test acc: "+str(acc_test))
        #save path
        save_path="./pre_dict_res18_40"
        if not os.path.exists(save_path):
                os.mkdir(save_path)
        torch.save(net.state_dict(), save_path+"/buffer{}.pth".format(i))
        
if __name__ == '__main__':
    main()
