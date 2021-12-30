import os, sys
import argparse
import cv2
import numpy as np
import json
import logging
import copy
from datetime import datetime

import torch
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms
import torch.optim as optim
from tensorboardX import SummaryWriter
import unet
from dataset import CarotidSet

def train(args, model, criterion, optimizer, dataloader, ep, writer):
    model.train()
    running_loss = 0
    
    for idx, data in enumerate(dataloader):
        train_sample, ground_truth = data
        # train_sample = train_sample.cuda()
        # ground_truth = ground_truth.cuda()
        logits = model(train_sample)
        loss = criterion(logits, ground_truth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (idx + 1) % args.print_every == 0:
            running_loss /= len(dataloader)
            print(f'[Epoch{ep}/{args.max_epoch}] [{idx}/{len(dataloader)}] Loss: {running_loss: .4ef}')
            writer.add_scalar('train/loss', running_loss, ep*len(dataloader)+idx)
            running_loss = 0.0
    
def test(args, model, testloader, ep, writer):
    model.eval()

    acc = 0.
    running_acc_li = 0.
    running_acc_ma = 0.

    with torch.no_grad():
        for idx, (data) in enumerate(testloader):
            test_sample, gt_li, gt_ma = data

            logits = model(test_sample)

            running_iou_li += calc_iou(logits[: , 0:1, :, :], gt_li, thres=0.5)
            running_iou_ma += calc_iou(logits[: , 1:2, :, :], gt_ma, thres=0.5)
            # todo: implementt calc_iou, calc_acc
            running_acc_li += calc_acc(logits[: , 0:1, :, :], gt_li, thres=0.5)
            running_acc_ma += calc_acc(logits[: , 1:2, :, :], gt_ma, thres=0.5)
    
    epoch_iou_li = running_iou_li / len(testloader)
    epoch_iou_ma = running_iou_ma / len(testloader)

    epoch_acc_li = running_acc_li / len(testloader)
    epoch_acc_ma = running_acc_ma / len(testloader)

    print("{0} - [{1}-Epoch-{2}] Iou(LI): {3:.2f}".format(
        datetime.now(), 'Validation', ep, epoch_iou_li.item()*100.0))
    
    print("{0} - [{1}-Epoch-{2}] Iou(MA): {3:.2f}".format(
        datetime.now(), 'Validation', ep, epoch_iou_ma.item()*100.0))
    
    print('='*80)

    writer.add_scalar('IoU/val/li', epoch_iou_li.item()*100.0, ep*len(testloader))
    writer.add_scalar('IoU/val/ma', epoch_iou_ma.item()*100.0, ep*len(testloader))
    writer.add_scalar('ACC/val/li', epoch_acc_li.item()*100.0, ep*len(testloader))
    writer.add_scalar('ACC/val/ma', epoch_acc_ma.item()*100.0, ep*len(testloader))

    return (epoch_iou_li.item() + epoch_iou_ma.item())/2


def opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, default='small_dataset', help='path to dataset')

    parser.add_argument('--json_train', type=str, default='gTruth_pp_small.json', help='path to train dataset')

    parser.add_argument('--max_epoch', type=int, default=200, help = 'max train epoch')

    parser.add_argument('--optimizer', type=str, default='sgd', help = 'optimizer')
    parser.add_argument('--batch_size', type=int, default=2048, help = 'batch size')

    parser.add_argument('--num_workers', type=int, default=16, help = 'the number of workers')
    parser.add_argument('--log_dir', type=str, default='./train_log_modular')

    parser.add_argument('--lr_drop', type=int, default=180)

    return parser.parse_args()

def main():
    args = opt()
    model = unet.CCANet(3, 2)
    criterion = nn.BCEWithLogitsLoss()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=1e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    else:
        raise ValueError

    carotid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )


    # train_set = CarotidSet(args.root, args.json_train, transform=carotid_transform, flip=False, rotation=False, translation=False)
    # next time
    # trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    trainloader = None
    testloader = None

    writer = SummaryWriter(args.log_dir)
    best_acc = 0.0
    for ep in range(args.max_epoch):

        train(args, model, criterion, optimizer, trainloader, ep, writer)

        if (ep+1) % args.test_every == 0:
            acc = test(args, model, testloader, ep, writer)
            if best_acc < acc:
                best_acc = acc
                best_model = copy.deepcopy(model.state_dict())

if __name__ == '__main__':
    main()