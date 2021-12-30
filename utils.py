import typing
import torch
import numpy as np
import cv2

def calc_iou(pred, gt, thres=0.5) -> float:
    probs = torch.sigmoid(pred)
    probs = torch.nn.functional.threshold(probs, thres, 0.)
    probs = probs.to(torch.bool)

    gt = gt.to(torch.bool)

    n, c, h, w = pred.shape

    intersection = torch.logical_and(pred.view(n,h,w), gt.view(n,h,w))
    union = torch.logical_or(pred.view(n,h,w), gt.view(n,h,w))

    intersection_sum = (torch.sum(torch.sum(intersection, dim=1), dim=1)).to(torch.float)
    union_sum = (torch.sum(torch.sum(union, dim=1), dim=1)).to(torch.float)
    iou_score = torch.div(intersection_sum, union_sum)

    return torch.mean(iou_score)

def calc_acc(logits, gt, thres=0.5) -> float:
    probs = torch.sigmoid(logits)
    probs = torch.nn.functional.threshold(probs, thres, 0.)
    probs = probs.to(torch.bool)

    gt = gt.to(torch.bool)
    
    n, c, h, w = logits.shape

    not_pred = torch.logical_not(probs.view(n,h,w))
    not_gt = torch.logical_not(gt.view(n,h,w))

    tp = torch.logical_and(probs.view(n,h,w), gt.view(n,h,w))
    tn = torch.logical_and(not_pred, not_gt)
    fp = torch.logical_and(probs.view(n,h,w), not_gt)
    fn = torch.logical_and(not_pred, gt.view(n,h,w))

    tp_sum = (torch.sum(torch.sum(tp, dim=1), dim=1)).to(torch.float)
    tn_sum = (torch.sum(torch.sum(tn, dim=1), dim=1)).to(torch.float)

    fp_sum = (torch.sum(torch.sum(fp, dim=1), dim=1)).to(torch.float)
    fn_sum = (torch.sum(torch.sum(fn, dim=1), dim=1)).to(torch.float)

    acc_score = torch.div(tp_sum+tn_sum, tp_sum+tn_sum+fp_sum+fn_sum)

    return torch.mean(acc_score)



