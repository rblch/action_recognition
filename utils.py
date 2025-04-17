import csv
import random
import logging
import os
import re
from pathlib import Path
from functools import partialmethod

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, multilabel_confusion_matrix


class AverageMeter(object):
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


class Logger(object):
    def __init__(self, path, header):
        self.log_file = path.open('w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


def calculate_precision_and_recall(outputs, targets, pos_label=1):
    with torch.no_grad():
        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        precision, recall, _, _ = precision_recall_fscore_support(
            targets.view(-1, 1).cpu().numpy(),
            pred.cpu().numpy())

        return precision[pos_label], recall[pos_label]


def calculate_precision_recall_f1(output, targets, num_classes, threshold=0.5):
    with torch.no_grad():
        output = output.reshape(-1, num_classes)
        targets = targets.reshape(-1, num_classes)
        output = (output.cpu().numpy() > threshold).astype(int)
        targets = (targets.cpu().numpy() > threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, output, average='samples', zero_division=0
        )
    
        return precision, recall, f1


def calculate_map(y_pred_proba, y_true, num_classes):
    if y_pred_proba.numel() == 0 or y_true.numel() == 0:
        logging.warning("Attempted to calculate mAP on empty predictions or targets.")
        return 0.0
    if y_pred_proba.shape[0] != y_true.shape[0] or y_pred_proba.shape[1] != num_classes or y_true.shape[1] != num_classes:
         logging.error(f"Shape mismatch for mAP calculation: preds {y_pred_proba.shape}, targets {y_true.shape}, num_classes {num_classes}")
         return 0.0

    y_true_np = y_true.cpu().numpy()
    y_pred_proba_np = y_pred_proba.cpu().numpy()

    ap_scores = []
    for i in range(num_classes):
        if np.sum(y_true_np[:, i]) > 0:
            ap = average_precision_score(y_true_np[:, i], y_pred_proba_np[:, i])
            ap_scores.append(ap)

    if not ap_scores:
        logging.warning("mAP calculation: No classes with positive samples found.")
        return 0.0

    mAP = np.mean(ap_scores)
    return mAP


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()

    random.seed(torch_seed + worker_id)

    if torch_seed >= 2**32:
        torch_seed = torch_seed % 2**32
    np.random.seed(torch_seed + worker_id)


def get_lr(optimizer):
    max_lr = 0.0
    for param_group in optimizer.param_groups:
        max_lr = max(max_lr, param_group['lr'])
    return max_lr


def partialclass(cls, *args, **kwargs):
    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return PartialClass


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    max_epoch = -1
    latest_ckpt_path = None
    epoch_pattern = re.compile(r'ckpt_epoch_(\d+)\.pth')

    if not checkpoint_dir.is_dir():
        return None

    for f in checkpoint_dir.glob('ckpt_epoch_*.pth'):
        match = epoch_pattern.match(f.name)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_ckpt_path = f

    return latest_ckpt_path