import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
import torch.amp

import numpy as np
import random
import time
import os
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import logging
import csv

import config as cfg
from loader import TinyViratCustom, collate_filter_none
from model import generate_model, set_parameter_requires_grad, get_optimizer_param_groups, make_data_parallel
from utils import AverageMeter, calculate_precision_recall_f1, get_lr, calculate_map, find_latest_checkpoint

log_format = '%(asctime)s,%(levelname)s,%(message)s'
csv_header = ['epoch', 'split', 'loss', 'accuracy', 'f1', 'precision', 'recall', 'map', 'lr']

log_dir = os.path.dirname(cfg.LOG_FILE)
try:
    os.makedirs(log_dir, exist_ok=True)
except Exception as e:
    print(f"!!! ERROR: Could not create log directory {log_dir}. Check permissions. Error: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO,
                    format=log_format,
                    filename=cfg.LOG_FILE,
                    filemode='a')

print(f"Text logs configured to be saved to: {cfg.LOG_FILE}")

logging.info(f"Starting Experiment: {cfg.EXPERIMENT_NAME}")
logging.info(f"Configuration: DEVICE={cfg.DEVICE}, EPOCHS={cfg.EPOCHS}, BATCH_SIZE={cfg.BATCH_SIZE}, LR={cfg.LEARNING_RATE_UNFROZEN}")

def set_random_seeds(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def calculate_class_weights(dataset):
    logging.info("Calculating class weights...")

    if not hasattr(dataset, 'class_name_to_id') or not dataset.class_name_to_id:
        logging.error("Dataset class map is missing or empty, cannot calculate weights.")
        raise ValueError("Dataset class map is missing or empty, cannot calculate weights.")
    elif len(dataset.class_name_to_id) != cfg.NUM_CLASSES:
         logging.warning(f"Dataset's 'class_name_to_id' map has {len(dataset.class_name_to_id)} classes, but config.NUM_CLASSES is {cfg.NUM_CLASSES}.")

    num_samples = len(dataset)
    class_counts = torch.zeros(cfg.NUM_CLASSES, dtype=torch.float64)
    skipped_count = 0
    valid_samples_count = 0

    for i, item in enumerate(tqdm(dataset, desc="Counting classes")):
        if item is None:
            skipped_count += 1
            continue

        _, target = item
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)

        class_counts += target.float()
        valid_samples_count += 1

    class_counts_clamped = class_counts.clamp(min=1)

    beta = 0.999
    effective_num = 1.0 - torch.pow(beta, class_counts_clamped)
    weights = (1.0 - beta) / effective_num

    weights = weights / torch.sum(weights) * cfg.NUM_CLASSES

    if skipped_count > 0:
        logging.warning(f"Skipped {skipped_count} samples during class weight calculation due to loading errors.")
    logging.info(f"Final Raw Class Counts (from {valid_samples_count} valid samples): {class_counts.long().tolist()}")
    logging.info(f"Calculated class weights: {weights.tolist()}")
    return weights.float().to(cfg.DEVICE)

def train_epoch(epoch, data_loader, model, criterion, optimizer, scheduler, scaler, device, tb_writer):
    model.train()

    current_lrs_in_epoch = [grp['lr'] for grp in optimizer.param_groups]
    logging.debug(f"Epoch {epoch} [Train] - LRs at start of train_epoch: {current_lrs_in_epoch}")

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    f1s = AverageMeter()
    accuracies = AverageMeter()

    start_time = time.time()
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for i, batch_data in enumerate(progress_bar):
        if batch_data is None:
            logging.warning(f"Skipping None batch {i}")
            continue
        inputs, targets = batch_data
        data_time.update(time.time() - start_time)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=cfg.USE_AMP):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds_proba = torch.sigmoid(outputs).cpu()
            preds_binary = (preds_proba > 0.5).float()
            targets_cpu = targets.cpu()

            precision, recall, f1 = calculate_precision_recall_f1(preds_proba, targets_cpu, cfg.NUM_CLASSES, threshold=0.5)
            acc = (preds_binary == targets_cpu).float().mean().item()

        batch_size = inputs.size(0)
        losses.update(loss.item(), batch_size)
        precisions.update(precision, batch_size)
        recalls.update(recall, batch_size)
        f1s.update(f1, batch_size)
        accuracies.update(acc, batch_size)

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        current_lr_for_bar = get_lr(optimizer)
        progress_bar.set_postfix(loss=losses.avg, f1=f1s.avg, acc=accuracies.avg, lr=f"{current_lr_for_bar:.8f}")

    progress_bar.close()
    logging.info(f"Epoch {epoch} [Train] Avg Loss: {losses.avg:.4f}, Avg F1: {f1s.avg:.4f}, Avg Acc: {accuracies.avg:.4f}")

    try:
        with open(cfg.CSV_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                'train',
                f"{losses.avg:.6f}",
                f"{accuracies.avg:.6f}",
                f"{f1s.avg:.6f}",
                f"{precisions.avg:.6f}",
                f"{recalls.avg:.6f}",
                '',
                f"{get_lr(optimizer):.8f}"
            ])
    except Exception as e:
        logging.error(f"Error writing training stats to CSV: {e}")

    if tb_writer:
        tb_writer.add_scalar('train_epoch/loss', losses.avg, epoch)
        tb_writer.add_scalar('train_epoch/precision', precisions.avg, epoch)
        tb_writer.add_scalar('train_epoch/recall', recalls.avg, epoch)
        tb_writer.add_scalar('train_epoch/f1', f1s.avg, epoch)
        tb_writer.add_scalar('train_epoch/accuracy', accuracies.avg, epoch)
        tb_writer.add_scalar('train_epoch/lr', get_lr(optimizer), epoch)

    return {'loss': losses.avg, 'f1': f1s.avg, 'acc': accuracies.avg, 'precision': precisions.avg, 'recall': recalls.avg}

def val_epoch(epoch, data_loader, model, criterion, device, tb_writer):
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    f1s = AverageMeter()
    accuracies = AverageMeter()

    start_time = time.time()
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} [Val]", leave=False)

    all_outputs_proba = []
    all_targets = []

    with torch.no_grad():
        for i, batch_data in enumerate(progress_bar):
            if batch_data is None:
                continue
            clips, targets = batch_data

            clips = clips.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            batch_size = targets.size(0)
            if batch_size != 1:
                logging.warning(f"Validation batch size is {batch_size} at index {i}, expected 1 due to multi-clip handling. Skipping batch.")
                continue

            clips = clips.squeeze(0)

            with torch.amp.autocast(device_type=device.type, enabled=cfg.USE_AMP):
                clip_outputs = model(clips)
                video_output = torch.mean(clip_outputs, dim=0, keepdim=True)
                loss = criterion(video_output, targets)

            preds_proba = torch.sigmoid(video_output).cpu()
            targets_cpu = targets.cpu()

            all_outputs_proba.append(preds_proba)
            all_targets.append(targets_cpu)

            preds_binary = (preds_proba > 0.5).float()
            precision, recall, f1 = calculate_precision_recall_f1(preds_proba, targets_cpu, cfg.NUM_CLASSES, threshold=0.5)
            acc = (preds_binary == targets_cpu).float().mean().item()

            losses.update(loss.item(), 1)
            precisions.update(precision, 1)
            recalls.update(recall, 1)
            f1s.update(f1, 1)
            accuracies.update(acc, 1)

            batch_time.update(time.time() - start_time)
            start_time = time.time()

            progress_bar.set_postfix(loss=losses.avg, f1=f1s.avg, acc=accuracies.avg)

    progress_bar.close()

    if not all_outputs_proba or not all_targets:
        logging.error(f"Epoch {epoch} [Val] No valid batches processed.")
        return {'loss': 0, 'f1': 0, 'acc': 0, 'precision': 0, 'recall': 0, 'map': 0, 'metric_val': 0}

    all_outputs_proba_tensor = torch.cat(all_outputs_proba, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)

    epoch_map = calculate_map(all_outputs_proba_tensor, all_targets_tensor, cfg.NUM_CLASSES)

    if isinstance(epoch, int):
        logging.info(f"Epoch {epoch} [Val] Avg Loss: {losses.avg:.4f}, Avg F1: {f1s.avg:.4f}, Avg Acc: {accuracies.avg:.4f}, mAP: {epoch_map:.4f}")
    else:
        logging.info(f"Epoch '{epoch}' [Eval] Avg Loss: {losses.avg:.4f}, Avg F1: {f1s.avg:.4f}, Avg Acc: {accuracies.avg:.4f}, mAP: {epoch_map:.4f}")

    try:
        with open(cfg.CSV_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                'val' if isinstance(epoch, int) else 'test',
                f"{losses.avg:.6f}",
                f"{accuracies.avg:.6f}",
                f"{f1s.avg:.6f}",
                f"{precisions.avg:.6f}",
                f"{recalls.avg:.6f}",
                f"{epoch_map:.6f}",
                ''
            ])
    except Exception as e:
        logging.error(f"Error writing validation/test stats to CSV: {e}")

    if tb_writer:
        if isinstance(epoch, int):
            tb_writer.add_scalar('val_epoch/loss', losses.avg, epoch)
            tb_writer.add_scalar('val_epoch/precision', precisions.avg, epoch)
            tb_writer.add_scalar('val_epoch/recall', recalls.avg, epoch)
            tb_writer.add_scalar('val_epoch/f1', f1s.avg, epoch)
            tb_writer.add_scalar('val_epoch/accuracy', accuracies.avg, epoch)
            tb_writer.add_scalar('val_epoch/mAP', epoch_map, epoch)

    if isinstance(epoch, int):
         logging.info(f"EPOCH_SUMMARY,{epoch+1},{cfg.EPOCHS},{losses.avg:.4f},{accuracies.avg:.4f},{f1s.avg:.4f},{epoch_map:.4f}")
    else:
         logging.info(f"TEST_SUMMARY,{losses.avg:.4f},{accuracies.avg:.4f},{f1s.avg:.4f},{epoch_map:.4f}")

    if isinstance(epoch, int):
        if cfg.EARLY_STOPPING_METRIC == 'val_loss':
            metric_val = losses.avg
        elif cfg.EARLY_STOPPING_METRIC == 'val_accuracy':
            metric_val = accuracies.avg
        elif cfg.EARLY_STOPPING_METRIC == 'val_f1':
            metric_val = f1s.avg
        elif cfg.EARLY_STOPPING_METRIC == 'val_map':
            metric_val = epoch_map
        else:
            logging.warning(f"Unrecognized EARLY_STOPPING_METRIC '{cfg.EARLY_STOPPING_METRIC}'. Defaulting to 'val_map'.")
            metric_val = epoch_map
    else:
        metric_val = None

    return {'loss': losses.avg,
            'f1': f1s.avg,
            'acc': accuracies.avg,
            'precision': precisions.avg,
            'recall': recalls.avg,
            'map': epoch_map,
            'metric_val': metric_val}

def main():
    set_random_seeds(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    output_path = Path(cfg.OUTPUT_DIR)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.critical(f"Could not create OUTPUT_DIR: {output_path}. Error: {e}")
        sys.exit(1)
    logging.info(f"Output directory: {output_path}")
    logging.info(f"Using device: {device}")

    try:
        csv_exists = os.path.exists(cfg.CSV_LOG_FILE)
        with open(cfg.CSV_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(csv_header)
                logging.info(f"Created and wrote header to CSV log: {cfg.CSV_LOG_FILE}")
    except Exception as e:
        logging.error(f"Could not initialize CSV log file {cfg.CSV_LOG_FILE}. Error: {e}")

    class_weights = None
    if cfg.USE_CLASS_WEIGHTS:
        if os.path.exists(cfg.CLASS_WEIGHTS_CACHE_FILE):
            try:
                class_weights = torch.load(cfg.CLASS_WEIGHTS_CACHE_FILE)
                class_weights = class_weights.to(device)
                logging.info(f"Loaded class weights from cache: {cfg.CLASS_WEIGHTS_CACHE_FILE}")
                if len(class_weights) != cfg.NUM_CLASSES:
                     logging.warning(f"Cached weights length ({len(class_weights)}) doesn't match NUM_CLASSES ({cfg.NUM_CLASSES}). Recalculating.")
                     class_weights = None
            except Exception as e:
                logging.error(f"Error loading class weights from cache: {e}. Recalculating.")
                class_weights = None

        if class_weights is None:
            logging.info("Loading full training dataset to calculate class weights...")
            full_train_dataset_for_weights = TinyViratCustom(data_split='train')
            if len(full_train_dataset_for_weights) == 0:
                logging.error("Training dataset for weights is empty!")
                sys.exit(1)
            class_weights = calculate_class_weights(full_train_dataset_for_weights)
            del full_train_dataset_for_weights

            try:
                torch.save(class_weights.cpu(), cfg.CLASS_WEIGHTS_CACHE_FILE)
                logging.info(f"Saved calculated class weights to cache: {cfg.CLASS_WEIGHTS_CACHE_FILE}")
            except Exception as e:
                logging.error(f"Error saving class weights to cache: {e}")
    else:
        logging.info("Class weights are disabled (USE_CLASS_WEIGHTS=False).")

    logging.info("Loading datasets for training/validation...")
    train_dataset = TinyViratCustom(data_split='train')
    val_dataset = TinyViratCustom(data_split='val')

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.BATCH_SIZE,
                              shuffle=True,
                              num_workers=cfg.NUM_WORKERS,
                              pin_memory=True,
                              worker_init_fn=set_random_seeds,
                              collate_fn=collate_filter_none)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.VAL_BATCH_SIZE,
                            shuffle=False,
                            num_workers=cfg.NUM_WORKERS,
                            pin_memory=True,
                            worker_init_fn=set_random_seeds,
                            collate_fn=collate_filter_none)
    logging.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    model = generate_model()
    model = make_data_parallel(model, cfg.IS_DISTRIBUTED, device)
    logging.info(f"Model {cfg.MODEL_NAME} initialized.")

    start_epoch = 0
    best_metric = -np.inf if cfg.EARLY_STOPPING_MODE == 'max' else np.inf
    early_stopping_counter = 0
    checkpoint_path = None
    loaded_optimizer_state = None

    best_model_path = output_path / 'best_model.pth'
    if best_model_path.exists():
        checkpoint_path = best_model_path
        logging.info(f"Found best model checkpoint: {checkpoint_path}")
    else:
        latest_ckpt = find_latest_checkpoint(output_path)
        if latest_ckpt:
            checkpoint_path = latest_ckpt
            logging.info(f"Found latest epoch checkpoint: {latest_ckpt}")
        else:
            logging.info("No checkpoint found. Starting training from scratch.")

    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            state_dict = checkpoint['state_dict']
            if not cfg.IS_DISTRIBUTED and all(key.startswith('module.') for key in state_dict):
                 state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}
            elif cfg.IS_DISTRIBUTED and not all(key.startswith('module.') for key in state_dict):
                 state_dict = {'module.' + k: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            logging.info("Successfully loaded model state_dict from checkpoint.")

            start_epoch = checkpoint.get('epoch', 0) + 1
            best_metric = checkpoint.get('best_metric', best_metric)
            early_stopping_counter = checkpoint.get('early_stopping_counter', 0)

            if 'optimizer' in checkpoint:
                loaded_optimizer_state = checkpoint['optimizer']
                logging.info("Stored optimizer state_dict from checkpoint.")
            else:
                logging.warning("Optimizer state not found in checkpoint.")

            if 'scheduler' in checkpoint:
                logging.warning("Scheduler state found in checkpoint, but SKIPPING loading due to potential scheduler type change.")
            else:
                logging.info("Scheduler state not found in checkpoint. Initializing scheduler from scratch.")

            logging.info(f"Resuming training from epoch {start_epoch}, Best Metric: {best_metric:.4f}, ES Counter: {early_stopping_counter}")

        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
            best_metric = -np.inf if cfg.EARLY_STOPPING_MODE == 'max' else np.inf
            early_stopping_counter = 0
            model = generate_model()
            model = make_data_parallel(model, cfg.IS_DISTRIBUTED, device)
            loaded_optimizer_state = None

    initial_backbone_frozen = (start_epoch < cfg.FROZEN_EPOCHS)
    set_parameter_requires_grad(model, backbone_frozen=initial_backbone_frozen)
    logging.info(f"Set initial requires_grad based on start_epoch={start_epoch}: Backbone frozen = {initial_backbone_frozen}")

    param_groups = get_optimizer_param_groups(model)

    for group in param_groups:
        if group['name'] == 'classifier':
            group['lr'] = cfg.LEARNING_RATE_FROZEN
        elif group['name'] == 'backbone':
            group['lr'] = 0.0

    if cfg.OPTIMIZER == 'AdamW':
        optimizer = optim.AdamW(param_groups, weight_decay=cfg.WEIGHT_DECAY)
        logging.info(f"Using optimizer: AdamW (Weight Decay: {cfg.WEIGHT_DECAY})")
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.OPTIMIZER}")

    if loaded_optimizer_state:
        try:
            optimizer.load_state_dict(loaded_optimizer_state)
            logging.info("Successfully loaded stored optimizer state_dict.")
        except Exception as e:
            logging.error(f"Error loading stored optimizer state: {e}. Optimizer may be reset.")

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)
    logging.info(f"Using loss: {cfg.LOSS_FUNCTION} with class weights: {'Enabled' if class_weights is not None else 'Disabled'}")

    scaler = torch.amp.GradScaler(enabled=cfg.USE_AMP)
    logging.info(f"Using Automatic Mixed Precision (AMP): {cfg.USE_AMP}")
    tb_writer = SummaryWriter(log_dir=output_path / 'tb_logs')
    logging.info(f"Logging to: {cfg.LOG_FILE}, {output_path / 'tb_logs'}")

    logging.info(f"Starting training from epoch {start_epoch} to {cfg.EPOCHS-1}")
    for epoch in range(start_epoch, cfg.EPOCHS):
        is_frozen_phase = epoch < cfg.FROZEN_EPOCHS
        if is_frozen_phase:
            for group in optimizer.param_groups:
                if group['name'] == 'backbone':
                    group['lr'] = 0.0
                    for p in group['params']: p.requires_grad = False
                elif group['name'] == 'classifier':
                    group['lr'] = 1e-3
                    for p in group['params']: p.requires_grad = True
            logging.info(f"Epoch {epoch} [Frozen Phase] - LRs set (Backbone: 0.0, Classifier: 1e-3)")
        else:
            if epoch == cfg.FROZEN_EPOCHS:
                logging.info(f"Epoch {epoch}: Unfreezing backbone...")
                for group in optimizer.param_groups:
                    if group['name'] == 'backbone':
                        for p in group['params']: p.requires_grad = True

            total_decay_epochs = cfg.EPOCHS - cfg.FROZEN_EPOCHS
            current_decay_epoch = epoch - cfg.FROZEN_EPOCHS
            denominator = max(1, total_decay_epochs - 1) if total_decay_epochs > 1 else 1
            decay_progress = current_decay_epoch / denominator
            classifier_lr = 1e-3 * (1 - decay_progress) + 1e-5 * decay_progress
            backbone_lr = classifier_lr * 0.1

            for group in optimizer.param_groups:
                if group['name'] == 'backbone':
                    group['lr'] = backbone_lr
                    for p in group['params']: p.requires_grad = True
                elif group['name'] == 'classifier':
                    group['lr'] = classifier_lr
                    for p in group['params']: p.requires_grad = True
            logging.info(f"Epoch {epoch} [Unfrozen Phase] - LRs set (Backbone: {backbone_lr:.6g}, Classifier: {classifier_lr:.6g})")

        current_lrs_before_epoch = [grp['lr'] for grp in optimizer.param_groups]
        logging.info(f"Epoch {epoch} [{ 'Frozen' if is_frozen_phase else 'Unfrozen' }] - LRs before calling train_epoch: {current_lrs_before_epoch}")

        try:
            train_stats = train_epoch(epoch, train_loader, model, criterion, optimizer, None, scaler, device, tb_writer)
        except Exception as e:
            logging.error(f"Exception during train_epoch at epoch {epoch}: {e}")
            break

        try:
            val_stats = val_epoch(epoch, val_loader, model, criterion, device, tb_writer)
        except Exception as e:
            logging.error(f"Exception during val_epoch at epoch {epoch}: {e}")
            break

        current_metric = val_stats['metric_val']

        is_best = (current_metric > best_metric) if cfg.EARLY_STOPPING_MODE == 'max' else (current_metric < best_metric)

        save_data = {
            'epoch': epoch,
            'arch': cfg.MODEL_NAME,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': None,
            'best_metric': best_metric,
            'early_stopping_counter': early_stopping_counter,
            'lrs': [group['lr'] for group in optimizer.param_groups]
        }

        if cfg.CHECKPOINT_FREQ > 0 and (epoch + 1) % cfg.CHECKPOINT_FREQ == 0:
            ckpt_path = output_path / f'ckpt_epoch_{epoch}.pth'
            try:
                torch.save(save_data, ckpt_path)
                logging.info(f"Saved periodic checkpoint to: {ckpt_path}")
            except Exception as e:
                 logging.error(f"Error saving periodic checkpoint {ckpt_path}: {e}")

        if is_best:
            logging.info(f"Validation metric improved ({best_metric:.4f} -> {current_metric:.4f})!")
            best_metric = current_metric
            early_stopping_counter = 0
            save_data['best_metric'] = best_metric
            best_ckpt_path = output_path / 'best_model.pth'
            try:
                torch.save(save_data, best_ckpt_path)
                logging.info(f"Saved best model checkpoint to: {best_ckpt_path}")
            except Exception as e:
                 logging.error(f"Error saving best model checkpoint {best_ckpt_path}: {e}")
        else:
            early_stopping_counter += 1
            logging.info(f"Validation metric did not improve. Counter: {early_stopping_counter}/{cfg.EARLY_STOPPING_PATIENCE}")

        if early_stopping_counter >= cfg.EARLY_STOPPING_PATIENCE:
            logging.info(f"Early stopping triggered after {cfg.EARLY_STOPPING_PATIENCE} epochs without improvement.")
            break

    logging.info("Training finished.")

    logging.info("--- Final Data Loading Error Report ---")
    if train_dataset and hasattr(train_dataset, 'report_load_errors'):
        logging.info("Train Dataset Errors:")
        train_dataset.report_load_errors()
    if val_dataset and hasattr(val_dataset, 'report_load_errors'):
        logging.info("Validation Dataset Errors:")
        val_dataset.report_load_errors()
    logging.info("---------------------------------------")

    if tb_writer:
        tb_writer.close()

    test_dataset = TinyViratCustom(data_split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=set_random_seeds,
        collate_fn=collate_filter_none
    )
    logging.info(f"Test samples: {len(test_dataset)}")

    best_model_path = output_path / 'best_model.pth'
    if best_model_path.exists():
        logging.info(f"Loading best model from {best_model_path} for test evaluation.")
        checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict']
        if not cfg.IS_DISTRIBUTED and all(key.startswith('module.') for key in state_dict):
            state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}
        elif cfg.IS_DISTRIBUTED and not all(key.startswith('module.') for key in state_dict):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        logging.warning("Best model checkpoint not found. Using last model state for test evaluation.")

    logging.info("Running test set evaluation...")
    test_stats = val_epoch("test", test_loader, model, criterion, device, tb_writer)
    logging.info(f"Test set results: {test_stats}")

    logging.shutdown()

if __name__ == '__main__':
    if cfg.DATA_FOLDER == '/path/to/your/TinyVirat-v2/dataset' or not os.path.exists(cfg.DATA_FOLDER):
        print("=" * 50)
        print("!!! ERROR: Please set a valid DATA_FOLDER path in config.py !!!")
        print("=" * 50)
        logging.error("Please set a valid DATA_FOLDER path in config.py")
        sys.exit(1)

    main()