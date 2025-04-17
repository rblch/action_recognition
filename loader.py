# loader.py
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision.transforms.v2 as transforms
from decord import VideoReader, cpu, DECORDError
import numpy as np
import random
import time
from collections import Counter
import re
import logging

import config as cfg

def collate_filter_none(batch):
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        return None
    return default_collate(batch)

class FrameDropout:
    def __init__(self, p=0.05):
        assert 0.0 <= p <= 1.0
        self.p = p

    def __call__(self, vid_tensor):
        if self.p == 0.0:
            return vid_tensor

        if vid_tensor.shape[0] == 3 and len(vid_tensor.shape) == 4:
            t_dim = 1
        elif len(vid_tensor.shape) == 4:
            t_dim = 0
        else:
            return vid_tensor

        num_frames = vid_tensor.shape[t_dim]
        num_to_drop = int(np.floor(self.p * num_frames))

        if num_to_drop == 0:
            return vid_tensor

        indices_to_drop = random.sample(range(num_frames), k=num_to_drop)
        indices_to_keep = [i for i in range(num_frames) if i not in indices_to_drop]

        if not indices_to_keep:
            indices_to_keep = random.sample(range(num_frames), k=1)

        if t_dim == 1:
            vid_tensor_kept = vid_tensor.index_select(t_dim, torch.tensor(indices_to_keep, device=vid_tensor.device))
            num_missing = num_frames - len(indices_to_keep)
            if num_missing > 0:
                last_frame = vid_tensor_kept.index_select(t_dim, torch.tensor([-1], device=vid_tensor.device))
                padding = torch.repeat_interleave(last_frame, num_missing, dim=t_dim)
                vid_tensor = torch.cat((vid_tensor_kept, padding), dim=t_dim)
            else:
                vid_tensor = vid_tensor_kept

        else:
            vid_tensor_kept = vid_tensor.index_select(t_dim, torch.tensor(indices_to_keep, device=vid_tensor.device))
            num_missing = num_frames - len(indices_to_keep)
            if num_missing > 0:
                last_frame = vid_tensor_kept.index_select(t_dim, torch.tensor([-1], device=vid_tensor.device))
                padding = torch.repeat_interleave(last_frame, num_missing, dim=t_dim)
                vid_tensor = torch.cat((vid_tensor_kept, padding), dim=t_dim)
            else:
                vid_tensor = vid_tensor_kept

        return vid_tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

class TinyViratCustom(Dataset):
    def __init__(self, data_split):
        assert data_split in ['train', 'val', 'test'], "data_split must be 'train', 'val', or 'test'"
        self.data_split = data_split
        self.num_frames = cfg.NUM_FRAMES
        self.skip_frames = cfg.SKIP_FRAMES if data_split == 'train' else 1
        self.input_size = cfg.INPUT_SIZE

        annotation_filename = f"new_{data_split}.json"
        class_map_filename = "new_class_map.json"
        annotation_path = os.path.join(cfg.DATA_FOLDER, annotation_filename)
        class_map_path = os.path.join(cfg.DATA_FOLDER, class_map_filename)

        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        if not os.path.exists(class_map_path):
            raise FileNotFoundError(f"Class map file not found: {class_map_path}")

        raw_annotations = json.load(open(annotation_path, 'r'))
        self.class_map = json.load(open(class_map_path, 'r'))
        self.class_labels = [k for k, v in sorted(self.class_map.items(), key=lambda item: item[1])]
        self.num_classes = len(self.class_labels)
        assert self.num_classes == cfg.NUM_CLASSES, \
            f"Expected {cfg.NUM_CLASSES} classes based on config, found {self.num_classes} in {class_map_filename}"

        self.class_id_to_name = {int(k): v for k, v in self.class_map.items()}
        self.class_name_to_id = {v: k for k, v in self.class_id_to_name.items()}

        self.annotations = {}
        self.load_errors = []
        skipped_short = 0
        for annotation in raw_annotations:
            if data_split == 'train' and annotation['dim'][0] < self.num_frames * self.skip_frames:
                skipped_short += 1
                continue

            video_id = annotation['id']
            self.annotations[video_id] = {
                'path': annotation['path'],
                'label': annotation['label'],
                'length': annotation['dim'][0]
            }
        self.video_ids = list(self.annotations.keys())

        kinetics_mean = [0.485, 0.456, 0.406]
        kinetics_std = [0.229, 0.224, 0.225]

        train_transforms_list = [
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            transforms.ConvertImageDtype(torch.float32),
            FrameDropout(p=cfg.DROPOUT_PROB),
            transforms.Resize((self.input_size, self.input_size), antialias=True),
            transforms.RandomHorizontalFlip(p=cfg.FLIP_PROB),
            transforms.RandomGrayscale(p=cfg.GRAYSCALE_PROB),
        ]
        if cfg.COLOR_JITTER_ENABLE:
            train_transforms_list.append(
                transforms.ColorJitter(brightness=cfg.COLOR_JITTER_BRIGHTNESS,
                                       contrast=cfg.COLOR_JITTER_CONTRAST,
                                       saturation=cfg.COLOR_JITTER_SATURATION,
                                       hue=cfg.COLOR_JITTER_HUE)
            )
        train_transforms_list.extend([
            transforms.Normalize(mean=kinetics_mean, std=kinetics_std),
            transforms.Lambda(lambda x: x.permute(1, 0, 2, 3))
        ])
        self.train_transform = transforms.Compose(train_transforms_list)

        self.val_test_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((self.input_size, self.input_size), antialias=True),
            transforms.Normalize(mean=kinetics_mean, std=kinetics_std),
            transforms.Lambda(lambda x: x.permute(1, 0, 2, 3))
        ])

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        annotation = self.annotations[video_id]
        video_path = os.path.join(cfg.DATA_FOLDER, 'videos', annotation['path'])
        labels = annotation['label']

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            if total_frames == 0:
                return None

            if self.data_split == 'train':
                max_possible_start = max(0, total_frames - (self.num_frames * self.skip_frames))
                start_index = random.randint(0, max_possible_start)
                frame_indices = [start_index + i * self.skip_frames for i in range(self.num_frames)]
                frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]

                frames_np = vr.get_batch(frame_indices).asnumpy()
                del vr

                frames_tensor = torch.from_numpy(frames_np)

                clips_tensor = self.train_transform(frames_tensor)

            else:
                all_indices = list(range(total_frames))
                frames_np = vr.get_batch(all_indices).asnumpy()
                del vr

                num_clips = total_frames // self.num_frames
                clips_list = []
                if num_clips == 0 and total_frames > 0:
                    num_padding = self.num_frames - total_frames
                    last_frame = frames_np[-1:]
                    padding = np.repeat(last_frame, num_padding, axis=0)
                    frames_np = np.concatenate((frames_np, padding), axis=0)
                    num_clips = 1

                for i in range(num_clips):
                    start = i * self.num_frames
                    end = start + self.num_frames
                    clip_frames_np = frames_np[start:end]

                    clip_frames_tensor = torch.from_numpy(clip_frames_np)

                    clip_tensor_transformed = self.val_test_transform(clip_frames_tensor)
                    clips_list.append(clip_tensor_transformed)

                if not clips_list:
                    return None

                clips_tensor = torch.stack(clips_list, dim=0)

            label_tensor = torch.zeros(self.num_classes, dtype=torch.float32)

            if self.class_name_to_id:
                for name in labels:
                    if name in self.class_name_to_id:
                        class_id = self.class_name_to_id[name]
                        if 0 <= class_id < cfg.NUM_CLASSES:
                            label_tensor[class_id] = 1.0

            return clips_tensor, label_tensor

        except (DECORDError, FileNotFoundError, Exception):
            return None

    def report_load_errors(self):
        num_errors = len(self.load_errors)
        if num_errors == 0:
            return

        error_counts = Counter(err_type for _, err_type, _ in self.load_errors)

        for err_type, count in error_counts.items():
            pass

        if num_errors > 0:
            for i, (path, err_type, msg) in enumerate(self.load_errors[:5]):
                msg_short = msg
                if err_type == 'DECORDError':
                    match = re.search(r"Check failed:.*", msg)
                    if match:
                        msg_short = match.group(0)
                    else:
                        msg_short = (msg[:150] + '...') if len(msg) > 150 else msg
                else:
                    msg_short = (msg[:150] + '...') if len(msg) > 150 else msg

if __name__ == '__main__':
    if cfg.DATA_FOLDER == '/path/to/your/TinyVirat-v2/dataset':
        pass

    train_dataset = None
    try:
        train_dataset = TinyViratCustom(data_split='train')

        if len(train_dataset) == 0:
            pass
        else:
            train_dataloader = DataLoader(train_dataset,
                                          batch_size=cfg.BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=cfg.NUM_WORKERS,
                                          pin_memory=True,
                                          collate_fn=collate_filter_none)

            start_time = time.time()
            batch_count = 0
            for batch_data in train_dataloader:
                if batch_data is None:
                    continue
                clips, labels = batch_data
                batch_count += 1
                if batch_count >= 3:
                    break

    except FileNotFoundError as e:
        pass
    except Exception as e:
        pass
    finally:
        if train_dataset:
            train_dataset.report_load_errors()

    val_dataset = None
    try:
        val_dataset = TinyViratCustom(data_split='val')

        if len(val_dataset) == 0:
            pass
        else:
            val_dataloader = DataLoader(val_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=cfg.NUM_WORKERS,
                                        pin_memory=True,
                                        collate_fn=collate_filter_none)

            start_time = time.time()
            batch_count = 0
            for batch_data in val_dataloader:
                if batch_data is None:
                    continue
                clips, labels = batch_data
                batch_count += 1
                if batch_count >= 3:
                    break

    except FileNotFoundError as e:
        pass
    except Exception as e:
        pass
    finally:
        if val_dataset:
            val_dataset.report_load_errors()

