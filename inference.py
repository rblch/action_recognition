import torch
import torchvision.transforms as transforms
from torchvision.io import read_video
import numpy as np
import json
import os

import config as cfg
from model import generate_model

class ShortSideResize(object):
    """Resize the shorter side of a video to the given size."""
    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        # video: (C, T, H, W)
        c, t, h, w = video.shape
        if h < w:
            new_h, new_w = self.size, int(self.size * w / h)
        else:
            new_h, new_w = int(self.size * h / w), self.size
        video = torch.nn.functional.interpolate(
            video, size=(new_h, new_w), mode='bilinear', align_corners=False
        )
        return video

class CenterCropVideo(object):
    """Center crop for video tensor (C, T, H, W)."""
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, video):
        c, t, h, w = video.shape
        th, tw = self.size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return video[:, :, i:i+th, j:j+tw]

class NormalizeVideo(object):
    """Normalize video tensor (C, T, H, W) with mean and std."""
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)[:, None, None, None]
        self.std = torch.tensor(std)[:, None, None, None]

    def __call__(self, video):
        return (video - self.mean.to(video.device)) / self.std.to(video.device)

def load_inference_model(model_path, device):
    """Load trained model for inference."""
    original_pretrained_setting = cfg.PRETRAINED
    cfg.PRETRAINED = False
    print("Generating model structure for inference (pretrained weights will be loaded from checkpoint)...")
    model = generate_model()
    cfg.PRETRAINED = original_pretrained_setting # Restore original setting

    print(f"Loading trained weights from checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    if all(k.startswith('module.') for k in state_dict):
        state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Trained model loaded successfully for inference.")
    return model

def get_video_transform(input_size=cfg.INPUT_SIZE):
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]
    return transforms.Compose([
        transforms.Lambda(lambda x: x / 255.0),
        ShortSideResize(input_size),
        CenterCropVideo((input_size, input_size)),
        NormalizeVideo(mean, std)
    ])

def preprocess_video(video_path, num_frames=cfg.NUM_FRAMES, transform=None, device=cfg.DEVICE):
    if transform is None:
        transform = get_video_transform()
    # read_video with output_format="TCHW" returns (T, C, H, W)
    frames, _, info = read_video(video_path, pts_unit='sec', output_format="TCHW")
    total_frames = frames.shape[0] # Time dimension is 0
    indices = torch.linspace(0, total_frames - 1, num_frames, dtype=torch.long)
    # Sample frames along the time dimension (dim 0) -> shape (num_frames, C, H, W)
    sampled_frames = frames[indices, :, :, :]
    # Permute dimensions to (C, T, H, W) as expected by the transforms
    sampled_frames = sampled_frames.permute(1, 0, 2, 3)
    transformed = transform(sampled_frames)
    input_tensor = transformed.unsqueeze(0).to(device)
    return input_tensor

def load_class_map(map_path=os.path.join(cfg.DATA_FOLDER, 'new_class_map.json')):
    if not os.path.exists(map_path):
        return None
    with open(map_path, 'r') as f:
        class_name_to_id = json.load(f)
    id_to_class_name = {v: k for k, v in class_name_to_id.items()}
    return id_to_class_name

def predict_video(model, video_path, class_map, device=cfg.DEVICE, threshold=0.5):
    input_tensor = preprocess_video(video_path, device=device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs).squeeze(0).cpu().numpy()
    results = {}
    for i, prob in enumerate(probs):
        if prob >= threshold:
            results[str(i)] = float(prob)
    return results

