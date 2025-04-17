import torch
from torch import nn
import torchvision.models.video as video_models
from torchvision.models.video import R2Plus1D_18_Weights
import os
import config as cfg

def generate_model():
    print(f"Generating torchvision model: {cfg.MODEL_NAME}")
    if cfg.PRETRAINED:
        print(f"Loading pretrained weights: {R2Plus1D_18_Weights.KINETICS400_V1}")
        weights = R2Plus1D_18_Weights.KINETICS400_V1
        model = video_models.r2plus1d_18(weights=weights)
    else:
        print("Loading model with random weights.")
        model = video_models.r2plus1d_18(weights=None)
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        print(f"Original classifier input features: {in_features}")
        model.fc = nn.Linear(in_features, cfg.NUM_CLASSES)
        print(f"Replaced final classifier (fc) with {in_features} input features "
              f"and {cfg.NUM_CLASSES} output classes (from config.py).")
    else:
        print("Warning: Could not find attribute 'fc' or it's not nn.Linear. "
              "Classifier not adapted automatically.")
    return model

def set_parameter_requires_grad(model, backbone_frozen):
    print(f"Setting requires_grad: Backbone frozen = {backbone_frozen}")
    for name, param in model.named_parameters():
        if name.startswith('fc.'):
            param.requires_grad = True
        else:
            param.requires_grad = not backbone_frozen

def get_optimizer_param_groups(model):
    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name.startswith('fc.'):
                classifier_params.append(param)
            else:
                backbone_params.append(param)
    param_groups = [
        {'params': backbone_params, 'name': 'backbone'},
        {'params': classifier_params, 'name': 'classifier'}
    ]
    print(f"Optimizer groups: Backbone ({len(backbone_params)} tensors), "
          f"Classifier ({len(classifier_params)} tensors)")
    return param_groups

def make_data_parallel(model, is_distributed, device):
    if is_distributed:
        print("Using DistributedDataParallel.")
        if device.type == 'cuda' and device.index is not None:
            print(f"  Setting DDP device to: {device}")
            torch.cuda.set_device(device)
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)
        else:
            print(f"  Setting DDP device (CPU or single GPU): {device}")
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model, device_ids=None).cuda()
    elif device.type == 'cuda':
        print("Using single GPU (no parallel wrapper needed).")
        model.to(device)
    else:
        print("Model running on CPU (no parallel wrapper).")
        model.to(device)
    return model

if __name__ == '__main__':
    if cfg.DATA_FOLDER == '/path/to/your/TinyVirat-v2/dataset':
         print("="*50)
         print("!!! WARNING: Please set the DATA_FOLDER path in config.py !!!")
         print("="*50)
    print("--- Generating Model ---")
    model = generate_model()
    print("\n--- Setting requires_grad for Epoch 0 ---")
    set_parameter_requires_grad(model, backbone_frozen=True)
    print("\n--- Checking requires_grad status for Epoch 0 ---")
    for name, param in model.named_parameters():
         if name.startswith("fc."):
              print(f"  {name}: requires_grad={param.requires_grad} (Should be True)")
              assert param.requires_grad is True
         elif name.startswith("stem.0") or name.startswith("layer1.0.conv"):
              print(f"  {name}: requires_grad={param.requires_grad} (Should be False)")
              assert param.requires_grad is False
    print("Requires_grad status verified for frozen epoch.")
    print("\n--- Getting Optimizer Parameter Groups ---")
    param_groups = get_optimizer_param_groups(model)
    print("\n--- Setting requires_grad for Epoch", cfg.FROZEN_EPOCHS, "---")
    set_parameter_requires_grad(model, backbone_frozen=False)
    print("\n--- Checking requires_grad status for Epoch", cfg.FROZEN_EPOCHS, "---")
    for name, param in model.named_parameters():
         if name.startswith("fc."):
              assert param.requires_grad is True
         elif name.startswith("stem.0") or name.startswith("layer1.0.conv"):
              print(f"  {name}: requires_grad={param.requires_grad} (Should be True)")
              assert param.requires_grad is True
    print("Requires_grad status verified for unfrozen epoch.")
    print("\n--- Getting Optimizer Parameter Groups (Unfrozen) ---")
    param_groups_unfrozen = get_optimizer_param_groups(model)
    print("\n--- Setting up Data Parallel ---")
    is_distributed = False
    device = torch.device(cfg.DEVICE)
    model = make_data_parallel(model, is_distributed, device)
    print(f"Model ready on device: {device}")
    try:
        print("\n--- Testing Forward Pass ---")
        B, C, T, H, W = cfg.BATCH_SIZE, 3, cfg.NUM_FRAMES, cfg.INPUT_SIZE, cfg.INPUT_SIZE
        test_B = 2
        dummy_input = torch.randn(test_B, C, T, H, W).to(device)
        print(f"Input shape: {dummy_input.shape}")
        with torch.no_grad():
             output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        assert output.shape == (test_B, cfg.NUM_CLASSES)
        print("Forward pass test successful!")
    except Exception as e:
        print(f"Error during forward pass test: {e}")
        import traceback
        traceback.print_exc()

