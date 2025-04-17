import torch
import os

PROJECT_NAME = "SurveillanceActionRecognition"
EXPERIMENT_NAME = "R2plus1D_Torchvision_Run3"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

environment =  'local'

if environment == 'colab':
    DATA_FOLDER = '/content/datasets/TinyVIRAT_V2'
    gdrive_base_output = f'/content/drive/MyDrive/{PROJECT_NAME}/outputs'
    os.makedirs(gdrive_base_output, exist_ok=True)
    OUTPUT_DIR = os.path.join(gdrive_base_output, EXPERIMENT_NAME)
elif environment == 'local':
    DATA_FOLDER = 'datasets/TinyVIRAT_V2'
    OUTPUT_DIR = os.path.join("outputs")
else:
    raise ValueError(f"Unknown environment specified: {environment}. Choose 'local' or 'colab'.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 9
CLASS_WEIGHTS_CACHE_FILE = os.path.join(OUTPUT_DIR, 'class_weights_cache.pth')

NUM_FRAMES = 16
SKIP_FRAMES = 1
INPUT_SIZE = 112
BATCH_SIZE = 128
VAL_BATCH_SIZE = 1
NUM_WORKERS = 2

FLIP_PROB = 0.5
GRAYSCALE_PROB = 0.1
DROPOUT_PROB = 0.05
COLOR_JITTER_ENABLE = True
COLOR_JITTER_BRIGHTNESS = 0.4
COLOR_JITTER_CONTRAST = 0.4
COLOR_JITTER_SATURATION = 0.4
COLOR_JITTER_HUE = 0.0

MODEL_NAME = 'r2plus1d_18'
PRETRAINED = True

EPOCHS = 20
FROZEN_EPOCHS = 5
OPTIMIZER = 'AdamW'
LEARNING_RATE_FROZEN = 1e-3
LEARNING_RATE_UNFROZEN = 5e-4
BACKBONE_LR_FACTOR = 0.1
WEIGHT_DECAY = 5e-4
LR_SCHEDULER = 'StepLR'
LR_WARMUP_EPOCHS = 2
LR_STEP_SIZE = 2
LR_GAMMA = 0.6

LOSS_FUNCTION = 'BCEWithLogitsLoss'
USE_CLASS_WEIGHTS = True
USE_AMP = True
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_METRIC = 'val_map'
EARLY_STOPPING_MODE = 'max'
IS_DISTRIBUTED = False
CHECKPOINT_FREQ = 1
SAVE_BEST_ONLY = True

EVAL_FREQ = 1

LOG_FREQ = 50
LOG_FILE = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_log.txt")
CSV_LOG_FILE = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_metrics.csv")

assert os.path.exists(DATA_FOLDER), f"DATA_FOLDER path does not exist: {DATA_FOLDER}"